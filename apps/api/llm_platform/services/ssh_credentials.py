"""Local SSH keys. Passwords are used once to authorize a dedicated public key."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from uuid import uuid4

from ..config import get_settings


def credential_directory() -> Path:
    url = get_settings().database_url
    base = (
        Path(url.removeprefix("sqlite:///")).resolve().parent
        if url.startswith("sqlite:///")
        else Path(".llm-platform").resolve()
    )
    directory = base / "ssh-credentials"
    if get_settings().storage_root:
        from llm_platform_core.storage_layout import platform_state_root
        target = platform_state_root(get_settings().storage_root) / "config" / "ssh-credentials"
        # Existing keys remain readable until the audited relocation completes.
        directory = directory if directory.exists() and not target.exists() else target
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def protect_path(path: Path) -> None:
    if os.name == "nt":
        # Replace the DACL: removing inheritance alone leaves explicit logon-session
        # ACEs on some Windows profiles, which OpenSSH correctly refuses.
        script = r"""
$ErrorActionPreference = 'Stop'
$path = $env:FORGELM_ACL_TARGET
$isDirectory = [System.IO.Directory]::Exists($path)
$acl = if ($isDirectory) { [System.IO.Directory]::GetAccessControl($path) } else { [System.IO.File]::GetAccessControl($path) }
$acl.SetAccessRuleProtection($true, $false)
foreach ($rule in @($acl.Access)) { $acl.RemoveAccessRuleSpecific($rule) }
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', 'Allow')
$acl.SetAccessRule($rule)
if ($isDirectory) { [System.IO.Directory]::SetAccessControl($path, $acl) } else { [System.IO.File]::SetAccessControl($path, $acl) }
"""
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "FORGELM_ACL_TARGET": str(path.resolve())},
            check=True, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        path.chmod(0o700 if path.is_dir() else 0o600)


def managed_key_path(reference: str) -> Path | None:
    if re.fullmatch(r"managed_[a-f0-9]{32}", reference):
        return credential_directory() / reference
    return None


def prepare_credential(payload, host_key: str) -> str:
    if payload.credential_ref:
        return payload.credential_ref
    import paramiko

    reference = f"managed_{uuid4().hex}"
    path = credential_directory() / reference
    protect_path(path.parent)
    path.touch(mode=0o600)
    protect_path(path)
    try:
        if payload.private_key_path:
            source = Path(payload.private_key_path).expanduser().resolve()
            if not source.is_file() or source.stat().st_size > 65536:
                raise ValueError("私钥文件不存在或格式不合法")
            path.write_bytes(source.read_bytes())
        else:
            key = paramiko.RSAKey.generate(3072)
            key.write_private_key_file(str(path))
            with paramiko.SSHClient() as client:
                entry = paramiko.hostkeys.HostKeyEntry.from_line(host_key)
                if entry is None:
                    raise ValueError("无效的 SSH 主机公钥")
                hostname = (
                    payload.ssh_host
                    if payload.ssh_port == 22
                    else f"[{payload.ssh_host}]:{payload.ssh_port}"
                )
                client.get_host_keys().add(hostname, entry.key.get_name(), entry.key)
                client.connect(
                    payload.ssh_host,
                    port=payload.ssh_port,
                    username=payload.ssh_user,
                    password=payload.ssh_password.get_secret_value(),
                    allow_agent=False,
                    look_for_keys=False,
                    timeout=15,
                    banner_timeout=15,
                    auth_timeout=15,
                )
                public_key = f"{key.get_name()} {key.get_base64()} forgelm-{reference}"
                command = (
                    'umask 077; mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh" && printf '
                    + shlex.quote("\n%s\n")
                    + " "
                    + shlex.quote(public_key)
                    + ' >> "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"'
                )
                _, stdout, _ = client.exec_command(command, timeout=20)
                if stdout.channel.recv_exit_status() != 0:
                    raise ValueError("无法安装平台专用 SSH 公钥")
        # Verify noninteractive key login now, before accepting a reconnectable configuration.
        with paramiko.SSHClient() as client:
            entry = paramiko.hostkeys.HostKeyEntry.from_line(host_key)
            hostname = (
                payload.ssh_host
                if payload.ssh_port == 22
                else f"[{payload.ssh_host}]:{payload.ssh_port}"
            )
            client.get_host_keys().add(hostname, entry.key.get_name(), entry.key)
            client.connect(
                payload.ssh_host,
                port=payload.ssh_port,
                username=payload.ssh_user,
                key_filename=str(path),
                allow_agent=False,
                look_for_keys=False,
                timeout=15,
                banner_timeout=15,
                auth_timeout=15,
            )
        return reference
    except Exception as exc:
        path.unlink(missing_ok=True)
        # Do not echo credentials, paramiko exceptions, or supplied key contents.
        raise ValueError("SSH 凭据验证失败，请检查密码、私钥和主机是否允许公钥登录") from exc
