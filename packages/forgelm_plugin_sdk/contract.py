"""Run trusted developer contract fixtures inside fresh offline environments."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from .package import extract_package


def test_contract(archive: Path, fixtures: Path | None = None):
    with tempfile.TemporaryDirectory(prefix="forgelm-contract-") as temporary:
        root = Path(temporary)
        package = extract_package(archive, root / "package")
        arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
        target = next((t for t in package["manifest"]["targets"] if (t["os"], t["arch"], t["python"]) == (platform.system().lower(), arch, f"{sys.version_info.major}.{sys.version_info.minor}")), None)
        if not target:
            raise ValueError("No compatible target for this contract test host")
        environments = {}
        clean_env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG"}}
        clean_env.update(PYTHONPATH="", PYTHONNOUSERSITE="1", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        for group in target["runtimeGroups"]:
            environment = root / group
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, env=clean_env)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            lock = package["locks"][target["id"] + "/" + group]
            requirements = root / (group + ".txt")
            requirements.write_text("\n".join(f"{(root / 'package' / p['path']).as_uri()} --hash=sha256:{p['sha256']}" for p in lock["packages"]), encoding="utf-8")
            subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes", "-r", str(requirements)], check=True, env=clean_env, capture_output=True)
            subprocess.run([str(python), "-m", "pip", "check"], check=True, env=clean_env, capture_output=True)
            environments[group] = python
        report = []
        cases = json.loads(fixtures.read_text(encoding="utf-8")) if fixtures else []
        if not isinstance(cases, list):
            raise ValueError("Fixtures must be an array")
        for component in package["manifest"]["components"]:
            nodes = [n for n in package["nodes"] if n["component"] == component["id"]]
            requests = [{"mode": "probe", "execution_kinds": sorted({n["execution"]["kind"] for n in nodes})}]
            for case in cases:
                node = next((n for n in nodes if n["id"] == case["node_id"]), None)
                if not node:
                    continue
                if node["execution"]["kind"] != "task":
                    raise ValueError("Service fixtures use the service HTTP integration contract")
                inputs = {name: {"path": str((fixtures.parent / value).resolve())} for name, value in case.get("inputs", {}).items()}
                requests.append({"node": node, "config": case.get("config", {}), "inputs": inputs})
            for index, request in enumerate(requests):
                raw = {**request, "component": component, "output_dir": str(root / "outputs" / component["id"] / str(index)), "run_id": "contract"}
                path = root / "request.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                result = subprocess.run([str(environments[component["runtimeGroup"]]), "-m", "forgelm_plugin_sdk.runner", "--request", str(path)], input="", capture_output=True, text=True, env=clean_env, timeout=120)
                if result.returncode:
                    raise ValueError(f"Contract failed for {component['id']}: {result.stdout}\n{result.stderr}")
                events = [json.loads(line) for line in result.stdout.splitlines()]
                if sum(event.get("event") == "result" for event in events) != 1 or any(event.get("seq") != i + 1 for i, event in enumerate(events)):
                    raise ValueError("Invalid result/event sequence")
                report.append({"component": component["id"], "kind": request.get("mode", "task"), "passed": True})
        return report
