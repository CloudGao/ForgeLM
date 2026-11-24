"""Local developer tooling. Build and resolution are delegated to pip."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

from .package import accepts_range, file_hash, identity, inspect_package, safe_path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def init(path: Path, suite_id: str):
    identity(suite_id)
    if "." not in suite_id:
        raise ValueError("Use publisher.name")
    path.mkdir(parents=True, exist_ok=False)
    dist = suite_id.replace(".", "-")
    module = suite_id.replace(".", "_").replace("-", "_")
    target = f"{platform.system().lower()}-{platform.machine().lower().replace('amd64', 'x86_64')}-cp{sys.version_info.major}{sys.version_info.minor}-cpu"
    write_json(path / "suite.json", {
        "schema": "forgelm.suite-package.v1", "id": suite_id, "publisher": suite_id.split(".")[0], "name": suite_id,
        "version": "1.0.0", "description": "Text normalization component", "license": "UNLICENSED",
        "compatibility": {"pluginHost": ">=1.0.0,<2.0.0", "pluginApi": "1.0", "nodeSpec": "1.0"},
        "targets": [{"id": target, "os": platform.system().lower(), "arch": platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64"), "python": f"{sys.version_info.major}.{sys.version_info.minor}", "accelerator": "cpu", "runtimeGroups": {"default": {"lock": f"locks/{target}.json"}}}],
        "components": [{"id": "normalize", "implementationVersion": "1.0.0", "distribution": dist, "entryPoint": {"group": "forgelm.components.v1", "name": "normalize"}, "runtimeGroup": "default", "nodes": ["nodes/normalize.json"]}],
        "executionRequirements": {"network": False, "gpu": False, "secrets": [], "externalWrites": []},
    })
    write_json(path / "nodes/normalize.json", {
        "schema": "forgelm.node-spec.v1", "id": suite_id + ".normalize", "displayName": "文本标准化", "capability": "transform", "component": "normalize",
        "execution": {"kind": "task", "protocol": "forgelm.plugin.v1"},
        "inputs": [{"name": "source", "type": "artifact", "format": "text/plain", "required": True}],
        "outputs": [{"name": "result", "type": "artifact", "format": "text/plain", "required": True}],
        "configSchema": "schemas/normalize.json", "resources": {"cpu": 1, "memoryMiB": 256, "gpu": 0},
        "controls": {"cancel": True, "checkpoint": False, "resume": False}, "cache": {"eligible": False},
    })
    write_json(path / "schemas/normalize.json", {"type": "object", "additionalProperties": False, "properties": {"lowercase": {"type": "boolean", "default": False, "title": "转换为小写"}}})
    source = path / "src" / module
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "normalize.py").write_text('''from forgelm_plugin_sdk import Output, Result

class NormalizePlugin:
    def probe(self, context):
        return {"ready": True, "implementation_version": "1.0.0", "issues": []}

    def validate(self, request, context):
        return []

    def run(self, request, context):
        output = context.output_dir / "normalized.txt"
        with context.input_path("source").open(encoding="utf-8") as source, output.open("w", encoding="utf-8", newline="\\n") as target:
            while True:
                line = source.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024:
                    raise ValueError("Input line exceeds 1 MiB characters")
                context.raise_if_cancelled()
                value = line.strip()
                target.write((value.lower() if request.config["lowercase"] else value) + "\\n")
        return Result(outputs=[Output("result", output, "text/plain")])
''', encoding="utf-8")
    (path / "pyproject.toml").write_text(f'''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "{dist}"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = ["forgelm-plugin-sdk==1.1.0"]
[project.entry-points."forgelm.components.v1"]
normalize = "{module}.normalize:NormalizePlugin"
[tool.setuptools.packages.find]
where = ["src"]
''', encoding="utf-8")
    (path / "README.md").write_text("# " + suite_id + "\n\nGenerated local development suite. Build, lock and validate before importing.\n", encoding="utf-8")
    (path / "LICENSE").write_text("Set the actual license before distributing this suite.\n", encoding="utf-8")


def lock(root: Path, target_id: str, sdk_source: Path | None = None):
    manifest = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    target = next(t for t in manifest["targets"] if t["id"] == target_id)
    current_os = platform.system().lower()
    arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    if target["os"] != current_os or target["arch"] != arch or target["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError("Build locks on a matching target interpreter/OS/architecture")
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    if sdk_source:
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(sdk_source)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(root)], check=True)
    sdk_requirement = "forgelm-plugin-sdk>=1.0.0,<1.2.0" if accepts_range("1.0.0", manifest["compatibility"]["pluginHost"]) else "forgelm-plugin-sdk==1.1.0"
    for group, spec in target["runtimeGroups"].items():
        selected = [c for c in manifest["components"] if c["runtimeGroup"] == group]
        requirements = [f"{c['distribution']}=={c['implementationVersion']}" for c in selected]
        with tempfile.TemporaryDirectory(prefix="forgelm-resolve-") as temp:
            subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--find-links", str(wheels), "--dest", temp, sdk_requirement, *requirements], check=True)
            import shutil
            packages = []
            for item in sorted(Path(temp).glob("*.whl")):
                with zipfile.ZipFile(item) as whl:
                    meta_path = next(n for n in whl.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA"))
                    meta = Parser().parsestr(whl.read(meta_path).decode())
                destination = wheels / item.name
                shutil.copyfile(item, destination)
                packages.append({"name": meta["Name"], "version": meta["Version"], "path": f"wheels/{item.name}", "sha256": file_hash(destination)})
            write_json(root / spec["lock"], {"schema": "forgelm.wheel-lock.v1", "packages": packages})


def pack(root: Path, output: Path) -> Path:
    manifest = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    names = {"suite.json", "README.md", "LICENSE"}
    for component in manifest["components"]:
        for name in component["nodes"]:
            names.add(safe_path(name))
            node = json.loads((root / name).read_text(encoding="utf-8"))
            names.add(safe_path(node["configSchema"]))
    for target in manifest["targets"]:
        for spec in target["runtimeGroups"].values():
            name = safe_path(spec["lock"])
            names.add(name)
            frozen = json.loads((root / name).read_text(encoding="utf-8"))
            names.update(safe_path(p["path"]) for p in frozen["packages"])
    checks = {}
    for name in names:
        source = root / name
        source.resolve().relative_to(root.resolve())
        if source.is_symlink():
            raise ValueError("Links are not allowed in a suite")
        checks[name] = file_hash(source)
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"{manifest['id']}-{manifest['version']}.flsuite"
    with tempfile.TemporaryDirectory(prefix="forgelm-pack-") as temp:
        candidate = Path(temp) / "candidate.flsuite"
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(names | {"checksums.json"}):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED if name.endswith(".whl") else zipfile.ZIP_DEFLATED
                if name == "checksums.json":
                    archive.writestr(info, json.dumps(checks, sort_keys=True).encode())
                else:
                    info.file_size = (root / name).stat().st_size
                    with (root / name).open("rb") as source, archive.open(info, "w") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
        inspect_package(candidate)
        if result.exists() and file_hash(result) != file_hash(candidate):
            raise ValueError("Same version output already exists with different content; bump the suite version")
        shutil.copyfile(candidate, result)
    return result


def init_adamw_backend(path: Path, backend_id: str, distribution: str):
    identity(backend_id)
    if not distribution or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in distribution):
        raise ValueError("Invalid distribution")
    source = f'''"""Register this backend inside the suite training component, then publish a new version."""
from importlib.metadata import version
from forgelm_plugin_sdk.training import Backend


def factory(parameters, lr):
    import torch
    # Replace this function with your implementation; preserve AdamW semantics.
    return torch.optim.AdamW(parameters, lr=lr, foreach=True, fused=False)


def supports(environment, config):
    return None if environment.get("device") in {{"cpu", "cuda"}} else "Requires CPU or CUDA"


backend = Backend({backend_id!r}, "optimizer.adamw", version({distribution!r}), {distribution!r}, factory, supports, semantic_contract="adamw-decoupled-fp32-v1")
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(source)


def main():
    parser = argparse.ArgumentParser(prog="forgelm-plugin")
    commands = parser.add_subparsers(dest="command", required=True)
    initial = commands.add_parser("init")
    initial.add_argument("id")
    initial.add_argument("--directory")
    backend = commands.add_parser("init-adamw-backend")
    backend.add_argument("id")
    backend.add_argument("--distribution", required=True)
    backend.add_argument("--output", type=Path, required=True)
    bundle = commands.add_parser("pack-training-inputs")
    bundle.add_argument("--task-spec", type=Path, required=True)
    bundle.add_argument("--model-config", type=Path, required=True)
    bundle.add_argument("--validation-file", type=Path)
    bundle.add_argument("--output", type=Path, required=True)
    for name in ("inspect", "validate"):
        item = commands.add_parser(name)
        item.add_argument("path", type=Path)
    item = commands.add_parser("lock")
    item.add_argument("--root", type=Path, default=Path.cwd())
    item.add_argument("--target", required=True)
    item.add_argument("--sdk-source", type=Path)
    item = commands.add_parser("pack")
    item.add_argument("--root", type=Path, default=Path.cwd())
    item.add_argument("--output", type=Path, default=Path("dist"))
    item = commands.add_parser("test")
    item.add_argument("path", type=Path)
    item.add_argument("--contract", action="store_true", required=True)
    item.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        init(Path(args.directory or args.id), args.id)
    elif args.command == "init-adamw-backend":
        init_adamw_backend(args.output, args.id, args.distribution)
    elif args.command == "pack-training-inputs":
        from .training_bundle import pack_training_bundle
        print(pack_training_bundle(args.task_spec, args.model_config, args.output, args.validation_file))
    elif args.command == "lock":
        lock(args.root.resolve(), args.target, args.sdk_source)
    elif args.command == "pack":
        print(pack(args.root.resolve(), args.output.resolve()))
    elif args.command == "test":
        from .contract import test_contract
        print(json.dumps(test_contract(args.path.resolve(), args.fixtures.resolve() if args.fixtures else None), indent=2))
    else:
        if args.path.is_dir():
            with tempfile.TemporaryDirectory() as temp:
                result = inspect_package(pack(args.path.resolve(), Path(temp)))
        else:
            result = inspect_package(args.path)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
