"""Static, execution-free validation shared by the control plane and Worker."""
from __future__ import annotations

import configparser
import hashlib
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ID = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_PACKAGE = 8 * 1024**3
MAX_EXPANDED = 16 * 1024**3
MAX_METADATA = 4 * 1024**2


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Unsafe package path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", "..", ""} for part in value.split("/")):
        raise ValueError("Unsafe package path")
    if any(part.rstrip(" .") != part or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))} for part in path.parts):
        raise ValueError("Unsafe cross-platform package path")
    return value


def json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    if len(raw) > MAX_METADATA:
        raise ValueError("Metadata exceeds size limit")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def archive_json(archive, name):
    if archive.getinfo(name).file_size > MAX_METADATA:
        raise ValueError("Metadata exceeds size limit")
    return json_object(archive.read(name))


def fields(value: dict, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"Invalid fields; required={sorted(required)}, allowed={sorted(required | optional)}")


def identity(value: str) -> str:
    if not isinstance(value, str) or len(value) > 180 or not ID.fullmatch(value):
        raise ValueError("Invalid namespaced identity")
    return value


def version_key(value: str) -> tuple:
    match = VERSION.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("Version must be SemVer")
    pre = match[4]
    if match[5] and any(not p for p in match[5].split(".")):
        raise ValueError("Invalid SemVer build metadata")
    parts = pre.split(".") if pre else []
    if any(not p or p.isdigit() and len(p) > 1 and p.startswith("0") for p in parts):
        raise ValueError("Invalid SemVer prerelease")
    return (*map(int, match.group(1, 2, 3)), not bool(pre), tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts))


def accepts_range(value: str, expression: str) -> bool:
    current = version_key(value)
    for condition in expression.split(","):
        match = re.fullmatch(r"(>=|<=|==|>|<)(\d+\.\d+\.\d+)", condition.strip())
        if not match:
            raise ValueError("Unsupported version range")
        other = version_key(match[2])
        if not {">=": current >= other, "<=": current <= other, "==": current == other, ">": current > other, "<": current < other}[match[1]]:
            return False
    return True


def inspect_package(path: Path) -> dict:
    if path.stat().st_size > MAX_PACKAGE:
        raise ValueError("Suite package exceeds 8 GiB")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 20000 or sum(i.file_size for i in infos) > MAX_EXPANDED:
            raise ValueError("Package expansion limit exceeded")
        names = set()
        folded = set()
        for info in infos:
            name = safe_path(info.filename)
            if info.is_dir() or name.casefold() in folded or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Duplicate, directory or symlink package entry")
            names.add(name)
            folded.add(name.casefold())
        checks = archive_json(archive, "checksums.json")
        if set(checks) != names - {"checksums.json"}:
            raise ValueError("Incomplete package checksums")
        for name, digest in checks.items():
            if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
                raise ValueError("Invalid checksum")
            with archive.open(name) as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(f"Checksum mismatch: {name}")
        manifest = archive_json(archive, "suite.json")
        fields(manifest, {"schema", "id", "publisher", "name", "version", "description", "license", "compatibility", "targets", "components", "executionRequirements"})
        if manifest["schema"] != "forgelm.suite-package.v1":
            raise ValueError("Unsupported suite schema")
        if any(not isinstance(manifest[k], str) or not manifest[k] or len(manifest[k]) > 10000 for k in ("name", "description", "license")):
            raise ValueError("Invalid package display metadata")
        identity(manifest["id"])
        identity(manifest["publisher"])
        if not manifest["id"].startswith(manifest["publisher"] + "."):
            raise ValueError("Publisher namespace mismatch")
        version_key(manifest["version"])
        compat = manifest["compatibility"]
        fields(compat, {"pluginHost", "pluginApi", "nodeSpec"})
        if not accepts_range("1.1.0", compat["pluginHost"]) or compat["pluginApi"] != "1.0" or compat["nodeSpec"] != "1.0":
            raise ValueError("Incompatible plugin host/API")
        reqs = manifest["executionRequirements"]
        fields(reqs, {"network", "gpu", "secrets", "externalWrites"})
        if not isinstance(reqs["network"], bool) or not isinstance(reqs["gpu"], bool) or reqs["secrets"] != [] or reqs["externalWrites"] != []:
            raise ValueError("Unsupported execution requirements (secret/external-write brokers are not enabled)")
        components = manifest["components"]
        if not isinstance(components, list) or not 1 <= len(components) <= 100:
            raise ValueError("Suite must contain 1–100 components")
        component_ids, node_ids, nodes = set(), set(), []
        for component in components:
            fields(component, {"id", "implementationVersion", "distribution", "entryPoint", "runtimeGroup", "nodes"})
            cid = identity(component["id"])
            identity(component["runtimeGroup"])
            version_key(component["implementationVersion"])
            if cid in component_ids:
                raise ValueError("Duplicate component")
            component_ids.add(cid)
            fields(component["entryPoint"], {"group", "name"})
            if component["entryPoint"]["group"] != "forgelm.components.v1":
                raise ValueError("Invalid entry point group")
            identity(component["entryPoint"]["name"])
            if not isinstance(component["distribution"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component["distribution"]):
                raise ValueError("Invalid distribution name")
            if not isinstance(component["nodes"], list) or not component["nodes"]:
                raise ValueError("Component must contribute nodes")
            for node_path in component["nodes"]:
                node = archive_json(archive, safe_path(node_path))
                fields(node, {"schema", "id", "displayName", "capability", "component", "execution", "inputs", "outputs", "configSchema", "resources", "controls", "cache"}, {"training"})
                identity(node["id"])
                if node["schema"] != "forgelm.node-spec.v1" or node["component"] != cid or node["id"] in node_ids:
                    raise ValueError("Invalid or duplicate node identity")
                node_ids.add(node["id"])
                if node["execution"] not in ({"kind": "task", "protocol": "forgelm.plugin.v1"}, {"kind": "service", "protocol": "forgelm.serve.v1"}):
                    raise ValueError("Unsupported execution protocol")
                for ports in (node["inputs"], node["outputs"]):
                    if not isinstance(ports, list) or len(ports) > 100:
                        raise ValueError("Invalid port declarations")
                    port_names = set()
                    for port in ports:
                        fields(port, {"name", "type", "format", "required"})
                        if not isinstance(port["format"], str) or not 1 <= len(port["format"]) <= 200:
                            raise ValueError("Invalid artifact format")
                        if port["type"] != "artifact" or identity(port["name"]) in port_names or not isinstance(port["required"], bool):
                            raise ValueError("Invalid port")
                        port_names.add(port["name"])
                fields(node["resources"], {"cpu", "memoryMiB", "gpu"})
                if any(type(v) is not int or v < 0 for v in node["resources"].values()):
                    raise ValueError("Resources must be nonnegative integers")
                fields(node["controls"], {"cancel", "checkpoint", "resume"})
                if any(type(v) is not bool for v in node["controls"].values()):
                    raise ValueError("Invalid controls")
                if node["controls"]["checkpoint"] or node["controls"]["resume"]:
                    from .training import validate_training_contract
                    validate_training_contract(node.get("training"))
                    if node["capability"] != "train" or node["execution"]["kind"] != "task" or not all(node["controls"][k] for k in ("checkpoint", "resume")) or accepts_range("1.0.0", compat["pluginHost"]):
                        raise ValueError("Training checkpoint/resume requires a train task and plugin host >=1.1")
                elif "training" in node:
                    raise ValueError("Training contract requires checkpoint/resume controls")
                if node["cache"] != {"eligible": False}:
                    raise ValueError("Cross-run caching is not enabled for external components")
                config = archive_json(archive, safe_path(node["configSchema"]))
                from .validation import check_schema
                check_schema(config)
                nodes.append({**node, "config_schema": config, "digest": checks[node_path]})
        targets = manifest["targets"]
        if not isinstance(targets, list) or not 1 <= len(targets) <= 32:
            raise ValueError("Invalid target count")
        target_ids, locks, target_sizes = set(), {}, []
        for target in targets:
            installed_bytes = 0
            fields(target, {"id", "os", "arch", "python", "accelerator", "runtimeGroups"}, {"cuda", "minDriver"})
            if identity(target["id"]) in target_ids or target["os"] not in {"linux", "windows", "darwin"} or target["arch"] not in {"x86_64", "aarch64"}:
                raise ValueError("Invalid target")
            target_ids.add(target["id"])
            if not re.fullmatch(r"3\.(11|12|13|14)", target["python"]) or target["accelerator"] not in {"cpu", "cuda"}:
                raise ValueError("Unsupported runtime")
            if target["accelerator"] == "cuda" and not all(target.get(k) for k in ("cuda", "minDriver")):
                raise ValueError("CUDA target requires runtime and minimum driver")
            if not isinstance(target["runtimeGroups"], dict) or not target["runtimeGroups"]:
                raise ValueError("Invalid runtime group map")
            for group, spec in target["runtimeGroups"].items():
                identity(group)
                fields(spec, {"lock"})
                lock = archive_json(archive, safe_path(spec["lock"]))
                fields(lock, {"schema", "packages"})
                if lock["schema"] != "forgelm.wheel-lock.v1" or not isinstance(lock["packages"], list) or not lock["packages"]:
                    raise ValueError("Invalid dependency lock")
                distributions, distribution_versions = {}, {}
                for wheel in lock["packages"]:
                    fields(wheel, {"name", "version", "path", "sha256"})
                    normalized = re.sub(r"[-_.]+", "-", wheel["name"]).lower()
                    wheelpath = safe_path(wheel["path"])
                    if normalized in distributions or not wheelpath.endswith(".whl") or checks.get(wheelpath) != wheel["sha256"]:
                        raise ValueError("Invalid or duplicate locked wheel")
                    # CUDA wheels can exceed a GiB; keep validation memory bounded.
                    with tempfile.TemporaryFile() as wheel_file:
                        with archive.open(wheelpath) as source:
                            shutil.copyfileobj(source, wheel_file, length=1024 * 1024)
                        wheel_file.seek(0)
                        with zipfile.ZipFile(wheel_file) as whl:
                            installed_bytes += sum(item.file_size for item in whl.infolist())
                            if installed_bytes > 64 * 1024**3:
                                raise ValueError("Wheel installation exceeds 64 GiB limit")
                            metadata = [n for n in whl.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
                            if len(metadata) != 1 or whl.getinfo(metadata[0]).file_size > MAX_METADATA:
                                raise ValueError("Invalid wheel metadata")
                            from email.parser import Parser
                            meta = Parser().parsestr(whl.read(metadata[0]).decode())
                            if re.sub(r"[-_.]+", "-", meta["Name"]).lower() != normalized or meta["Version"] != wheel["version"]:
                                raise ValueError("Wheel identity mismatch")
                            entry = metadata[0].removesuffix("METADATA") + "entry_points.txt"
                            parser = configparser.ConfigParser(interpolation=None)
                            parser.optionxform = str
                            if entry in whl.namelist():
                                if whl.getinfo(entry).file_size > MAX_METADATA:
                                    raise ValueError("Wheel entry points exceed size limit")
                                parser.read_string(whl.read(entry).decode())
                            distributions[normalized] = parser
                            distribution_versions[normalized] = wheel["version"]
                sdk_versions = {"1.1.0"} if any(n.get("training") for n in nodes) else {"1.0.0", "1.1.0"}
                if distribution_versions.get("forgelm-plugin-sdk") not in sdk_versions:
                    raise ValueError("Lock must include the component SDK/Runner wheel")
                for component in components:
                    if component["runtimeGroup"] != group:
                        continue
                    dist = re.sub(r"[-_.]+", "-", component["distribution"]).lower()
                    parser = distributions.get(dist)
                    if parser is None or distribution_versions[dist] != component["implementationVersion"] or not parser.has_option("forgelm.components.v1", component["entryPoint"]["name"]):
                        raise ValueError("Component entry point missing from locked distribution")
                locks[f"{target['id']}/{group}"] = {**lock, "digest": checks[spec["lock"]]}
            if any(c["runtimeGroup"] not in target["runtimeGroups"] for c in components):
                raise ValueError("Target does not cover every component")
            target_sizes.append(installed_bytes)
        return {"manifest": manifest, "nodes": nodes, "locks": locks, "digest": file_hash(path),
                "size_bytes": path.stat().st_size, "expanded_bytes": sum(i.file_size for i in infos), "installed_bytes": max(target_sizes)}


def extract_package(path: Path, destination: Path) -> dict:
    package = inspect_package(path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = destination / safe_path(info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                import shutil
                shutil.copyfileobj(source, output)
    return package
