from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LaunchSpec:
    executable: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    ports: tuple[int, ...] = ()


@dataclass(frozen=True)
class RuntimeHandle:
    runtime_id: str
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str
    field: str | None = None


class EngineAdapter(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def validate(
        self, task_spec: dict[str, Any], runtime_context: dict[str, Any]
    ) -> list[ValidationIssue]: ...

    def prepare(
        self,
        task_spec: dict[str, Any],
        paths: dict[str, Path],
        resources: dict[str, Any],
    ) -> LaunchSpec: ...

    def parse_event(self, raw_event: str) -> dict[str, Any] | None: ...

    def discover_outputs(self, paths: dict[str, Path]) -> list[dict[str, Any]]: ...


class LauncherAdapter(Protocol):
    def validate(
        self, launch_spec: LaunchSpec, allocation: dict[str, Any]
    ) -> list[ValidationIssue]: ...

    def launch(self, launch_spec: LaunchSpec, allocation: dict[str, Any]) -> RuntimeHandle: ...

    def inspect(self, handle: RuntimeHandle) -> str: ...

    def control(self, handle: RuntimeHandle, action: str) -> dict[str, Any]: ...

    def cleanup(self, handle: RuntimeHandle) -> None: ...
