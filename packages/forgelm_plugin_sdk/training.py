"""Versioned training contracts, independent of the platform and tensor framework."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


STATE_GROUPS = ("weights", "optimizer", "scheduler", "rng", "data")


def continuation_policy(mode: str, *, keep_optimizer: bool = False) -> dict:
    if mode not in {"full", "configured", "weights"}:
        raise ValueError("Unknown continuation mode")
    if keep_optimizer and mode != "configured":
        raise ValueError("keep_optimizer is only valid for configured continuation")
    return {group: "keep" if group == "weights" or mode == "full" or (group == "optimizer" and keep_optimizer) else "reset" for group in STATE_GROUPS}


def validate_training_contract(contract: dict) -> None:
    if not isinstance(contract, dict) or set(contract) != {"checkpoint_format", "state_compatibility", "modes"}:
        raise ValueError("Training requires checkpoint_format, state_compatibility and modes")
    if any(not isinstance(contract[k], str) or not 1 <= len(contract[k]) <= 180 for k in ("checkpoint_format", "state_compatibility")):
        raise ValueError("Invalid checkpoint compatibility identity")
    modes = contract["modes"]
    if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes) or set(modes) - {"full", "configured", "weights"}:
        raise ValueError("Invalid continuation modes")


@dataclass(frozen=True)
class Extension:
    id: str
    semantic_contract: str


@dataclass(frozen=True)
class Backend:
    id: str
    extension: str
    version: str
    distribution: str
    factory: Callable
    supports: Callable  # returns None or an actionable incompatibility reason
    semantic_contract: str = ""
    covers: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()


class BackendRegistry:
    """Only code packaged in the frozen component registers implementations."""
    def __init__(self):
        self._backends = {}
        self._extensions = {}

    def declare(self, extension: Extension):
        if not extension.id or not extension.semantic_contract or extension.id in self._extensions:
            raise ValueError("Invalid or duplicate model extension")
        self._extensions[extension.id] = extension

    def register(self, backend: Backend):
        if backend.id in self._backends:
            raise ValueError(f"Duplicate backend: {backend.id}")
        extension = self._extensions.get(backend.extension)
        if not extension or extension.semantic_contract != backend.semantic_contract:
            raise ValueError("Backend mathematical semantics do not match the model extension")
        if any(name not in self._extensions for name in backend.covers):
            raise ValueError("Backend covers an undeclared model extension")
        self._backends[backend.id] = backend

    def resolve(self, extension: str, backend_id: str, environment: dict, config: dict):
        backend = self._backends.get(backend_id)
        if backend is None or backend.extension != extension:
            raise ValueError(f"Unknown {extension} backend: {backend_id}")
        reason = backend.supports(environment, config)
        if reason:
            raise ValueError(f"{backend_id}: {reason}")
        return backend, {"extension": extension, "id": backend.id, "version": backend.version,
                         "distribution": backend.distribution, "semantic_contract": backend.semantic_contract,
                         "config": config, "dispatch": "selected"}

    def resolve_many(self, bindings: list[dict], environment: dict):
        """Reject overlapping ownership and dependency cycles before constructing code."""
        selected, owners = {}, {}
        for binding in bindings:
            backend, receipt = self.resolve(binding["extension"], binding["backend"], environment, binding.get("config", {}))
            if backend.id in selected:
                raise ValueError("A backend may only be selected once")
            selected[backend.id] = (backend, receipt)
            for extension in {backend.extension, *backend.covers}:
                if extension in owners:
                    raise ValueError(f"Overlapping backend ownership: {extension}")
                owners[extension] = backend.id
        pending, ordered = set(selected), []
        while pending:
            ready = sorted(name for name in pending if set(selected[name][0].requires) <= set(ordered))
            if not ready:
                raise ValueError("Missing backend dependency or dependency cycle")
            ordered.extend(ready)
            pending.difference_update(ready)
        return [selected[name] for name in ordered]
