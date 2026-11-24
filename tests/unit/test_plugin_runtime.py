from __future__ import annotations

import json
import sys

import pytest
from llm_platform_worker.plugin_runners.llm_compressor import _recipe
from llm_platform_worker.plugin_runtime import (
    InstalledPluginAdapter,
    PluginManagerAdapter,
    discover_plugin_capabilities,
    installation_directory,
    venv_python,
)


def test_discover_plugin_capabilities_only_accepts_valid_catalog_manifest(tmp_path) -> None:
    root = tmp_path / "plugins"
    target = installation_directory(
        root, "plugin.llm-compressor.quantization", "0.1.0"
    )
    python = venv_python(target)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    (target / "installation.json").write_text(
        json.dumps(
            {
                "plugin_id": "plugin.llm-compressor.quantization",
                "version": "0.1.0",
                "entrypoint": "plugin.llm-compressor.quantization",
                "runner": "llm_compressor",
                "isolation": "venv",
                "requirements": ["llmcompressor==0.10.0"],
                "python": str(python),
            }
        ),
        encoding="utf-8",
    )
    assert discover_plugin_capabilities(root) == [
        "plugin.llm-compressor.quantization",
        "plugin.llm-compressor.quantization@0.1.0",
    ]


def test_plugin_paths_reject_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        installation_directory(tmp_path, "../escape", "0.1.0")




def test_llm_compressor_recipe_builds_only_supported_presets() -> None:
    class Modifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    rtn = _recipe(
        {
            "algorithm": "RTN",
            "scheme": "FP8_DYNAMIC",
            "targets": ["Linear"],
            "ignore": ["lm_head"],
        },
        Modifier,
        Modifier,
    )
    assert rtn.kwargs["scheme"] == "FP8_DYNAMIC"
    gptq = _recipe(
        {
            "algorithm": "GPTQ",
            "scheme": "W4A16",
            "targets": ["Linear"],
            "block_size": 64,
            "dampening_frac": 0.02,
        },
        Modifier,
        Modifier,
    )
    assert gptq.kwargs["block_size"] == 64
    with pytest.raises(ValueError, match="only supports"):
        _recipe(
            {"algorithm": "RTN", "scheme": "W4A16", "targets": ["Linear"]},
            Modifier,
            Modifier,
        )


@pytest.mark.asyncio
async def test_installed_adapter_rejects_unknown_plugin_without_starting_process(
    tmp_path,
) -> None:
    adapter = InstalledPluginAdapter(tmp_path)

    async def emit(_event, _payload):
        return "running"

    with pytest.raises(ValueError, match="unsupported"):
        await adapter.run(
            {
                "payload": {
                    "plugin": {
                        "id": "plugin.unknown.quantization",
                        "version": "0.1.0",
                        "entrypoint": "plugin.unknown.quantization",
                    }
                }
            },
            emit,
        )
