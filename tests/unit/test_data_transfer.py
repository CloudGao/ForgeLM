from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from llm_platform_worker.adapters.data_transfer import (
    DEFAULT_DATA_PATTERNS,
    ProcessInterrupted,
    ScriptTransferRunner,
)


async def collect_event(events: list[tuple[str, dict]], event_type: str, payload: dict) -> str:
    events.append((event_type, payload))
    return "running"


@pytest.mark.asyncio
async def test_official_huggingface_source_drives_hf_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ScriptTransferRunner(tmp_path / ".llm-platform" / "runs" / "task" / "attempt")
    captured: dict = {}

    async def fake_run(args: list[str], **kwargs: object) -> None:
        captured.update(args=args, **kwargs)

    monkeypatch.setattr(runner, "_resolve_hf", lambda: "hf-test")
    monkeypatch.setattr(runner, "_run_process", fake_run)
    result = await runner.download_huggingface(
        "org/corpus",
        {
            "hf_endpoint": "https://huggingface.co",
            "revision": "v2",
            "include_patterns": ["data/*.parquet", "meta.json"],
            "exclude_patterns": ["data/debug-*"],
            "max_workers": 4,
        },
        tmp_path / "dataset",
        lambda event_type, payload: collect_event([], event_type, payload),
        None,
    )

    args = captured["args"]
    assert args[:3] == ["hf-test", "download", "org/corpus"]
    assert args[args.index("--include") + 1 :] == [
        "data/*.parquet",
        "meta.json",
        "--exclude",
        "data/debug-*",
    ]
    assert args[args.index("--max-workers") + 1] == "4"
    assert captured["env"]["HF_ENDPOINT"] == "https://huggingface.co"
    assert result["tool"] == "hf download"


@pytest.mark.asyncio
async def test_hf_mirror_source_drives_hfd_and_aria2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = ScriptTransferRunner(tmp_path / ".llm-platform" / "runs" / "task" / "attempt")
    captured: dict = {}

    async def fake_run(args: list[str], **kwargs: object) -> None:
        captured.update(args=args, **kwargs)

    monkeypatch.setattr(
        runner, "_resolve_hfd_tools", lambda: ("bash-test", "hfd.sh", "aria2c-test")
    )
    monkeypatch.setattr(runner, "_run_process", fake_run)
    result = await runner.download_huggingface(
        "org/corpus",
        {"hf_endpoint": "https://hf-mirror.com", "max_workers": 6},
        tmp_path / "dataset",
        lambda event_type, payload: collect_event([], event_type, payload),
        None,
    )

    args = captured["args"]
    assert args[0] == "bash-test"
    assert args[2:6] == ["org/corpus", "--dataset", "--tool", "aria2c"]
    assert args[args.index("-x") + 1] == "6"
    assert args[args.index("-j") + 1] == "6"
    assert args[args.index("--include") + 1 :] == DEFAULT_DATA_PATTERNS
    assert captured["env"]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert result["tool"] == "hfd + aria2c"


def test_hfd_progress_is_translated_to_platform_progress() -> None:
    progress = ScriptTransferRunner._progress_from_line(
        "[ 25%] 2/8 files | 1.50MB/6.00MB | 500.00KB/s | ETA 00:09",
        "hfd + aria2c",
    )

    assert progress is not None
    assert progress["downloaded_bytes"] == 1_500_000
    assert progress["total_bytes"] == 6_000_000
    assert progress["speed_bps"] == 500_000
    assert progress["file_count_completed"] == 2
    assert progress["file_count_total"] == 8


@pytest.mark.asyncio
async def test_external_process_streams_logs_and_honors_cancel(tmp_path: Path) -> None:
    runner = ScriptTransferRunner(tmp_path / "run")
    events: list[tuple[str, dict]] = []
    state = "running"

    async def cancel_later() -> None:
        nonlocal state
        await asyncio.sleep(0.7)
        state = "cancelled"

    cancel = asyncio.create_task(cancel_later())
    with pytest.raises(ProcessInterrupted) as error:
        await runner._run_process(
            [
                sys.executable,
                "-c",
                "import time; print('external tool started', flush=True); time.sleep(10)",
            ],
            tool="test-cli",
            emit=lambda event_type, payload: collect_event(events, event_type, payload),
            get_desired_state=lambda: state,
        )
    await cancel

    assert error.value.state == "cancelled"
    assert any(
        event_type == "log" and payload.get("message") == "external tool started"
        for event_type, payload in events
    )
