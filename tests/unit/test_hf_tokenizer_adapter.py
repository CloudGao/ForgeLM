from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import llm_platform_worker.adapters.hf_tokenizer as hf_tokenizer_module
import pytest
from llm_platform_worker.adapters.data_import import AnalysisInterrupted, DataImportAdapter
from llm_platform_worker.adapters.disk_bpe_tokenizer import (
    DiskBpeTokenizerAdapter,
    ShardedWordCounter,
)
from llm_platform_worker.adapters.hf_tokenizer import HfTokenizerAdapter


def test_parquet_selection_stops_without_reading_remaining_manifest(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.table({"text": ["第一条正文"] * 4096}), path)
    real_file = pq.ParquetFile
    reads = []
    closed = []

    class BoundedFile:
        def __init__(self, source):
            self.reader = real_file(source)
            self.metadata = self.reader.metadata
            self.num_row_groups = self.reader.num_row_groups

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.reader.close()
            closed.append(True)

        def iter_batches(self, **kwargs):
            for batch in self.reader.iter_batches(**kwargs):
                reads.append(batch.num_rows)
                yield batch

    monkeypatch.setattr(pq, "ParquetFile", BoundedFile)

    def manifest():
        yield {"logical_path": path.name, "record_index": 0}
        raise AssertionError("unused manifest tail must not be consumed")

    stream = HfTokenizerAdapter._iter_selected_texts(path, manifest())
    assert next(stream)[0] == "第一条正文"
    stream.close()
    assert reads == [512]
    assert closed == [True]


def test_parquet_selection_preserves_cross_group_and_duplicate_order(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "dataset.parquet"
    texts = [f"第 {index} 条正文" for index in range(2200)]
    pq.write_table(pa.table({"text": texts}), path, row_group_size=1000)
    positions = [0, 511, 512, 999, 1000, 1000, 2199]
    records = [
        {"logical_path": path.name, "record_index": index,
         "content_hash": hashlib.sha256(texts[index].encode()).hexdigest()}
        for index in positions
    ]
    result = list(HfTokenizerAdapter._iter_selected_texts(path, iter(records)))
    assert [text for text, _ in result] == [texts[index] for index in positions]
    assert [record for _, record in result] == records


def test_parquet_selection_falls_back_after_verified_prefix(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "dataset.parquet"
    texts = ["第一条", "第二条", "第三条"]
    pq.write_table(pa.table({"text": [texts[0], "", texts[1], texts[2]]}), path)
    records = [
        {"logical_path": path.name, "record_index": index,
         "content_hash": hashlib.sha256(text.encode()).hexdigest()}
        for index, text in enumerate(texts)
    ]
    assert [text for text, _ in HfTokenizerAdapter._iter_selected_texts(path, records)] == texts


def test_parquet_selection_checks_cancellation_and_order(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.table({"text": ["第一条", "第二条"]}), path)
    records = [{"logical_path": path.name, "record_index": index} for index in [1, 0]]
    with pytest.raises(ValueError, match="not ordered"):
        list(HfTokenizerAdapter._iter_selected_texts(path, records))
    state = "running"
    stream = HfTokenizerAdapter._iter_selected_texts(path, records, lambda: state)
    assert next(stream)[0] == "第二条"
    state = "cancelled"
    with pytest.raises(AnalysisInterrupted):
        next(stream)


def _dataset_input(tmp_path: Path) -> dict:
    source = tmp_path / "source"
    source.mkdir()
    documents = [
        "hello world and tokenizer training",
        "你好，世界。中文分词测试。",
        "def answer(value):\n    return value + 42",
    ]
    (source / "train.jsonl").write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in documents),
        encoding="utf-8",
    )
    manifest = tmp_path / "records.jsonl"
    lines = [
        {
            "type": "header",
            "format": "llm-dataset-records-v1",
            "dataset_version_id": "dataset-version-1",
        }
    ]
    lines.extend(
        {
            "type": "record",
            "record_id": f"record-{index}",
            "logical_path": "train.jsonl",
            "record_index": index,
            "token_estimate": 20,
        }
        for index in range(len(documents))
    )
    manifest.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    return {
        "format": "llm-dataset-input-v1",
        "runtime_path": str(manifest),
        "runtime_sources": [{"source_index": 0, "runtime_path": str(source)}],
    }


@pytest.mark.asyncio
async def test_hf_tokenizer_trains_real_bpe_artifact(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await HfTokenizerAdapter().run(
        {
            "runtime_context": {"run_dir": str(tmp_path / "run")},
            "payload": {
                "algorithm": "BPE",
                "vocab_size": 300,
                "config": {
                    "sample_tokens": 60,
                    "normalization": "NFKC",
                    "byte_fallback": True,
                    "special_tokens": ["<unk>", "<s>", "</s>"],
                },
                "data_input": _dataset_input(tmp_path),
            },
        },
        emit,
    )

    output_dir = Path(result["outputs"][0]["path"])
    assert result["framework"] == "huggingface-tokenizers"
    assert "gate_passed" not in result
    assert result["metrics"]["round_trip"] == 1
    assert result["metrics"]["newline_round_trip"] == 1
    assert result["metrics"]["indentation_round_trip"] == 1
    assert result["metrics"]["special_token_integrity"] == 1
    assert result["metrics"]["math_chars_per_token"] is not None
    assert result["metrics"]["byte_fallback_rate"] >= 0
    assert result["metrics"]["vocab_garbage_rate"] >= 0
    assert (output_dir / "tokenizer.json").is_file()
    metadata = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_map = json.loads(
        (output_dir / "special_tokens_map.json").read_text(encoding="utf-8")
    )
    assert metadata["algorithm"] == "BPE"
    assert metadata["sampled_documents"] == 3
    assert special_map == {
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
    }
    assert any(
        payload.get("phase") == "streaming_training" and payload.get("sampled_tokens")
        for _event_type, payload in events
    )
    assert events[-1][1]["phase"] == "completed"


@pytest.mark.asyncio
async def test_hf_tokenizer_rejects_wordpiece_byte_fallback(tmp_path: Path) -> None:
    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    with pytest.raises(ValueError, match="WordPiece does not support byte fallback"):
        await HfTokenizerAdapter().run(
            {
                "runtime_context": {"run_dir": str(tmp_path / "run")},
                "payload": {
                    "algorithm": "WordPiece",
                    "vocab_size": 300,
                    "config": {"sample_tokens": 20, "byte_fallback": True},
                    "data_input": _dataset_input(tmp_path),
                },
            },
            emit,
        )


@pytest.mark.asyncio
async def test_hf_tokenizer_loading_does_not_block_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desired_state = "running"
    loading_started = asyncio.Event()

    def blocking_corpus(
        _cls,
        _data_input: dict,
        _sample_tokens: int,
        _work_dir: Path,
        _report_progress=None,
        get_desired_state=None,
    ):
        loading_started_loop.call_soon_threadsafe(loading_started.set)
        while True:
            state = get_desired_state() if get_desired_state else "running"
            if state in {"paused", "cancelled"}:
                raise AnalysisInterrupted(state)
            time.sleep(0.005)

    loading_started_loop = asyncio.get_running_loop()
    monkeypatch.setattr(HfTokenizerAdapter, "_corpus", classmethod(blocking_corpus))

    async def emit(_event_type: str, _payload: dict) -> str:
        return desired_state

    run = asyncio.create_task(
        HfTokenizerAdapter().run(
            {
                "runtime_context": {"run_dir": str(tmp_path / "run")},
                "payload": {
                    "algorithm": "BPE",
                    "vocab_size": 300,
                    "config": {"sample_tokens": 20},
                    "data_input": _dataset_input(tmp_path),
                },
            },
            emit,
            lambda: desired_state,
        )
    )
    await asyncio.wait_for(loading_started.wait(), timeout=1)
    desired_state = "cancelled"

    result = await asyncio.wait_for(run, timeout=1)
    assert result == {"controlled_state": "cancelled"}


def test_dataset_corpus_is_lazy_and_stops_at_sample_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_input = _dataset_input(tmp_path)
    read_documents = 0

    def tracked_documents(*_args, **_kwargs):
        nonlocal read_documents
        for index in range(1000):
            read_documents += 1
            yield f"document {index}"

    manifest = Path(data_input["runtime_path"])
    lines = [
        {
            "type": "header",
            "format": "llm-dataset-records-v1",
            "dataset_version_id": "dataset-version-1",
        }
    ]
    lines.extend(
        {
            "type": "record",
            "record_id": f"record-{index}",
            "logical_path": "train.jsonl",
            "record_index": index,
            "token_estimate": 20,
        }
        for index in range(1000)
    )
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    monkeypatch.setattr(DataImportAdapter, "_iter_documents", tracked_documents)

    corpus, evaluation, stats = HfTokenizerAdapter._corpus(data_input, 60, tmp_path / "index")

    assert read_documents == 0
    assert list(corpus) == ["document 0", "document 1", "document 2"]
    assert read_documents == 3
    assert stats == {"documents": 3, "token_estimate": 60}
    assert evaluation == ["document 0", "document 1", "document 2"]


def test_training_corpus_crosses_runtime_boundary_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HfTokenizerAdapter, "TRAINING_BATCH_DOCUMENTS", 2)
    monkeypatch.setattr(HfTokenizerAdapter, "TRAINING_BATCH_CHARACTERS", 10)

    batches = list(HfTokenizerAdapter._batched_corpus(iter(["aaaa", "bbbb", "cccc", "d"])))

    assert batches == [["aaaa", "bbbb"], ["cccc", "d"]]


def test_manifest_iteration_periodically_releases_file_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "records.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"type": "header", "format": "llm-dataset-records-v1"}),
                json.dumps({"type": "record", "record_id": "one"}),
                json.dumps({"type": "record", "record_id": "two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    released: list[Path] = []
    monkeypatch.setattr(HfTokenizerAdapter, "MANIFEST_CACHE_RELEASE_BYTES", 1)
    monkeypatch.setattr(
        hf_tokenizer_module,
        "release_file_cache",
        lambda path: released.append(path) or True,
    )

    records = list(
        HfTokenizerAdapter._iter_manifest_records(
            manifest, "llm-dataset-records-v1"
        )
    )

    assert [record["record_id"] for record in records] == ["one", "two"]
    assert released == [manifest, manifest, manifest]


def test_mixture_corpus_uses_reference_only_index_and_preserves_sampling(
    tmp_path: Path,
) -> None:
    sources = []
    manifest_sources = []
    records = []
    for source_index, weight in enumerate((700_000, 300_000)):
        source = tmp_path / f"source-{source_index}"
        source.mkdir()
        documents = [f"source {source_index} document {index}" for index in range(10)]
        (source / "train.jsonl").write_text(
            "".join(json.dumps({"text": text}) + "\n" for text in documents),
            encoding="utf-8",
        )
        source_records = [
            {
                "type": "record",
                "source_index": source_index,
                "record_id": f"{source_index}-{index}",
                "logical_path": "train.jsonl",
                "record_index": index,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_estimate": 10,
                "byte_length": len(text.encode()),
            }
            for index, text in enumerate(documents)
        ]
        record_manifest = tmp_path / f"source-{source_index}-records.jsonl"
        record_manifest.write_text(
            "".join(
                json.dumps(item) + "\n"
                for item in [
                    {"type": "header", "format": "llm-dataset-records-v1"},
                    *source_records,
                ]
            ),
            encoding="utf-8",
        )
        sources.append(
            {
                "source_index": source_index,
                "runtime_path": str(source),
                "runtime_record_manifest_path": str(record_manifest),
            }
        )
        manifest_sources.append(
            {
                "source_index": source_index,
                "weight_units": weight,
                "selected_document_count": len(source_records),
                "selected_token_estimate": 10 * len(source_records),
            }
        )
        records.extend(source_records)
    manifest = tmp_path / "mixture.jsonl"
    header = {
        "type": "header",
        "format": "llm-mixture-manifest-v1",
        "seed": 42,
        "stream_key": "stable-test-stream",
        "sources": manifest_sources,
    }
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in [header, *records]),
        encoding="utf-8",
    )
    data_input = {
        "format": "llm-mixture-manifest-v1",
        "runtime_path": str(manifest),
        "runtime_sources": sources,
    }

    corpus, _evaluation, stats = HfTokenizerAdapter._corpus(data_input, 40, tmp_path / "index")
    first = list(corpus)
    corpus_again, _evaluation_again, stats_again = HfTokenizerAdapter._corpus(
        data_input, 40, tmp_path / "index-again"
    )

    assert first == list(corpus_again)
    assert len(first) == 4
    assert stats == stats_again == {"documents": 4, "token_estimate": 40}
    index_path = tmp_path / "index" / "mixture-corpus.sqlite3"
    assert index_path.is_file()
    with sqlite3.connect(index_path) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(samples)")
        }
        assert "text" not in columns
        assert columns == {
            "sample_index",
            "source_index",
            "ordinal",
            "logical_path",
            "record_index",
            "content_hash",
            "token_estimate",
            "byte_length",
        }
        assert connection.execute("SELECT count(*) FROM samples").fetchone()[0] == 4
    assert not (tmp_path / "index" / "mixture-corpus.requests.sqlite3").exists()

    reference_manifest = tmp_path / "mixture-v2.jsonl"
    reference_header = {
        **header,
        "format": "llm-mixture-index-v2",
        "record_order": "record_manifest",
    }
    reference_manifest.write_text(json.dumps(reference_header) + "\n", encoding="utf-8")
    reference_input = {
        "format": "llm-mixture-index-v2",
        "runtime_path": str(reference_manifest),
        "runtime_sources": sources,
    }
    reference_corpus, _evaluation, reference_stats = HfTokenizerAdapter._corpus(
        reference_input, 40, tmp_path / "reference-index"
    )
    assert list(reference_corpus) == first
    assert reference_stats == stats

    validation_prefix = list(
        HfTokenizerAdapter.validation_corpus_prefix(reference_input, 20)
    )
    assert validation_prefix == [
        "source 0 document 0",
        "source 0 document 1",
        "source 1 document 0",
    ]
    random_input = {
        **reference_input,
        "manifest_hash": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "seed": 42,
        "validation_sampling": {
            "strategy": "weighted_random_materialized_v1",
            "seed": 42,
        },
    }
    random_validation = list(
        HfTokenizerAdapter.validation_corpus_random(random_input, 20)
    )
    assert random_validation == list(
        HfTokenizerAdapter.validation_corpus_random(random_input, 20)
    )
    assert len(random_validation) == 3
    assert len(set(random_validation)) == 3
    assert set(random_validation) != set(validation_prefix)
    assert sum(text.startswith("source 0") for text in random_validation) == 2
    assert sum(text.startswith("source 1") for text in random_validation) == 1
    changed_seed = {
        **random_input,
        "validation_sampling": {
            "strategy": "weighted_random_materialized_v1",
            "seed": 43,
        },
    }
    assert random_validation != list(
        HfTokenizerAdapter.validation_corpus_random(changed_seed, 20)
    )
    assert not (tmp_path / "validation-mixture-corpus.sqlite3").exists()

    spool_path = tmp_path / "spooled-index" / "ordered-corpus.sqlite3"
    spooled_corpus, _spooled_evaluation, spooled_stats = HfTokenizerAdapter._corpus(
        reference_input,
        40,
        tmp_path / "spooled-index",
        spool_path=spool_path,
    )
    assert spool_path.is_file()
    assert list(spooled_corpus) == first
    assert spooled_stats == stats
    assert not spool_path.exists()


def test_sharded_word_counter_aggregates_exact_counts_and_prunes_deterministically(
    tmp_path: Path,
) -> None:
    with ShardedWordCounter(tmp_path / "counts", shard_count=4, batch_unique_limit=2) as counter:
        counter.add(iter(["beta", "alpha", "beta", "gamma", "alpha", "beta", "delta"]))
        candidates, stats = counter.select_candidates(
            min_frequency=2,
            max_unique_words=2,
            max_candidate_characters=100,
            max_word_length=20,
        )

    assert candidates == [("beta", 3), ("alpha", 2)]
    assert stats == {
        "total_unique_words": 4,
        "total_word_occurrences": 7,
        "candidate_unique_words": 2,
        "candidate_word_occurrences": 5,
        "candidate_characters": 9,
        "pruned_unique_words": 2,
    }


def test_disk_bpe_replay_is_deterministic_and_bounded() -> None:
    candidates = [("common", 100), ("middle", 10), ("rare", 2)]

    first, used = DiskBpeTokenizerAdapter._replay_counts(candidates, 20)
    second, used_again = DiskBpeTokenizerAdapter._replay_counts(candidates, 20)

    assert first == second
    assert used == used_again <= 20
    assert all(repetitions >= 1 for _, repetitions in first)
    assert dict(first)["common"] > dict(first)["middle"] >= dict(first)["rare"]


@pytest.mark.asyncio
async def test_disk_bpe_tokenizer_trains_from_external_counts(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await DiskBpeTokenizerAdapter().run(
        {
            "runtime_context": {"run_dir": str(tmp_path / "run")},
            "payload": {
                "algorithm": "BPE",
                "vocab_size": 300,
                "config": {
                    "sample_tokens": 60,
                    "normalization": "NFKC",
                    "byte_fallback": True,
                    "special_tokens": ["<unk>", "<s>", "</s>"],
                    "min_word_frequency": 1,
                    "max_unique_words": 1_000,
                    "max_candidate_characters": 100_000,
                    "max_word_length": 128,
                    "merge_occurrence_budget": 2_000,
                },
                "data_input": _dataset_input(tmp_path),
            },
        },
        emit,
    )

    output_dir = Path(result["outputs"][0]["path"])
    metadata = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_map = json.loads(
        (output_dir / "special_tokens_map.json").read_text(encoding="utf-8")
    )
    phases = [payload.get("phase") for _event_type, payload in events]
    progress_values = [
        payload["progress"] for _event_type, payload in events if "progress" in payload
    ]
    assert result["framework"] == "disk-backed-huggingface-bpe"
    assert "gate_passed" not in result
    assert metadata["aggregation"]["total_unique_words"] > 0
    assert metadata["aggregation"]["candidate_unique_words"] <= 1_000
    assert metadata["aggregation"]["candidate_characters"] <= 100_000
    assert metadata["aggregation"]["merge_replay_occurrences"] <= 2_000
    assert special_map["eos_token"] == special_map["pad_token"] == "</s>"
    assert "external_frequency_counting" in phases
    assert "selecting_bounded_candidates" in phases
    assert "bounded_bpe_merge" in phases
    assert progress_values == sorted(progress_values)
    assert not (tmp_path / "run" / "word-frequency").exists()


@pytest.mark.asyncio
async def test_disk_bpe_external_counting_can_be_cancelled_and_cleans_shards(
    tmp_path: Path,
) -> None:
    desired_state = "running"

    async def emit(_event_type: str, payload: dict) -> str:
        nonlocal desired_state
        if payload.get("phase") == "external_frequency_counting":
            desired_state = "cancelled"
        return desired_state

    result = await DiskBpeTokenizerAdapter().run(
        {
            "runtime_context": {"run_dir": str(tmp_path / "run")},
            "payload": {
                "algorithm": "BPE",
                "vocab_size": 300,
                "config": {
                    "sample_tokens": 60,
                    "min_word_frequency": 1,
                    "max_unique_words": 1_000,
                    "max_candidate_characters": 100_000,
                    "merge_occurrence_budget": 2_000,
                },
                "data_input": _dataset_input(tmp_path),
            },
        },
        emit,
        lambda: desired_state,
    )

    assert result == {"controlled_state": "cancelled"}
    assert not (tmp_path / "run" / "word-frequency").exists()
