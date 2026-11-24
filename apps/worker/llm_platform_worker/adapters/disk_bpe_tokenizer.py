from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import shutil
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llm_platform_core.tokenizers import (
    validate_builtin_tokenizer_sample_tokens,
    validate_disk_bpe_config,
)

from .command import Emit, GetDesiredState
from .data_import import AnalysisInterrupted
from .hf_tokenizer import HfTokenizerAdapter, _raise_if_controlled, hf_tokenizer_available


class ShardedWordCounter:
    """Exact word-frequency aggregation with a bounded Python batch and SQLite shards."""

    def __init__(self, root: Path, *, shard_count: int = 64, batch_unique_limit: int = 100_000):
        self.root = root
        self.shard_count = shard_count
        self.batch_unique_limit = batch_unique_limit
        self.batch: Counter[str] = Counter()
        self.connections: list[sqlite3.Connection] = []
        self.total_occurrences = 0
        root.mkdir(parents=True, exist_ok=True)
        for index in range(shard_count):
            connection = sqlite3.connect(root / f"words-{index:03d}.sqlite3")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS words (word TEXT PRIMARY KEY, count INTEGER NOT NULL) "
                "WITHOUT ROWID"
            )
            self.connections.append(connection)

    def __enter__(self) -> ShardedWordCounter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _shard(self, word: str) -> int:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.shard_count

    def add(self, words: Iterator[str]) -> None:
        for word in words:
            if not word:
                continue
            self.batch[word] += 1
            self.total_occurrences += 1
            if len(self.batch) >= self.batch_unique_limit:
                self.flush()

    def flush(self) -> None:
        if not self.batch:
            return
        rows: list[list[tuple[str, int]]] = [[] for _ in range(self.shard_count)]
        for word, count in self.batch.items():
            rows[self._shard(word)].append((word, count))
        for connection, shard_rows in zip(self.connections, rows, strict=True):
            if not shard_rows:
                continue
            connection.executemany(
                "INSERT INTO words(word, count) VALUES (?, ?) "
                "ON CONFLICT(word) DO UPDATE SET count = count + excluded.count",
                shard_rows,
            )
            connection.commit()
        self.batch.clear()

    def select_candidates(
        self,
        *,
        min_frequency: int,
        max_unique_words: int,
        max_candidate_characters: int,
        max_word_length: int,
        get_desired_state: GetDesiredState | None = None,
    ) -> tuple[list[tuple[str, int]], dict[str, int]]:
        self.flush()
        total_unique = 0
        total_occurrences = 0
        streams = []
        for connection in self.connections:
            unique, occurrences = connection.execute(
                "SELECT count(*), coalesce(sum(count), 0) FROM words"
            ).fetchone()
            total_unique += int(unique)
            total_occurrences += int(occurrences)
            streams.append(
                connection.execute(
                    "SELECT word, count FROM words WHERE count >= ? AND length(word) <= ? "
                    "ORDER BY count DESC, word ASC",
                    (min_frequency, max_word_length),
                )
            )

        merged = heapq.merge(*streams, key=lambda row: (-int(row[1]), str(row[0])))
        selected: list[tuple[str, int]] = []
        candidate_characters = 0
        for position, row in enumerate(merged):
            if position % 4096 == 0:
                _raise_if_controlled(get_desired_state)
            word, count = str(row[0]), int(row[1])
            if len(selected) >= max_unique_words:
                break
            if candidate_characters + len(word) > max_candidate_characters:
                break
            selected.append((word, count))
            candidate_characters += len(word)
        return selected, {
            "total_unique_words": total_unique,
            "total_word_occurrences": total_occurrences,
            "candidate_unique_words": len(selected),
            "candidate_word_occurrences": sum(count for _, count in selected),
            "candidate_characters": candidate_characters,
            "pruned_unique_words": total_unique - len(selected),
        }

    def close(self) -> None:
        for connection in self.connections:
            connection.close()
        self.connections.clear()


class DiskBpeTokenizerAdapter(HfTokenizerAdapter):
    """Train BPE from exact disk-backed word counts and a bounded merge working set."""

    SHARD_COUNT = 64
    AGGREGATION_BATCH_UNIQUE = 100_000
    REPLAY_BATCH_SIZE = 2048

    @staticmethod
    def _pretokenized_words(tokenizer, text: str) -> Iterator[str]:
        normalized = tokenizer.normalizer.normalize_str(text) if tokenizer.normalizer else text
        if tokenizer.pre_tokenizer is None:
            yield normalized
            return
        for word, _offset in tokenizer.pre_tokenizer.pre_tokenize_str(normalized):
            if word:
                yield word

    @classmethod
    def _replay_counts(
        cls, candidates: list[tuple[str, int]], occurrence_budget: int
    ) -> tuple[list[tuple[str, int]], int]:
        if not candidates:
            return [], 0
        budget = max(occurrence_budget, len(candidates))
        remaining = budget - len(candidates)
        excess_total = sum(max(count - 1, 0) for _, count in candidates)
        replay: list[tuple[str, int]] = []
        used = 0
        for word, count in candidates:
            additional = (
                (remaining * max(count - 1, 0)) // excess_total if excess_total and remaining else 0
            )
            repetitions = 1 + additional
            replay.append((word, repetitions))
            used += repetitions
        return replay, used

    @classmethod
    def _replay_batches(
        cls,
        replay: list[tuple[str, int]],
        get_desired_state: GetDesiredState | None,
    ) -> Iterator[list[str]]:
        batch: list[str] = []
        emitted = 0
        for word, repetitions in replay:
            for _ in range(repetitions):
                if emitted % 4096 == 0:
                    _raise_if_controlled(get_desired_state)
                batch.append(word)
                emitted += 1
                if len(batch) >= cls.REPLAY_BATCH_SIZE:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def _aggregate_and_train(
        self,
        *,
        corpus: Iterator[str],
        template_tokenizer,
        trainer,
        work_dir: Path,
        bounds: dict[str, int],
        report_progress,
        get_desired_state: GetDesiredState | None,
    ) -> tuple[Any, dict[str, int]]:
        from tokenizers import Tokenizer, models

        frequency_dir = work_dir / "word-frequency"
        last_report = 0.0
        with ShardedWordCounter(
            frequency_dir,
            shard_count=self.SHARD_COUNT,
            batch_unique_limit=self.AGGREGATION_BATCH_UNIQUE,
        ) as counter:
            for document_index, text in enumerate(corpus, start=1):
                _raise_if_controlled(get_desired_state)
                counter.add(self._pretokenized_words(template_tokenizer, text))
                now = time.monotonic()
                if now - last_report >= self.PROGRESS_INTERVAL_SECONDS:
                    report_progress(
                        {
                            "phase": "external_frequency_counting",
                            "counted_documents": document_index,
                            "counted_word_occurrences": counter.total_occurrences,
                        }
                    )
                    last_report = now
            _raise_if_controlled(get_desired_state)
            report_progress(
                {
                    "phase": "selecting_bounded_candidates",
                    "counted_word_occurrences": counter.total_occurrences,
                }
            )
            candidates, aggregation = counter.select_candidates(
                min_frequency=bounds["min_word_frequency"],
                max_unique_words=bounds["max_unique_words"],
                max_candidate_characters=bounds["max_candidate_characters"],
                max_word_length=bounds["max_word_length"],
                get_desired_state=get_desired_state,
            )
            if not candidates:
                raise ValueError("Disk BPE candidate selection produced no words")
            aggregation["frequency_store_bytes"] = sum(
                path.stat().st_size for path in frequency_dir.glob("*.sqlite3")
            )

            replay, replay_occurrences = self._replay_counts(
                candidates, bounds["merge_occurrence_budget"]
            )
            aggregation["merge_replay_occurrences"] = replay_occurrences
            report_progress({"phase": "bounded_bpe_merge", **aggregation})
            _raise_if_controlled(get_desired_state)

            training_tokenizer = Tokenizer(
                models.BPE(
                    unk_token=template_tokenizer.model.unk_token,
                    byte_fallback=bool(template_tokenizer.model.byte_fallback),
                )
            )
            training_tokenizer.train_from_iterator(
                self._replay_batches(replay, get_desired_state), trainer=trainer
            )
            training_tokenizer.normalizer = template_tokenizer.normalizer
            training_tokenizer.pre_tokenizer = template_tokenizer.pre_tokenizer
            training_tokenizer.decoder = template_tokenizer.decoder
            return training_tokenizer, aggregation

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        if not hf_tokenizer_available():
            raise RuntimeError("Hugging Face tokenizers is not installed on this Worker")
        payload = task.get("payload") or {}
        config = payload.get("config") or {}
        if str(payload.get("algorithm") or "BPE").lower() != "bpe":
            raise ValueError("Disk-backed tokenizer supports BPE only")
        sample_tokens = validate_builtin_tokenizer_sample_tokens(config.get("sample_tokens"))
        bounds = validate_disk_bpe_config(config)
        data_input = payload.get("data_input") or {}
        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        loop = asyncio.get_running_loop()
        last_sampled_tokens = 0
        max_task_progress = 0.0

        def report_progress(progress: dict[str, Any]) -> None:
            nonlocal last_sampled_tokens, max_task_progress
            phase = str(progress.get("phase") or "")
            if "sampled_tokens" in progress:
                last_sampled_tokens = max(
                    last_sampled_tokens, int(progress.get("sampled_tokens") or 0)
                )
            sampled_tokens = last_sampled_tokens
            target_tokens = max(int(progress.get("target_tokens") or sample_tokens), 1)
            if phase in {"streaming_training", "external_frequency_counting"}:
                task_progress = 0.1 + 0.6 * min(sampled_tokens / target_tokens, 1.0)
                phase = "external_frequency_counting"
            elif phase == "selecting_bounded_candidates":
                task_progress = 0.74
            elif phase == "bounded_bpe_merge":
                task_progress = 0.8
            else:
                task_progress = 0.08
            max_task_progress = max(max_task_progress, task_progress)
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **progress,
                        "phase": phase,
                        "sampled_tokens": sampled_tokens,
                        "progress": max_task_progress,
                    },
                ),
                loop,
            )
            future.result()

        await emit("progress", {"progress": 0.03, "phase": "preparing_corpus"})
        try:
            corpus, evaluation_corpus, corpus_stats = await asyncio.to_thread(
                self._corpus,
                data_input,
                sample_tokens,
                run_dir / "corpus-index",
                report_progress,
                get_desired_state,
            )
            template, trainer, special_tokens = self._build_tokenizer(
                "BPE", config, int(payload.get("vocab_size") or 32_000)
            )
            tokenizer, aggregation = await asyncio.to_thread(
                self._aggregate_and_train,
                corpus=corpus,
                template_tokenizer=template,
                trainer=trainer,
                work_dir=run_dir,
                bounds=bounds,
                report_progress=report_progress,
                get_desired_state=get_desired_state,
            )
        except AnalysisInterrupted as exc:
            await emit("progress", {"progress": 0.1, "phase": exc.state})
            return {"controlled_state": exc.state}
        finally:
            shutil.rmtree(run_dir / "word-frequency", ignore_errors=True)

        if corpus_stats["documents"] <= 0:
            raise ValueError("Tokenizer input selected no readable documents")
        await emit(
            "progress",
            {
                "progress": 0.92,
                "phase": "finalizing",
                "sampled_documents": corpus_stats["documents"],
                "sampled_tokens": corpus_stats["token_estimate"],
                **aggregation,
            },
        )
        metrics = self._metrics(tokenizer, evaluation_corpus, special_tokens)
        output_dir = run_dir / "tokenizer"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_dir / "tokenizer.json"))
        metadata = {
            "framework": "disk-backed-huggingface-bpe",
            "algorithm": "BPE",
            "vocab_size_requested": int(payload.get("vocab_size") or 32_000),
            "vocab_size_actual": tokenizer.get_vocab_size(),
            "normalization": config.get("normalization", "NFKC"),
            "byte_fallback": bool(config.get("byte_fallback", False)),
            "special_tokens": special_tokens,
            "sampled_documents": corpus_stats["documents"],
            "sample_token_estimate": corpus_stats["token_estimate"],
            "aggregation": {**bounds, **aggregation},
            "frequency_compression_ratio": round(
                aggregation["candidate_word_occurrences"]
                / max(aggregation["merge_replay_occurrences"], 1),
                6,
            ),
            "metrics": metrics,
        }
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "unk_token": special_tokens[0],
                    **({"bos_token": special_tokens[1]} if len(special_tokens) >= 2 else {}),
                    **(
                        {"eos_token": special_tokens[2], "pad_token": special_tokens[2]}
                        if len(special_tokens) >= 3
                        else {}
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        await emit("progress", {"progress": 1.0, "phase": "completed", "metrics": metrics})
        return {
            "framework": "disk-backed-huggingface-bpe",
            "metrics": metrics,
            "aggregation": aggregation,
            "outputs": [
                {
                    "ref": "tokenizer",
                    "path": str(output_dir),
                    "kind": "tokenizer",
                    "media_type": "application/gzip",
                }
            ],
        }
