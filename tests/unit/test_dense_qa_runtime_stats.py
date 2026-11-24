from llm_platform_core.dense_qa import corpus_stats
from llm_platform_worker.dense_qa_runtime import _CorpusStatsAccumulator


def _row(node_id, angle, difficulty, answer_length, source_count=1, *, annotate=False):
    return {
        "primary_node_id": node_id,
        "angle": angle,
        "difficulty": difficulty,
        "answer_length": answer_length,
        "source_nodes": [{"id": str(index)} for index in range(source_count)],
        **({"review_policy": "annotate_only"} if annotate else {}),
    }


def test_incremental_corpus_stats_match_full_scan():
    first = [
        _row("n1", "identity", "L1", "short", annotate=True),
        _row("n1", "mechanism", "L2", "medium", annotate=True),
    ]
    second = [
        _row("n2", "synthesis", "L3", "long", source_count=2, annotate=True),
    ]

    stats = _CorpusStatsAccumulator(first)
    stats.add(second)

    assert stats.snapshot() == corpus_stats(first + second)


def test_incremental_corpus_stats_handle_empty_input():
    assert _CorpusStatsAccumulator().snapshot() == corpus_stats([])
