from __future__ import annotations

from typing import Any

SEQUENCE_ORDER_STRATEGIES: dict[str, dict[str, Any]] = {
    "sequential_v1": {
        "id": "sequential_v1",
        "label": "全局顺序读取",
        "storage_mode": "indexed",
        "shuffle_granularity": "none",
        "requires_block_size": False,
        "materializes_sequences": False,
    },
    "block_shuffle_v1": {
        "id": "block_shuffle_v1",
        "label": "块间随机、块内顺序",
        "storage_mode": "indexed",
        "shuffle_granularity": "sequence_block",
        "requires_block_size": True,
        "materializes_sequences": False,
    },
    "materialized_global_shuffle_v1": {
        "id": "materialized_global_shuffle_v1",
        "label": "物化全局随机文件",
        "storage_mode": "materialized",
        "shuffle_granularity": "sequence",
        "requires_block_size": False,
        "materializes_sequences": True,
    },
}


def sequence_order_strategy(strategy_id: str) -> dict[str, Any]:
    try:
        return SEQUENCE_ORDER_STRATEGIES[strategy_id]
    except KeyError as exc:
        raise ValueError(f"unsupported sequence order policy: {strategy_id}") from exc


def materialized_sequence_size_bytes(
    *, sequence_count: int, sequence_length: int, token_bytes: int
) -> int:
    if sequence_count < 0 or sequence_length <= 0 or token_bytes not in {2, 4}:
        raise ValueError("invalid materialized sequence dimensions")
    boundary_bytes = (sequence_length + 7) // 8
    return sequence_count * (sequence_length * token_bytes + boundary_bytes + 4)
