"""Engine-independent acceptance and selection for measured training configurations."""

import math

PERFORMANCE_PROTOCOL = "forgelm.training-performance.v1"
SEARCH_PROTOCOL = "forgelm.training-performance-search.v2"


def automatic_performance_candidates(baseline, effective_tokens, sequence_length):
    """Enumerate legal batch geometries; the engine prunes measured memory failures."""
    rows = [dict(baseline)]
    sizes = [size for size in range(1, 65)
             if effective_tokens % (sequence_length * size) == 0]
    # Grow batches first, then reduce recomputation at the largest viable batch.
    larger = [size for size in sizes if size > baseline["micro_batch_size"]]
    for size in larger:
        rows.append({**baseline, "micro_batch_size": size})
    for enabled, interval in ((True, 1), (True, 2), (True, 4), (False, 1)):
        for size in [baseline["micro_batch_size"], *larger]:
            value = {"micro_batch_size": size, "activation_checkpointing": enabled,
                     "checkpoint_every_n_layers": interval}
            if value not in rows:
                rows.append(value)
    for size in reversed([s for s in sizes if s < baseline["micro_batch_size"]]):
        rows.append({"micro_batch_size": size, "activation_checkpointing": True, "checkpoint_every_n_layers": 1})
    return rows


def projected_candidate_memory(candidate, rows):
    """Conservative reservation extrapolation, including observed external usage.

    Two measured batch sizes establish batch growth. Two recomputation policies
    establish retained-activation growth; scale that increment with batch size.
    This prevents Windows shared-memory spill before an allocator raises OOM.
    """
    usable = [r for r in rows if r.get("peak_reserved_mb") and r.get("non_allocator_memory_mb") is not None]
    if not usable:
        return None
    def retained(c):
        return 1 - 1 / c["checkpoint_every_n_layers"] if c["activation_checkpointing"] else 1.0
    size, keep = candidate["micro_batch_size"], retained(candidate)
    same = sorted([r for r in usable if retained(r["configuration"]) == keep],
                  key=lambda r: r["configuration"]["micro_batch_size"])
    estimates = []
    if len(same) >= 2:
        a, b = same[-2:]
        x, y = a["configuration"]["micro_batch_size"], b["configuration"]["micro_batch_size"]
        if y > x and size > y:
            estimates.append(b["peak_reserved_mb"] + max(0, b["peak_reserved_mb"] - a["peak_reserved_mb"]) * (size-y)/(y-x))
    for base in usable:
        c = base["configuration"]
        if c["micro_batch_size"] != size or retained(c) >= keep:
            continue
        for a in usable:
            ca = a["configuration"]
            if retained(ca) != retained(c):
                continue
            for b in usable:
                cb = b["configuration"]
                if ca["micro_batch_size"] == cb["micro_batch_size"] and retained(ca) < retained(cb) <= keep:
                    increment = max(0, b["peak_reserved_mb"] - a["peak_reserved_mb"])
                    estimates.append(base["peak_reserved_mb"] + increment * size / ca["micro_batch_size"]
                                     * (keep-retained(ca))/(retained(cb)-retained(ca)))
    return max(estimates) + max(r["non_allocator_memory_mb"] for r in usable) if estimates else None


def performance_memory_headroom(total_mb, requested=None):
    return float(requested) if requested is not None else min(2048.0, max(1024.0, total_mb * 0.05))


def memory_dominated_candidate(candidate, rows):
    """Only memory failures prune. Numerical rejection must not stop batch exploration."""
    def interval(config):
        return config["checkpoint_every_n_layers"] if config["activation_checkpointing"] else float("inf")
    return any(
        row.get("status") == "oom" or row.get("memory_passed") is False
        for row in rows
        if candidate["micro_batch_size"] >= row["configuration"]["micro_batch_size"]
        and interval(candidate) >= interval(row["configuration"])
    )


def projected_memory_dominated_candidate(candidate, skipped):
    # A projected failure is an envelope across both batch and retained
    # activations. Do not reconsider a larger configuration merely because its
    # exact batch/interval pair has never been measured.
    bounds = [{"configuration": row["configuration"], "memory_passed": False}
              for row in skipped
              if row.get("reason") in {"projected_memory_limit", "projected_memory_envelope"}]
    return memory_dominated_candidate(candidate, bounds)


def performance_candidate_passes(row, memory_limit_mb):
    fields = ("peak_reserved_mb", "tokens_per_second", "gradient_relative_l2",
              "gradient_cosine", "loss_absolute_difference")
    if any(type(row.get(key)) not in {int, float} or not math.isfinite(row[key]) for key in fields):
        return False
    gradient_limit, cosine_limit = 0.01, 0.9999
    policy = row.get("numerical_validation")
    if policy in {"adam-update-and-loss-window-v1", "bf16-heldout-stability-v2", "sft-preserved-loss-groups-adam-v1"}:
        update_fields = ("update_relative_l2", "update_cosine", "max_loss_absolute_difference")
        if any(type(row.get(key)) not in {int, float} or not math.isfinite(row[key]) for key in update_fields):
            return False
        update_limit, update_cosine = (0.03, 0.999) if policy == "bf16-heldout-stability-v2" else (0.01, 0.9999)
        if not (0 <= row["update_relative_l2"] <= update_limit and row["update_cosine"] >= update_cosine
                and 0 <= row["max_loss_absolute_difference"] <= 0.001):
            return False
        gradient_limit, cosine_limit = 0.05, 0.99875
        if policy == "sft-preserved-loss-groups-adam-v1":
            # Lookahead uses separate frozen training packs, not a quality/heldout set.
            for key in ("verification_loss", "verification_loss_absolute_difference"):
                if type(row.get(key)) not in {int, float} or not math.isfinite(row[key]):
                    return False
            if not (row["verification_loss"] >= 0 and 0 <= row["verification_loss_absolute_difference"] <= 0.001
                    and type(row.get("verification_examples")) is int and row["verification_examples"] >= 32):
                return False
        if policy == "bf16-heldout-stability-v2":
            for key in ("validation_loss", "validation_loss_absolute_difference"):
                if type(row.get(key)) not in {int, float} or not math.isfinite(row[key]):
                    return False
            if not (row["validation_loss"] >= 0 and 0 <= row["validation_loss_absolute_difference"] <= 0.001
                    and type(row.get("validation_examples")) is int and row["validation_examples"] >= 32):
                return False
            gradient_limit, cosine_limit = 0.10, 0.995
    return (0 < row["peak_reserved_mb"] <= memory_limit_mb and row["tokens_per_second"] > 0
            and 0 <= row["gradient_relative_l2"] <= gradient_limit and row["gradient_cosine"] >= cosine_limit
            and 0 <= row["loss_absolute_difference"] <= 0.001)


def select_performance_candidate(rows):
    passing = [row for row in rows if row.get("status") == "passed"]
    if not rows or rows[0].get("status") not in {"passed", "reference_only"}:
        raise ValueError("Baseline performance probe must pass before selecting a candidate")
    if not passing:
        raise ValueError("No performance candidate satisfies memory and numerical requirements")
    best = max(passing, key=lambda row: row["tokens_per_second"])
    # Short measurements cannot reliably distinguish small differences.
    if rows[0]["status"] == "passed" and best["tokens_per_second"] < rows[0]["tokens_per_second"] * 1.03:
        best = rows[0]
    return dict(best["configuration"])
