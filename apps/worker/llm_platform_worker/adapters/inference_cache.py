"""Release idle inference objects before the serial Worker changes GPU engines."""

import gc
import time


def release_expired(owner, idle_seconds: float = 600, *, now: float | None = None) -> bool:
    """Active generation is never interrupted; completed requests reset idle time."""
    last_used = getattr(owner, "_last_used", None)
    if (idle_seconds < 0 or getattr(owner, "_active", False) or last_used is None
            or (time.monotonic() if now is None else now) - last_used < idle_seconds):
        return False
    return owner.release_runtime()


def release_runtime(owner) -> bool:
    owner._last_used = None
    owner._prefix = None
    owner.resident = None
    runtime = owner._runtime
    if runtime is None:
        return False
    owner._runtime = None
    torch, device = runtime.torch, runtime.device
    del runtime
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return True
