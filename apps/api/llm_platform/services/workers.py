from __future__ import annotations

from ..models import WorkerNode


def worker_view(worker: WorkerNode) -> dict:
    """Return the public Worker representation without authentication state."""

    view = worker.to_dict()
    view.pop("token_hash", None)
    return view
