"""Commit-triggered delivery notifications for the single-process control plane.

Notifications are hints; tasks and leases remain the durable source of truth.
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.orm import Session

allocation_lock = threading.RLock()
_lock = threading.Lock()
_listeners: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]] = {}


@contextmanager
def subscribe(worker_id: str):
    entry = (asyncio.get_running_loop(), asyncio.Event())
    with _lock:
        _listeners.setdefault(worker_id, set()).add(entry)
    try:
        yield entry[1]
    finally:
        with _lock:
            _listeners[worker_id].discard(entry)
            if not _listeners[worker_id]:
                del _listeners[worker_id]


def notify(worker_id: str | None = None) -> None:
    with _lock:
        entries = list(_listeners.get(worker_id, ())) if worker_id else [
            item for group in _listeners.values() for item in group
        ]
    for loop, signal in entries:
        try:
            loop.call_soon_threadsafe(signal.set)
        except RuntimeError:
            pass  # A disconnected/restarted server loop cannot receive a hint.


def notify_after_commit(session: Session) -> None:
    session.info["worker_dispatch_pending"] = True


@event.listens_for(Session, "after_commit")
def _committed(session: Session) -> None:
    if session.info.pop("worker_dispatch_pending", False):
        notify()


@event.listens_for(Session, "after_rollback")
def _rolled_back(session: Session) -> None:
    session.info.pop("worker_dispatch_pending", None)
