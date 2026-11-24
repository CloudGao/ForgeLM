"""Shared bounded-concurrency primitives for Worker batch runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


async def bounded_map_unordered(
    items: Iterable[InputT],
    worker: Callable[[InputT], Awaitable[OutputT]],
    concurrency: int,
) -> AsyncIterator[OutputT]:
    """Yield results as they finish while keeping at most ``concurrency`` tasks alive."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    iterator = iter(items)
    pending: set[asyncio.Task[OutputT]] = set()

    def submit_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.add(asyncio.create_task(worker(item)))
        return True

    try:
        for _ in range(concurrency):
            if not submit_one():
                break
        while pending:
            finished, still_pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            pending = set(still_pending)
            for task in finished:
                result = task.result()
                submit_one()
                yield result
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
