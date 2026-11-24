import asyncio

import pytest

from llm_platform_worker.bounded_async import bounded_map_unordered


def test_bounded_map_never_prefetches_or_runs_beyond_limit():
    async def run():
        produced = 0
        active = 0
        peak = 0
        release = asyncio.Event()

        def inputs():
            nonlocal produced
            for value in range(20):
                produced += 1
                yield value

        async def worker(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1
            return value

        stream = bounded_map_unordered(inputs(), worker, 3)
        first_result = asyncio.create_task(anext(stream))
        while active < 3:
            await asyncio.sleep(0)
        assert produced == 3
        assert peak == 3
        release.set()
        assert await first_result in range(3)
        results = [value async for value in stream]
        assert len(results) == 19
        assert peak == 3

    asyncio.run(run())


def test_bounded_map_cancels_pending_work_after_error():
    async def run():
        cancelled = 0

        async def worker(value):
            nonlocal cancelled
            if value == 0:
                raise RuntimeError("boom")
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled += 1
                raise

        with pytest.raises(RuntimeError, match="boom"):
            async for _ in bounded_map_unordered(range(10), worker, 4):
                pass
        assert cancelled == 3

    asyncio.run(run())
