from datetime import UTC, datetime, timedelta

from llm_platform.models import Task, TaskAttempt
from llm_platform.services.tasks import estimate_task_timing

NOW = datetime(2026, 8, 10, 4, 0, tzinfo=UTC)


def task(state: str = "running", progress: float = 0.5) -> Task:
    return Task(kind="train", title="训练", observed_state=state, progress=progress)


def attempt(number: int, started_seconds_ago: int, ended_seconds_ago: int | None = None) -> TaskAttempt:
    return TaskAttempt(
        attempt_number=number,
        status="running",
        started_at=NOW - timedelta(seconds=started_seconds_ago),
        ended_at=(NOW - timedelta(seconds=ended_seconds_ago)) if ended_seconds_ago is not None else None,
    )


def test_first_attempt_uses_start_as_zero_progress_baseline() -> None:
    estimate = estimate_task_timing(
        task(progress=0.5),
        [attempt(1, 600)],
        [(NOW - timedelta(seconds=300), 0.25), (NOW, 0.5)],
        now=NOW,
    )

    assert estimate["status"] == "available"
    assert estimate["actual_duration_seconds"] == 600
    assert estimate["estimated_total_seconds"] == 1200
    assert estimate["remaining_seconds"] == 600
    assert estimate["estimated_completion_at"] == (NOW + timedelta(seconds=600)).isoformat()


def test_preparation_plateau_is_not_multiplied_across_remaining_work() -> None:
    estimate = estimate_task_timing(
        task(progress=0.01),
        [attempt(1, 240)],
        [
            (NOW - timedelta(seconds=240), 0.0),
            (NOW - timedelta(seconds=60), 0.0),
            (NOW, 0.01),
        ],
        now=NOW,
    )

    assert estimate["status"] == "available"
    assert estimate["remaining_seconds"] == 5940
    assert estimate["estimated_total_seconds"] == 6180


def test_resumed_attempt_requires_two_real_progress_samples() -> None:
    previous = attempt(1, 900, 600)
    resumed = attempt(2, 300)

    collecting = estimate_task_timing(
        task(progress=0.55),
        [resumed, previous],
        [(NOW, 0.55)],
        now=NOW,
    )
    available = estimate_task_timing(
        task(progress=0.55),
        [resumed, previous],
        [(NOW - timedelta(seconds=120), 0.5), (NOW, 0.55)],
        now=NOW,
    )

    assert collecting["status"] == "estimating"
    assert collecting["remaining_seconds"] is None
    assert available["status"] == "available"
    assert available["actual_duration_seconds"] == 600
    assert available["remaining_seconds"] == 1080
    assert available["estimated_total_seconds"] == 1680


def test_slow_long_task_does_not_require_a_fixed_progress_percentage() -> None:
    estimate = estimate_task_timing(
        task(progress=0.004),
        [attempt(1, 1800)],
        [(NOW - timedelta(seconds=1800), 0.0), (NOW, 0.004)],
        now=NOW,
    )

    assert estimate["status"] == "available"
    assert estimate["remaining_seconds"] == 448200
    assert estimate["estimated_completion_at"] == (
        NOW + timedelta(seconds=448200)
    ).isoformat()


def test_timing_stays_unavailable_when_progress_does_not_increase() -> None:
    estimate = estimate_task_timing(
        task(progress=0.2),
        [attempt(2, 120)],
        [(NOW - timedelta(seconds=60), 0.2), (NOW, 0.2)],
        now=NOW,
    )

    assert estimate["status"] == "estimating"
    assert estimate["remaining_seconds"] is None


def test_paused_task_reports_resume_time_without_completion_clock() -> None:
    estimate = estimate_task_timing(
        task(state="paused", progress=0.5),
        [attempt(1, 600, 0)],
        [(NOW - timedelta(seconds=300), 0.25), (NOW, 0.5)],
        now=NOW,
    )

    assert estimate["status"] == "paused"
    assert estimate["remaining_seconds"] == 600
    assert estimate["estimated_completion_at"] is None


def test_running_task_discards_progress_older_than_the_recent_window() -> None:
    estimate = estimate_task_timing(
        task(progress=0.5),
        [attempt(1, 7200)],
        [
            (NOW - timedelta(seconds=3700), 0.25),
            (NOW - timedelta(seconds=3600), 0.5),
        ],
        now=NOW,
    )

    assert estimate["status"] == "estimating"
    assert estimate["estimated_completion_at"] is None
    assert estimate["sample_count"] == 0


def test_paused_task_keeps_the_last_running_window_even_when_pause_is_old() -> None:
    estimate = estimate_task_timing(
        task(state="paused", progress=0.5),
        [attempt(1, 7200, 3600)],
        [
            (NOW - timedelta(seconds=3700), 0.25),
            (NOW - timedelta(seconds=3600), 0.5),
        ],
        now=NOW,
    )

    assert estimate["status"] == "paused"
    assert estimate["remaining_seconds"] == 200
    assert estimate["estimated_completion_at"] is None


def test_terminal_task_reports_actual_attempt_runtime() -> None:
    estimate = estimate_task_timing(
        task(state="succeeded", progress=1),
        [attempt(1, 900, 300)],
        [],
        now=NOW,
    )

    assert estimate == {
        "status": "complete",
        "actual_duration_seconds": 600,
        "estimated_total_seconds": None,
        "remaining_seconds": 0,
        "estimated_completion_at": None,
        "sample_count": 0,
    }
