from __future__ import annotations

from itertools import pairwise

import pytest
from llm_platform_core.domain import InvalidTaskTransition, ensure_task_transition


def test_happy_path_transitions_are_allowed() -> None:
    path = ["queued", "assigned", "starting", "running", "succeeded"]
    for current, target in pairwise(path):
        ensure_task_transition(current, target)


def test_terminal_task_cannot_return_to_running() -> None:
    with pytest.raises(InvalidTaskTransition):
        ensure_task_transition("succeeded", "running")


def test_checkpoint_pause_returns_through_queue() -> None:
    path = ["running", "pausing", "paused", "queued"]
    for current, target in pairwise(path):
        ensure_task_transition(current, target)


def test_cancelled_task_can_be_explicitly_requeued_for_a_new_attempt() -> None:
    ensure_task_transition("cancelled", "queued")
