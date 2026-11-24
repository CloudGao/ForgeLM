from copy import deepcopy

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_platform.models import TaskEvent
from llm_platform.services.training_metrics import (
    merge_training_metrics,
    training_run_metrics_view,
)


def test_completion_preserves_curve_and_last_loss_instead_of_training_average():
    history = [{"step": 10, "loss": 3.1}]
    current = {"loss": 3.1, "eval_loss": 3.2, "history": history}
    final = {"train_loss": 3.5, "epoch": 1.0, "history": []}
    merged = merge_training_metrics(current, final)
    assert merged == {**current, "train_loss": 3.5, "epoch": 1.0}
    assert final["history"] == []


def test_completed_history_is_recovered_from_own_events_and_sampled_across_full_run():
    engine = create_engine("sqlite://")
    TaskEvent.__table__.create(engine)
    try:
        with Session(engine) as session:
            session.add_all([
                TaskEvent(task_id="ours", sequence=step, event_type="progress",
                          payload={"step": step, "tokens": step * 100,
                                   "metrics": {"loss": 10 - step / 100}})
                for step in range(1, 1002)
            ])
            session.add_all([
                TaskEvent(task_id="ours", sequence=1002, event_type="progress",
                          payload={"step": 1001, "metrics": {"train_loss": 20}}),
                TaskEvent(task_id="ours", sequence=1003, event_type="progress",
                          payload={"step": 1001, "metrics": {"eval_loss": 30}}),
                TaskEvent(task_id="other", sequence=1, event_type="progress",
                          payload={"step": 5000, "metrics": {"loss": 999}}),
            ])
            session.commit()
            run = {"task_id": "ours", "latest_metrics": {"train_loss": 20}}
            original = deepcopy(run)
            result = training_run_metrics_view(session, run)
            metrics = result["latest_metrics"]
            assert len(metrics["history"]) == 500
            assert metrics["history_total_points"] == 1001
            assert metrics["history_sampled"] is True
            assert metrics["history"][0]["step"] == 1
            assert metrics["history"][-1]["step"] == 1001
            assert metrics["loss"] == 10 - 1001 / 100
            assert metrics["train_loss"] == 20
            assert run == original
            assert not session.dirty
    finally:
        engine.dispose()


def test_imported_history_without_events_is_retained_and_cross_entropy_is_supported():
    engine = create_engine("sqlite://")
    TaskEvent.__table__.create(engine)
    try:
        with Session(engine) as session:
            old = {"task_id": "imported", "latest_metrics": {"history": [{"step": 1, "loss": 2}]}}
            assert training_run_metrics_view(session, old) == old
            session.add(TaskEvent(task_id="ce", sequence=1, event_type="progress",
                                  payload={"step": 5, "metrics": {"cross_entropy": 2.5}}))
            session.commit()
            metrics = training_run_metrics_view(session, {"task_id": "ce"})["latest_metrics"]
            assert metrics["loss"] == 2.5
            assert metrics["history_sampled"] is False
    finally:
        engine.dispose()
