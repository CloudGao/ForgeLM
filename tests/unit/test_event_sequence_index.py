import json

from llm_platform.models import TaskEvent, numeric_task_progress, worker_event_sequence
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import sqlite


def test_idempotency_lookup_uses_expression_index_with_large_attempt():
    engine = create_engine("sqlite://")
    TaskEvent.__table__.create(engine)
    with engine.begin() as c:
        c.exec_driver_sql(
            "INSERT INTO task_events(task_id,attempt_id,sequence,event_type,payload,created_at) "
            "VALUES ('task','attempt',?,'log',?,CURRENT_TIMESTAMP)",
            [(i, json.dumps({"worker_sequence": i})) for i in range(10000)],
        )
        q = select(TaskEvent.id).where(TaskEvent.task_id == "task",
            TaskEvent.attempt_id == "attempt", worker_event_sequence() == 9999)
        sql = str(q.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
        plan = str(c.exec_driver_sql("EXPLAIN QUERY PLAN " + sql).all())
        assert "ix_task_event_worker_sequence" in plan
        assert c.execute(q).scalar_one() == 10000
        progress = select(TaskEvent.id).where(TaskEvent.task_id == "task",
            TaskEvent.attempt_id == "attempt", numeric_task_progress()).order_by(TaskEvent.sequence.desc()).limit(120)
        sql = str(progress.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
        plan = str(c.exec_driver_sql("EXPLAIN QUERY PLAN " + sql).all())
        assert "ix_task_event_numeric_progress" in plan
        assert c.execute(progress).all() == []
