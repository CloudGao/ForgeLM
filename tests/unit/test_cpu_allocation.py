from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_platform.services.tasks import _task_allocation


def test_auto_cpu_allocation_uses_remaining_capacity_and_respects_manual_request():
    worker = SimpleNamespace(id="worker", labels={}, capabilities=["adapter", "launcher"],
        resources={"cpu_count": 12, "gpus": []})
    session = MagicMock()
    class Rows(list):
        def all(self):
            return self
    session.scalars.return_value = Rows([SimpleNamespace(worker_id="worker", task_attempt_id="other", allocation={"cpus": 4, "gpu_indexes": []})])
    task = SimpleNamespace(kind="train", adapter_id="adapter", launcher_id="launcher", payload={"resource_request": {"cpus": 0}})
    assert _task_allocation(session, worker, task)["cpus"] == 8
    task.payload["resource_request"]["cpus"] = 10
    assert _task_allocation(session, worker, task) is None
    task.payload["resource_request"]["cpus"] = 2
    assert _task_allocation(session, worker, task)["cpus"] == 2
