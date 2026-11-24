from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.LOST,
}

TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.QUEUED: {TaskStatus.ASSIGNED, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.ASSIGNED: {
        TaskStatus.STARTING,
        TaskStatus.QUEUED,
        TaskStatus.FAILED,
        TaskStatus.LOST,
    },
    TaskStatus.STARTING: {
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.STOPPING,
        TaskStatus.LOST,
    },
    TaskStatus.RUNNING: {
        TaskStatus.PAUSING,
        TaskStatus.STOPPING,
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.LOST,
    },
    TaskStatus.PAUSING: {
        TaskStatus.PAUSED,
        TaskStatus.FAILED,
        TaskStatus.STOPPING,
        TaskStatus.LOST,
    },
    TaskStatus.PAUSED: {TaskStatus.QUEUED, TaskStatus.STOPPING, TaskStatus.CANCELLED},
    TaskStatus.STOPPING: {
        TaskStatus.SUCCEEDED,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
        TaskStatus.LOST,
    },
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: {TaskStatus.QUEUED},
    # A cancelled training task may be explicitly requeued as a new attempt.
    # The control-plane endpoint restricts this edge to a stable TrainingRun.
    TaskStatus.CANCELLED: {TaskStatus.QUEUED},
    TaskStatus.LOST: {TaskStatus.QUEUED, TaskStatus.FAILED},
}


class InvalidTaskTransition(ValueError):
    pass


def ensure_task_transition(current: str, target: str) -> None:
    source = TaskStatus(current)
    destination = TaskStatus(target)
    if destination not in TASK_TRANSITIONS[source]:
        raise InvalidTaskTransition(f"Task cannot transition from {source} to {destination}")


class TaskKind(StrEnum):
    DATA_GENERATE = "data_generate"
    DATA_IMPORT = "data_import"
    DATA_QUERY = "data_query"
    DATA_CONVERT = "data_convert"
    DATA_PROCESS = "data_process"
    TOKENIZER_TRAIN = "tokenizer_train"
    TRAIN = "train"
    EVALUATE = "evaluate"
    QUANTIZE = "quantize"
    EXPORT = "export"
    RELEASE_EXPORT = "release_export"
    RELEASE_PUBLISH = "release_publish"
    INFERENCE = "inference"
    BENCHMARK = "benchmark"


class PluginKind(StrEnum):
    DATA_CONNECTOR = "data_connector"
    DATA_PROCESSOR = "data_processor"
    TOKENIZER = "tokenizer"
    TRAINING_ENGINE = "training_engine"
    EVALUATION = "evaluation"
    QUANTIZATION = "quantization"
    RELEASE_EXPORTER = "release_exporter"
    RELEASE_PUBLISHER = "release_publisher"
    INFERENCE = "inference"
    LAUNCHER = "launcher"


class GateSeverity(StrEnum):
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"
