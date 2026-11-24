import pytest

from llm_platform_worker.training_runtime import cache_backend_availability, runtime_command_args


@pytest.mark.parametrize("present", [False, True])
def test_backend_probe_once_but_dispatch_verifiers_remain_dynamic(present):
    class Backend:
        probes = 0
        enabled = True

        @classmethod
        def is_available(cls):
            cls.probes += 1
            return present

        @classmethod
        def is_enabled(cls):
            return cls.enabled

        def verify(self, size):
            return size < 8

    assert cache_backend_availability(Backend)
    assert not cache_backend_availability(Backend)
    for _ in range(84):
        assert Backend.is_available() is present
    assert Backend.probes == 1
    Backend.enabled = False
    assert not Backend.is_enabled()
    assert Backend().verify(1)
    assert not Backend().verify(16)


def test_availability_exceptions_are_not_cached():
    class Backend:
        @classmethod
        def is_available(cls):
            raise RuntimeError("broken runtime")

    cache_backend_availability(Backend)
    with pytest.raises(RuntimeError, match="broken runtime"):
        Backend.is_available()


@pytest.mark.parametrize("stage", ["pretrain", "sft", "dpo"])
def test_every_training_stage_gets_same_runtime_without_workflow_opt_in(stage):
    task = {"kind": "train", "payload": {"stages": [{"type": stage}],
            "command": {"executable": "__worker_python__"}}}
    assert runtime_command_args(task, ["train.py", "--config", "a.json"]) == [
        "-m", "llm_platform_worker.training_runtime", "train.py", "--config", "a.json"]


@pytest.mark.parametrize("kind,executable,args,env", [
    ("evaluate", "__worker_python__", ["train.py"], {}),
    ("train", "external-python", ["train.py"], {}),
    ("train", "__worker_python__", ["-c", "pass"], {}),
    ("train", "__worker_python__", ["train.py"], {"FORGELM_TRAINING_RUNTIME": "off"}),
])
def test_custom_runtimes_and_explicit_opt_out_are_preserved(kind, executable, args, env):
    task = {"kind": kind, "payload": {"command": {"executable": executable, "env": env}}}
    assert runtime_command_args(task, args) == args
