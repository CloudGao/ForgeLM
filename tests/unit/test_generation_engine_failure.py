import asyncio
import gc
import sys
from types import ModuleType, SimpleNamespace

import pytest
from llm_platform_core.data_generation import DataGenerationCreate
from llm_platform_worker.data_generation_runtime import (
    generate_records_vllm_async,
    shutdown_generation_engine,
)


@pytest.mark.parametrize("phase", ["production", "autotune"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_engine_death_drains_finished_and_pending_requests(
    tmp_path, monkeypatch, phase, cleanup_fails
):
    """A shared fatal error must not produce 127 unobserved traceback copies."""
    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)

    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    fatal = RuntimeError("engine died: injected CUDA failure")
    state = dict(requests=0, pending_cancelled=False, shutdown=0, synchronize=0)

    class Engine:
        @classmethod
        def from_engine_args(cls, args):
            return cls()

        async def generate(self, prompt, params, request_id):
            state["requests"] += 1
            number = state["requests"]
            if phase == "autotune" and number == 1:
                yield SimpleNamespace(outputs=[SimpleNamespace(
                    text="这是一段完整的中文事实说明。", token_ids=[1] * 8,
                    finish_reason="stop",
                )])
                return
            # Leave one sibling pending when all the others fail together.
            if number == 128 + int(phase == "autotune"):
                try:
                    await asyncio.Future()
                finally:
                    state["pending_cancelled"] = True
            raise fatal

        def shutdown(self):
            state["shutdown"] += 1
            if cleanup_fails:
                raise RuntimeError("secondary cleanup failure")

    def synchronize():
        state["synchronize"] += 1

    module("torch", cuda=SimpleNamespace(synchronize=synchronize))
    module("vllm", __version__="test", SamplingParams=lambda **kw: SimpleNamespace(**kw))
    module("transformers", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
    module("vllm.engine")
    module("vllm.engine.arg_utils", AsyncEngineArgs=lambda **kw: kw)
    module("vllm.sampling_params", RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"))
    module("vllm.v1")
    module("vllm.v1.engine")
    module("vllm.v1.engine.async_llm", AsyncLLM=Engine)
    cfg = DataGenerationCreate(
        name="failure-test", input_dataset_version_id="data", model_weight_version_id="model",
        engine="vllm", batch_size="auto" if phase == "autotune" else 128,
        max_batch_size=128, max_new_tokens=1024, enable_prefix_caching=False,
    )
    source = iter(dict(prompt="事实", source_file="source.jsonl", source_ordinal=i,
                       source_sha256=f"hash-{i}", character_start=0, character_end=2,
                       style="retrospective") for i in range(150))
    output = tmp_path / "data.partial"
    checkpoint = '{"text":"已保存的正文"}\n'
    output.write_text(checkpoint, encoding="utf-8")
    unhandled = []

    async def run():
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        try:
            with output.open("a", encoding="utf-8") as stream:
                with pytest.raises(RuntimeError, match="engine died") as caught:
                    await asyncio.wait_for(generate_records_vllm_async(
                        "fake", source, cfg, stream, {}, {"length_retries": 0},
                    ), timeout=2)
                assert caught.value is fatal
            # Drop tracebacks retaining the task sets so unobserved exceptions
            # cannot hide until after the loop and its handler are gone.
            fatal.__traceback__ = None
            del caught
            gc.collect()
            await asyncio.sleep(0)
            assert not (asyncio.all_tasks() - {asyncio.current_task()})
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(run())
    assert unhandled == []
    assert state == dict(requests=128 + int(phase == "autotune"), pending_cancelled=True,
                         shutdown=1, synchronize=0)
    assert output.read_text(encoding="utf-8") == checkpoint


def test_successful_generation_does_not_hide_shutdown_failure():
    def fail():
        raise RuntimeError("cleanup failure")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        shutdown_generation_engine(SimpleNamespace(shutdown=fail))
