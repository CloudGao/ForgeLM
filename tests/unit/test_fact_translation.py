import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from llm_platform_core.data_generation import (
    FactTranslationCreate,
    factual_surface_anchors,
    validate_factual_surface,
)
from llm_platform_core.fact_translation import (
    finish_translation,
    translation_plan,
    translation_schema,
)
from llm_platform_worker.data_generation_runtime import generate_records_vllm_async, jobs, records


def cfg(**kw):
    return FactTranslationCreate(
        name="translated", input_dataset_version_id="input", model_weight_version_id="weight", **kw
    )


def test_structure_and_known_relations_cannot_be_changed_by_model():
    facts = [
        {
            "主体": "John Doe",
            "关系": "member of sports team",
            "客体": "Example FC",
            "限定": "不保证当前状态",
        },
        {"主体": "John Doe", "关系": "职业为", "客体": "football player"},
    ]
    plan = translation_plan(json.dumps(facts), "wikidata5m")
    assert plan["slots"] == {"t0": "football player"}
    result = json.loads(finish_translation('{"t0":"足球运动员"}', plan))
    assert result == [{**facts[0], "关系": "所属运动队为"}, {**facts[1], "客体": "足球运动员"}]
    assert facts == plan["facts"]
    assert translation_schema(plan)["additionalProperties"] is False
    anchors = factual_surface_anchors(
        json.dumps(result), "facts", "wikidata5m", proper_name_policy="source_exact"
    )
    assert validate_factual_surface("John Doe 是 Example FC 的成员，是足球运动员。", anchors)


@pytest.mark.parametrize(
    "value,reason",
    [
        ("{}", "fields_mismatch"),
        ('{"t0":"甲","t1":"乙"}', "fields_mismatch"),
        ('{"t0":""}', "empty_field"),
        ('{"t0":"holds 2 apples"}', "untranslated_words"),
        ('{"t0":"拿着3个苹果"}', "number_mismatch"),
        ("[]", "fields_mismatch"),
        ('{"t0":null}', "empty_field"),
        ("not json", "invalid_json"),
    ],
)
def test_structural_missing_extra_number_and_language_errors_rejected(value, reason):
    plan = translation_plan(
        '[{"主体":"某人","关系":"可能具有的性质","客体":"holds 2 apples"}]', "atomic2020"
    )
    with pytest.raises(ValueError, match=reason):
        finish_translation(value, plan)


def test_repeated_fields_and_original_chinese_qualifiers_are_preserved():
    facts = [
        {
            "主体": "某人 leaves",
            "关系": "行动者可能的意图",
            "客体": "get home",
            "限定": "不保证发生",
        },
        {"主体": "某人 leaves", "关系": "行动者后续可能的意愿", "客体": "get home"},
    ]
    plan = translation_plan(json.dumps(facts), "atomic2020")
    assert len(plan["slots"]) == 2
    out = json.loads(finish_translation('{"t0":"某人离开","t1":"回家"}', plan))
    assert len(out) == 2 and out[0]["限定"] == "不保证发生" and out[1]["主体"] == "某人离开"
    with pytest.raises(ValueError, match="changed_chinese"):
        finish_translation('{"t0":"张三离开","t1":"回家"}', plan)


@pytest.mark.parametrize(
    "kw",
    [
        {"engine": "transformers"},
        {"document_mode": "chunk"},
        {"validation_retry_limit": 1},
        {"styles": ["textbook"]},
    ],
)
def test_node_rejects_unsupported_modes(kw):
    with pytest.raises(ValueError):
        cfg(**kw)


def test_native_runtime_translates_once_retains_provenance_and_skips_invalid(tmp_path, monkeypatch):
    def module(name, **attrs):
        obj = ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)

    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    class Engine:
        calls = []

        @classmethod
        def from_engine_args(cls, args):
            return cls()

        async def generate(self, prompt, params, request_id):
            self.calls.append(prompt)
            assert params.structured_outputs.json["additionalProperties"] is False
            text = '{"t0":"wrong English"}' if "bad value" in prompt else '{"t0":"足球运动员"}'
            yield SimpleNamespace(
                outputs=[SimpleNamespace(text=text, token_ids=[1] * 8, finish_reason="stop")]
            )

        def shutdown(self):
            pass

    module("torch", cuda=SimpleNamespace(synchronize=lambda: None))
    module(
        "transformers", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
    )
    module("vllm", __version__="test", SamplingParams=lambda **kw: SimpleNamespace(**kw))
    module("vllm.engine")
    module("vllm.engine.arg_utils", AsyncEngineArgs=lambda **kw: kw)
    module(
        "vllm.sampling_params",
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
        StructuredOutputsParams=lambda **kw: SimpleNamespace(**kw),
    )
    module("vllm.v1")
    module("vllm.v1.engine")
    module("vllm.v1.engine.async_llm", AsyncLLM=Engine)
    inputs = tmp_path / "input.jsonl"
    inputs.write_text(
        "".join(
            json.dumps(
                {
                    "source": "wikidata5m",
                    "text": json.dumps([{"主体": "John Doe", "关系": "职业为", "客体": obj}]),
                }
            )
            + "\n"
            for obj in ["football player", "政治人物", "bad value"]
        ),
        encoding="utf-8",
    )
    config = cfg(batch_size=2, max_batch_size=2)
    report = {"length_retries": 0}
    with (
        (tmp_path / "out.jsonl").open("w", encoding="utf-8") as output,
        (tmp_path / "skips").open("w", encoding="utf-8") as skipped,
    ):
        asyncio.run(
            generate_records_vllm_async(
                "model",
                iter(jobs({"runtime_path": str(inputs)}, config, "fact_translation")),
                config,
                output,
                {},
                report,
                set(),
                skipped,
                telemetry_dir=tmp_path,
            )
        )
    rows = [
        json.loads(x) for x in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2 and len(Engine.calls) == 2
    assert report["validation_retries"] == 0 and report["raw_completed_requests"] == 2
    assert all(row["metadata"]["translation_protocol"] == "fact-fields-zh-v1" for row in rows)
    assert all(row["metadata"]["source_fact_count"] == 1 for row in rows)
    assert all(
        source == "wikidata5m"
        for _, source in records(tmp_path / "out.jsonl", "text", with_source=True)
    )
    assert "fact_translation_untranslated_words" in (tmp_path / "skips").read_text(encoding="utf-8")


@pytest.mark.parametrize("original,expected", [("to leave where they are", "离开原地"), ("to leave the area", "离开该区域")])
def test_unambiguous_direction_is_fixed_before_decoding(original, expected):
    text = json.dumps([{"主体":"某人回到车旁","关系":"行动者可能的意图","客体":original}])
    plan = translation_plan(text, "atomic2020")
    assert not plan["slots"]
    assert json.loads(finish_translation('{}', plan))[0]["客体"] == expected


def test_leave_behind_and_annual_leave_are_not_reinterpreted():
    for phrase in ["annual leave", "leave the keys behind"]:
        plan = translation_plan(json.dumps([{"主体":"某人","关系":"可能具有的性质","客体":phrase}]), "atomic2020")
        assert plan["slots"] == {"t0":phrase}
