import json

import pytest
from llm_platform_core.data_generation import DataGenerationCreate
from llm_platform_core.dense_dialogue import node_quota, prompt, schema, validate
from llm_platform_worker.dense_dialogue_runtime import imported_dialogue_state


def config(**changes):
    values = dict(name="dialogue", input_dataset_version_id="mother", model_weight_version_id="model",
                  prompt_version="dense-dialogue-v1", engine="vllm", document_mode="whole",
                  text_field="dense_note", styles=["textbook"], batch_size=8,
                  qa_limit=8, qa_primary_limit=2, qa_source_mode="single",
                  dialogue_single_target=4, dialogue_multi_target=2, dialogue_natural_target=2)
    values.update(changes)
    return DataGenerationCreate(**values)


def test_exact_targets_are_spread_over_nodes():
    targets = {"single_qa": 38, "multi_qa": 19, "natural_chat": 19}
    quotas = [node_quota(i, 13, targets) for i in range(13)]
    assert {kind: sum(q[kind] for q in quotas) for kind in targets} == targets


def test_imported_dialogues_reduce_resume_targets(tmp_path):
    path = tmp_path / "data.partial"
    rows = [
        {"dialogue_kind": "single_qa"},
        {"dialogue_kind": "multi_qa"},
        {"dialogue_kind": "natural_chat"},
    ]
    path.write_text(json.dumps({"dialogue_records": rows}) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        nodes, counts, ordinals = imported_dialogue_state(
            stream, {"single_qa": 4, "multi_qa": 2, "natural_chat": 2}
        )
    assert nodes == 1
    assert counts == {"single_qa": 1, "multi_qa": 1, "natural_chat": 1}
    assert ordinals == set()


def test_imported_dialogues_allow_prior_overproduction(tmp_path):
    path = tmp_path / "data.partial"
    rows = [{"dialogue_kind": "single_qa"}, {"dialogue_kind": "single_qa"}]
    path.write_text(json.dumps({"dialogue_records": rows}) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        nodes, counts, _ = imported_dialogue_state(
            stream, {"single_qa": 1, "multi_qa": 0, "natural_chat": 0}
        )
    assert nodes == 1
    assert counts["single_qa"] == 2


def test_dialogue_contract_requires_manual_batch_and_matching_total():
    assert config().dialogue_multi_target == 2
    with pytest.raises(ValueError, match="之和"):
        config(qa_limit=9)
    with pytest.raises(ValueError, match="手动并发"):
        config(batch_size="auto")


def test_schema_prompt_and_validation_preserve_verbatim_evidence():
    source = {"id": "n1", "concept": "亚洲", "dense_note": "亚洲是地球面积最大、人口最多的大洲。主体位于北半球和东半球。"}
    quota = {"single_qa": 1, "multi_qa": 1, "natural_chat": 1}
    contract = schema(source, quota)
    assert contract["properties"]["items"]["minItems"] == 3
    assert contract["properties"]["items"]["items"]["properties"]["evidence"]["items"]["properties"]["quote"]["enum"] == [
        "亚洲是地球面积最大、人口最多的大洲。", "主体位于北半球和东半球。"
    ]
    assert "百科单轮1条" in prompt(source, quota)
    pairs = [
        {"role": "user", "content": "亚洲有什么突出特点？"},
        {"role": "assistant", "content": "亚洲是面积最大、人口最多的大洲。"},
    ]
    six = pairs + [
        {"role": "user", "content": "它主要位于哪些半球？"},
        {"role": "assistant", "content": "主体位于北半球和东半球。"},
        {"role": "user", "content": "把这两点合起来怎么概括？"},
        {"role": "assistant", "content": "亚洲规模和人口突出，主体位于北半球和东半球。"},
    ]
    value = {"items": [
        {"kind": "single_qa", "evidence": [{"quote": "亚洲是地球面积最大、人口最多的大洲"}], "messages": pairs},
        {"kind": "multi_qa", "evidence": [{"quote": "主体位于北半球和东半球"}], "messages": six},
        {"kind": "natural_chat", "evidence": [{"quote": "主体位于北半球和东半球"}], "messages": six},
    ]}
    assert len(validate(value, source, quota)) == 3
    value["items"][0]["evidence"][0]["quote"] = "亚洲位于南半球"
    with pytest.raises(ValueError, match="verbatim"):
        validate(value, source, quota)
