import json

import pytest

from llm_platform_core.data_generation import DataGenerationCreate
from llm_platform_core.dense_relations import prompt, schema, validate
from llm_platform_worker.dense_relations_runtime import imported_relation_state


def config(**changes):
    values = dict(name="relations", input_dataset_version_id="mother", model_weight_version_id="model",
                  prompt_version="dense-relations-v1", engine="vllm", document_mode="whole",
                  text_field="dense_note", styles=["textbook"], batch_size=8,
                  qa_limit=1000, qa_primary_limit=20, qa_source_mode="single", relation_per_node_limit=50)
    values.update(changes)
    return DataGenerationCreate(**values)


def test_relation_contract_requires_manual_concurrency():
    assert config().relation_per_node_limit == 50
    with pytest.raises(ValueError, match="手动并发"):
        config(batch_size="auto")


def test_schema_prompt_and_validation_require_two_distinct_quotes():
    source = {"id": "n1", "concept": "海洋", "dense_note": "海水受热后蒸发进入大气。水汽冷却凝结后形成降水。降水又可汇入海洋。"}
    contract = schema(source, 50)
    assert contract["properties"]["items"]["maxItems"] == 50
    assert contract["properties"]["items"]["items"]["properties"]["evidence_ids"]["items"]["enum"] == [1, 2, 3]
    assert "尽可能多生成" in prompt(source, 50)
    value = {"items": [{"relation_type": "system_relation", "statement": "海洋蒸发形成的水汽可经凝结降水重新汇入海洋，构成水的循环路径。",
                         "evidence_ids": [1, 2, 3]}]}
    assert validate(value, source, 50)[0]["evidence"] == ["海水受热后蒸发进入大气。", "水汽冷却凝结后形成降水。", "降水又可汇入海洋。"]
    value["items"][0]["evidence_ids"] = [1, 1]
    with pytest.raises(ValueError, match="distinct"):
        validate(value, source, 20)


def test_partial_batches_are_counted_for_resume(tmp_path):
    path = tmp_path / "data.partial"
    path.write_text(json.dumps({"metadata": {"source_ordinal": 3}, "relation_records": [{}, {}]}) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        assert imported_relation_state(stream) == (1, 2, {3})
