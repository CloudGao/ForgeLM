import json

import pytest
from llm_platform_worker.adapters.supervision import (
    _conversation,
    _filter_conversation,
    _iter_records,
    _question_hash,
)


def test_indexed_choice_mapping_and_object_collection(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "question": "太阳是否是一颗恒星？",
                    "choices": "[Yes,No]",
                    "answer": 0,
                    "reason": "太阳是恒星。",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = list(_iter_records(path, None, records_layout="object_values"))
    assert len(rows) == 1
    record = _conversation(
        rows[0][2],
        {
            "prompt_field": "question",
            "choices_field": "choices",
            "answer_index_field": "answer",
            "explanation_field": "reason",
            "choice_text_map": {"Yes": "是", "No": "否"},
        },
    )
    assert record["messages"][1]["content"] == "是。太阳是恒星。"
    _filter_conversation(record, {"min_chinese_ratio": 0.9})


def test_language_task_filter_and_shared_split_identity():
    a = {
        "messages": [
            {"role": "user", "content": "什么是 光合作用？"},
            {"role": "assistant", "content": "植物利用光能进行的过程。"},
        ]
    }
    b = {
        "messages": [
            {"role": "user", "content": "什么是光合作用？"},
            {"role": "assistant", "content": "另一个答案。"},
        ]
    }
    assert _question_hash(a) == _question_hash(b)
    _filter_conversation(a, {"min_chinese_ratio": 0.9, "hash_split": {"start": 0, "end": 10000}})
    a["messages"][0]["content"] = "请翻译成英文：光合作用"
    with pytest.raises(ValueError, match="language_learning_task"):
        _filter_conversation(a, {"exclude_language_learning": True})
    a["messages"][0]["content"] = "What is photosynthesis?"
    with pytest.raises(ValueError, match="non_chinese_content"):
        _filter_conversation(a, {"min_chinese_ratio": 0.8})


def test_field_parts_build_grounded_topic_and_evidence_completion():
    record = {
        "question": "亚洲的大部分区域位于哪个半球？",
        "messages": [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧答案"},
        ],
        "source_nodes": [{"concept": "亚洲"}, {"concept": "亚洲"}],
        "planning_evidence": [
            {"quote": "主体位于北半球和东半球"},
            {"quote": "纬度跨度从赤道附近延伸到北极圈以北"},
        ],
    }
    mapped = _conversation(
        record,
        {
            "prompt_parts": [{"field": {"path": "question"}}],
            "completion_parts": [
                {"literal": "主题："},
                {"field": {"path": "source_nodes.concept", "join": "、"}},
                {"literal": "\n原文片段："},
                {"field": {"path": "planning_evidence.quote", "join": "\n"}},
            ],
        },
    )
    assert mapped == {
        "messages": [
            {"role": "user", "content": "亚洲的大部分区域位于哪个半球？"},
            {
                "role": "assistant",
                "content": "主题：亚洲\n原文片段：主体位于北半球和东半球\n纬度跨度从赤道附近延伸到北极圈以北",
            },
        ]
    }
