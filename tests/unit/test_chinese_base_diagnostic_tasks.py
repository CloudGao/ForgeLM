from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from llm_platform_worker.lm_eval_tasks.zh_base_v2 import utils


class _Dataset:
    @staticmethod
    def from_list(rows: list[dict]) -> list[dict]:
        return rows


@pytest.fixture(autouse=True)
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=_Dataset))


def test_chid_expands_each_blank_and_teacher_forces_earlier_answers() -> None:
    rows = utils.process_chid(
        [
            {
                "idx": 7,
                "candidates": ["瓜熟蒂落", "画蛇添足", "急功近利"],
                "content": ["事情终于#idiom1#，不要再#idiom2#。"],
                "answers": {
                    "text": ["瓜熟蒂落", "画蛇添足"],
                    "candidate_id": [0, 1],
                },
            }
        ]
    )

    assert [row["target"] for row in rows] == [0, 1]
    assert rows[0]["prefix"] == "事情终于"
    assert rows[1]["prefix"] == "事情终于瓜熟蒂落，不要再"


def test_cmrc2019_expands_sentence_blanks_as_causal_prefixes() -> None:
    rows = utils.process_cmrc2019(
        [
            {
                "data": {
                    "context_id": "DEV_1",
                    "context": "甲。[BLANK1]乙。[BLANK2]丙。",
                    "choices": ["第一句。", "第二句。", "干扰项。"],
                    "answers": [0, 1],
                }
            }
        ]
    )

    assert [row["target"] for row in rows] == [0, 1]
    assert rows[0]["prefix"] == "甲。"
    assert rows[1]["prefix"] == "甲。第一句。乙。"


def test_cloze_task_accessors_preserve_numeric_looking_prompts_as_strings() -> None:
    document = {"prefix": "20", "choices": [1, "二"], "target": "1"}

    assert utils.cloze_prompt(document) == "20"
    assert isinstance(utils.cloze_prompt(document), str)
    assert utils.cloze_choices(document) == ["1", "二"]
    assert utils.cloze_target(document) == 1


def test_answer_text_tasks_do_not_expose_answer_letters() -> None:
    document = {
        "Question": "太阳从哪里升起？",
        "A": "东方",
        "B": "西方",
        "C": "南方",
        "D": "北方",
        "Answer": "A",
    }

    assert utils.four_choice_prompt(document) == "问题：太阳从哪里升起？\n答案："
    assert utils.four_choice_texts(document) == ["东方", "西方", "南方", "北方"]
    assert utils.four_choice_target(document) == 0
    assert "A" not in utils.four_choice_prompt(document)


def test_cmrc2018_metrics_normalize_punctuation_and_use_character_f1() -> None:
    document = {"answers": {"text": ["光荣和ω-force"], "answer_start": [0]}}

    exact = utils.process_cmrc2018_results(document, ["光荣和ω-force。\n解释"])
    partial = utils.process_cmrc2018_results(document, ["光荣和force"])

    assert exact == {"exact_match": 1.0, "char_f1": 1.0}
    assert partial["exact_match"] == 0.0
    assert 0.0 < partial["char_f1"] < 1.0


def test_cmrc2018_empty_generation_scores_zero_instead_of_crashing() -> None:
    document = {"answers": {"text": ["北京"], "answer_start": [0]}}

    assert utils.process_cmrc2018_results(document, [""]) == {
        "exact_match": 0.0,
        "char_f1": 0.0,
    }
