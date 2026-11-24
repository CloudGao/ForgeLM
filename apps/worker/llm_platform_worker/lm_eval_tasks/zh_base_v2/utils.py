from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

_CHID_BLANK = re.compile(r"#idiom\d+#")
_CMRC2019_BLANK = re.compile(r"\[BLANK\d+\]")


def _answer_values(raw: Any, key: str) -> list[Any]:
    if isinstance(raw, dict):
        values = raw.get(key) or []
        return list(values)
    if isinstance(raw, list):
        return [item[key] for item in raw]
    raise ValueError("unsupported answer structure")


def process_chid(dataset: Any) -> Any:
    import datasets

    rows: list[dict[str, Any]] = []
    for document in dataset:
        candidates = [str(value) for value in document["candidates"]]
        answer_texts = [str(value) for value in _answer_values(document["answers"], "text")]
        answer_index = 0
        for passage_index, passage in enumerate(document["content"]):
            rendered_prefix = ""
            cursor = 0
            for blank_index, match in enumerate(_CHID_BLANK.finditer(str(passage))):
                if answer_index >= len(answer_texts):
                    raise ValueError("CHID answer count is smaller than its blank count")
                answer = answer_texts[answer_index]
                rendered_prefix += str(passage)[cursor : match.start()]
                rows.append(
                    {
                        "id": f"{document['idx']}-{passage_index}-{blank_index}",
                        "prefix": rendered_prefix,
                        "choices": candidates,
                        "target": candidates.index(answer),
                    }
                )
                rendered_prefix += answer
                cursor = match.end()
                answer_index += 1
        if answer_index != len(answer_texts):
            raise ValueError("CHID answer count is larger than its blank count")
    return datasets.Dataset.from_list(rows)


def process_cmrc2019(dataset: Any) -> Any:
    import datasets

    rows: list[dict[str, Any]] = []
    for wrapped in dataset:
        document = wrapped["data"]
        context = str(document["context"])
        choices = [str(value) for value in document["choices"]]
        answers = [int(value) for value in document["answers"]]
        matches = list(_CMRC2019_BLANK.finditer(context))
        if len(matches) != len(answers):
            raise ValueError("CMRC2019 answer count does not match its blank count")
        rendered_prefix = ""
        cursor = 0
        for blank_index, (match, target) in enumerate(zip(matches, answers, strict=True)):
            rendered_prefix += context[cursor : match.start()]
            rows.append(
                {
                    "id": f"{document['context_id']}-{blank_index}",
                    "prefix": rendered_prefix,
                    "choices": choices,
                    "target": target,
                }
            )
            rendered_prefix += choices[target]
            cursor = match.end()
    return datasets.Dataset.from_list(rows)


def cloze_prompt(document: dict[str, Any]) -> str:
    # lm-eval uses a native Jinja environment, so a template such as
    # ``{{prefix}}`` can turn a numeric-looking string (for example "20")
    # back into an int. The task contract requires an actual string.
    return str(document["prefix"])


def cloze_choices(document: dict[str, Any]) -> list[str]:
    return [str(value) for value in document["choices"]]


def cloze_target(document: dict[str, Any]) -> int:
    return int(document["target"])


def c3_prompt(document: dict[str, Any]) -> str:
    context = "\n".join(str(value).strip() for value in document["context"])
    return f"{context}\n问题：{str(document['question']).strip()}\n答案："


def c3_choices(document: dict[str, Any]) -> list[str]:
    return [str(value) for value in document["choice"]]


def c3_target(document: dict[str, Any]) -> int:
    return c3_choices(document).index(str(document["answer"]))


def four_choice_prompt(document: dict[str, Any]) -> str:
    question = document.get("Question", document.get("question"))
    if question is None:
        raise ValueError("four-choice document is missing its question")
    return f"问题：{str(question).strip()}\n答案："


def four_choice_texts(document: dict[str, Any]) -> list[str]:
    return [str(document[key]) for key in ("A", "B", "C", "D")]


def four_choice_target(document: dict[str, Any]) -> int:
    answer = document.get("Answer", document.get("answer"))
    return ("A", "B", "C", "D").index(str(answer).strip())


def cmrc2018_prompt(document: dict[str, Any]) -> str:
    return (
        f"材料：{str(document['context']).strip()}\n"
        f"问题：{str(document['question']).strip()}\n答案："
    )


def first_cmrc2018_answer(document: dict[str, Any]) -> str:
    answers = _answer_values(document["answers"], "text")
    return str(answers[0]) if answers else ""


def _normalize_chinese_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z", "C"))
    )


def _character_f1(prediction: str, reference: str) -> float:
    predicted = list(_normalize_chinese_answer(prediction))
    expected = list(_normalize_chinese_answer(reference))
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def process_cmrc2018_results(
    document: dict[str, Any], results: list[str]
) -> dict[str, float]:
    rendered = str(results[0]) if results else ""
    lines = rendered.splitlines()
    prediction = lines[0].strip() if lines else ""
    references = [str(value) for value in _answer_values(document["answers"], "text")]
    if not references:
        return {"exact_match": 0.0, "char_f1": 0.0}
    normalized_prediction = _normalize_chinese_answer(prediction)
    exact_match = max(
        float(normalized_prediction == _normalize_chinese_answer(reference))
        for reference in references
    )
    char_f1 = max(_character_f1(prediction, reference) for reference in references)
    return {"exact_match": exact_match, "char_f1": char_f1}
