import json

import pytest
from llm_platform_core.data_generation import (
    DataGenerationCreate,
    factual_surface_anchors,
    validate_factual_surface,
)
from llm_platform_worker.data_generation_runtime import generation_checkpoint_identity, jobs


def fact(subject, relation, obj):
    return {"主体": subject, "关系": relation, "客体": obj}


def anchors(*facts, source="wikidata5m", policy="source_exact"):
    return factual_surface_anchors(json.dumps(list(facts), ensure_ascii=False), "facts", source,
                                   proper_name_policy=policy)


def config(**kwargs):
    return DataGenerationCreate(name="names", input_dataset_version_id="input", model_weight_version_id="original",
                                prompt_version="cpt-generation-v8", styles=["facts", "textbook"],
                                document_mode="whole", **kwargs)


@pytest.mark.parametrize("facts,text", [
    ([fact("John Doe", "国籍为", "英国")], "John Doe 的国籍是英国。"),
    ([fact("Fulbeck", "所在行政区为", "South Kesteven")], "Fulbeck 位于 South Kesteven。"),
    ([fact("Hositea cyanops", "parent taxon", "hositea")], "Hositea cyanops 的上级分类单元为 hositea。"),
    ([fact("The Future of Assisted Suicide and Euthanasia", "所属类型", "书籍")],
     "The Future of Assisted Suicide and Euthanasia 是一本书籍。"),
    ([fact("Chris Rose (journalist)", "职业为", "新闻工作者")], "Chris Rose 是新闻工作者。"),
    ([fact("BBC", "headquarters location", "London")], "BBC 总部位于 London。"),
])
def test_source_names_in_chinese_prose_are_preserved_without_rewriting(facts, text):
    assert validate_factual_surface(text, anchors(*facts)) == text


@pytest.mark.parametrize("text", [
    "John Doe is a British citizen.", "John Doe is British。中文。", "John Doe 是 British citizen。",
    "John Doe 是 a 英国人。", "John Doe 是英国人，hello。", "John Doe 是英国人，ｈｅｌｌｏ。",
    "John Doeman 是英国人。", "John Smith 是英国人。", "John Doe", "John Doe，London。",
])
def test_english_sentences_extra_words_invented_names_and_name_only_outputs_fail(text):
    with pytest.raises(ValueError):
        validate_factual_surface(text, anchors(fact("John Doe", "国籍为", "英国")))


@pytest.mark.parametrize("relation,obj", [("职业为", "journalist"), ("所属类型", "Common names"),
                                          ("genre", "Sports game"), ("main subject", "doctor assisted suicide"),
                                          ("languages spoken, written or signed", "English")])
def test_same_source_common_concept_objects_are_not_name_exemptions(relation, obj):
    a = anchors(fact("John Doe", "国籍为", "英国"), fact("John Doe", relation, obj))
    assert obj not in a["source_proper_names"]
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("John Doe 的相关信息是 " + obj + "。", a)


@pytest.mark.parametrize("source", [None, "conceptnet", "atomic2020", "unknown"])
def test_unknown_and_common_fact_sources_do_not_infer_names(source):
    a = anchors(fact("John Doe", "国籍为", "英国"), source=source)
    assert not a["source_proper_names"]
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("John Doe 的国籍是英国。", a)


def test_chinese_names_and_editor_paths_do_not_create_english_exemptions():
    a = anchors(fact("约翰", "出生地为", "WikiProject Georgia/Person"), fact("百度Baidu", "country", "中国"))
    assert not a["source_proper_names"]


def test_default_policy_remains_strict_and_numeric_checks_are_not_masked():
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("John Doe 的国籍是英国。", anchors(fact("John Doe", "国籍为", "英国"), policy="translate"))
    a = anchors(fact("Area 51", "country", "美国"))
    assert validate_factual_surface("Area 51 位于美国。", a) == "Area 51 位于美国。"
    with pytest.raises(ValueError, match="number_mismatch"):
        validate_factual_surface("Area 51 位于美国，面积52。", a)


@pytest.mark.parametrize("order", ["style_major", "record_major"])
def test_policy_reaches_both_style_prompts_anchors_and_frozen_identity(tmp_path, order):
    path = tmp_path / "input.jsonl"
    text = json.dumps([fact("John Doe", "国籍为", "英国")], ensure_ascii=False)
    path.write_text(json.dumps({"source": "wikidata5m", "text": text}), encoding="utf-8")
    cfg = config(proper_name_policy="source_exact", style_order=order)
    items = list(jobs({"runtime_path": str(path)}, cfg, "knowledge"))
    assert len(items) == 2
    for item in items:
        assert item["quality_anchors"]["source_proper_names"] == ["John Doe"]
        assert "普通词、职业、类型、关系、动作和句子必须译成中文" in item["prompt"]
        assert "全中文正文" not in item["prompt"]
    task = {"id": "task", "payload": {"generation": cfg.model_dump(), "output_dataset_version_id": "out",
                                        "generation_input": {"artifact_hash": "input"},
                                        "generation_model": {"identity_hash": "model"}}}
    current = generation_checkpoint_identity(task)
    task["payload"]["generation"]["proper_name_policy"] = "translate"
    assert current != generation_checkpoint_identity(task)
    original = list(jobs({"runtime_path": str(path)}, cfg.model_copy(update={"proper_name_policy": "translate"}), "knowledge"))
    assert original[0]["source_sha256"] == items[0]["source_sha256"]
    assert "全中文正文" in original[0]["prompt"]


@pytest.mark.parametrize("kwargs", [{"chinese_output_constraint": True, "engine": "vllm"},
                                    {"fact_input_selection": "chinese_ready"}])
def test_conflicting_frozen_language_settings_are_rejected(kwargs):
    with pytest.raises(ValueError, match="原文专名保留"):
        config(proper_name_policy="source_exact", **kwargs)


@pytest.mark.parametrize("facts,text", [
    ([fact("Green Valley Lake (Arizona)", "所属类型", "湖泊")], "Green Valley Lake (Arizona) 是湖泊。"),
    ([fact("mainbocher", "founder", "Main Rousseau Bocher")], "mainbocher 的创始人为 Main Rousseau Bocher。"),
    ([fact("jadoon", "所属类型", "姓")], "jadoon 是一个姓。"),
    ([fact("anti-terror units", "military branch", "QSD")], "反恐部队属于QSD。"),
    ([fact("Person", "就读学校为", "St John's College, Oxford")], "Person 就读于圣约翰学院（Oxford）。"),
    ([fact("William J. Rotch", "国籍为", "美国")], "威廉 J. 罗奇是美国人。"),
])
def test_audited_names_roles_annotations_and_initials_do_not_false_reject(facts,text):
    assert validate_factual_surface(text, anchors(*facts)) == text


def test_name_annotation_is_not_a_free_standing_english_exemption():
    a = anchors(fact("Andrew Todd (bowls)", "所属类型", "人类"))
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("Andrew Todd 属于人类，喜欢 bowls。",a)
