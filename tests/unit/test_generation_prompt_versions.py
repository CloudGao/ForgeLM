import hashlib
import json

import pytest
from llm_platform_core.data_generation import (
    DataGenerationCreate,
    factual_surface_anchors,
    parse_fact_cards,
    prompt_for,
    validate_factual_surface,
)
from llm_platform_worker.data_generation_runtime import generation_checkpoint_identity, jobs


def test_v7_two_genres_use_fact_anchors_and_preserve_old_protocol_bounds(tmp_path):
    text=json.dumps([{"主体":"某人备考","关系":"可能意图","客体":"取得好成绩"}],ensure_ascii=False)
    path=tmp_path/'facts.jsonl'
    path.write_text(json.dumps({'text':text},ensure_ascii=False)+'\n',encoding='utf-8')
    config=DataGenerationCreate(name='dual',input_dataset_version_id='input',model_weight_version_id='model',
        prompt_version='cpt-generation-v7',document_mode='whole',styles=['textbook','facts'])
    items=list(jobs({'runtime_path':str(path),'logical_files':[path.name]},config,'knowledge'))
    assert [r['style'] for r in items]==['textbook','facts']
    assert all(r['quality_anchors']['chinese_only'] for r in items)
    assert items[0]['prompt']!=items[1]['prompt']
    assert all('取得好成绩' in r['prompt'] for r in items)
    with pytest.raises(ValueError):
        DataGenerationCreate(**{**config.model_dump(),'document_mode':'chunk'})
    with pytest.raises(ValueError):
        DataGenerationCreate(**{**config.model_dump(),'styles':['retrospective']})
    with pytest.raises(ValueError):
        DataGenerationCreate(**{**config.model_dump(),'prompt_version':'cpt-generation-v3','styles':['facts']})


def test_v8_keeps_frozen_source_and_explicit_abstention():
    text=json.dumps([{"主体":"苏丹论坛报","关系":"所属类型","客体":"网站"}],ensure_ascii=False)
    cfg=DataGenerationCreate(name='candidate',input_dataset_version_id='input',model_weight_version_id='model',
                            prompt_version='cpt-generation-v8',document_mode='whole',styles=['facts','textbook'])
    for style in cfg.styles:
        prompt=prompt_for(text,'knowledge',style,cfg.prompt_version)
        assert '【无法可靠生成】' in prompt and '不等于隶属某机构' in prompt
        assert json.loads(prompt.split('<facts>\n')[1].split('\n</facts>')[0])==json.loads(text)
    assert '【无法可靠生成】' not in prompt_for(text,'knowledge','facts','cpt-generation-v7')


@pytest.mark.parametrize(
    "version,digest",
    [
        ("cpt-generation-v1", "305d9e6395aca4b6039e4c7d38ddf873295587fa504d2d8d768acb768d91ec72"),
        ("cpt-generation-v2", "82bafbb5baca166b9d6600be0524dc8de8362c662f7485a3851753e43c8e439b"),
        ("cpt-generation-v3", "b9ad58cf0fa7a8ca470e577cbe52718604aa04cb7b11d0ceb6bad72d3fef0f7b"),
    ],
)
def test_frozen_legacy_prompts_are_unchanged(version, digest):
    prompt = prompt_for("张三于1979年建立了甲村。", "knowledge", "textbook", version)
    assert hashlib.sha256(prompt.encode()).hexdigest() == digest


def test_v4_flows_through_whole_document_jobs_and_changes_resume_identity(tmp_path):
    text = "某村始建于1901年。\n2001年有39人。"
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    config = DataGenerationCreate(
        name="rewrite", input_dataset_version_id="input", model_weight_version_id="model",
        prompt_version="cpt-generation-v4", document_mode="whole", styles=["textbook", "plain"],
    )
    rows = list(jobs({"runtime_path": str(path), "logical_files": [path.name]}, config, "knowledge"))
    assert len(rows) == 2
    assert all(row["prompt"].endswith(f"<source>\n{text}\n</source>") for row in rows)
    assert rows[0]["prompt"] != rows[1]["prompt"]
    assert all(row["source_sha256"] == hashlib.sha256(text.encode()).hexdigest() for row in rows)
    task = {"id": "task", "payload": {
        "output_dataset_version_id": "output", "generation": config.model_dump(),
        "generation_input": {"artifact_hash": "input"},
        "generation_model": {"identity_hash": "model"},
    }}
    identity = generation_checkpoint_identity(task)
    task["payload"]["generation"]["prompt_version"] = "cpt-generation-v3"
    assert generation_checkpoint_identity(task) != identity


def test_v4_does_not_change_translation_or_default_protocol():
    assert prompt_for("source", "translation", "translation", "cpt-generation-v4") == prompt_for(
        "source", "translation", "translation", "cpt-generation-v1"
    )
    assert DataGenerationCreate.model_fields["prompt_version"].default == "cpt-generation-v1"


@pytest.mark.parametrize("version", ["cpt-generation-v5", "cpt-generation-v6"])
def test_v5_preserves_whole_source_and_isolates_style_checkpoints(tmp_path, version):
    cards = ["甲在2001年策划考察但未成功。", "乙执导关于丙创作的纪录片。", "甲出生于丁地。"]
    text = json.dumps(cards, ensure_ascii=False)
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    config = DataGenerationCreate(
        name="v5", input_dataset_version_id="input", model_weight_version_id="model",
        prompt_version=version, styles=["retrospective", "thematic"],
        document_mode="whole",
    )
    rows = list(jobs({"runtime_path": str(path), "logical_files": [path.name]}, config, "knowledge"))
    assert [row["style"] for row in rows] == ["retrospective", "thematic"]
    assert all(all(card in row["prompt"] for card in cards) for row in rows)
    assert rows[0]["quality_anchors"]["numbers"] == ["2001"]
    assert rows[0]["prompt"] != rows[1]["prompt"]
    assert all(row["source_sha256"] == hashlib.sha256(text.encode()).hexdigest() for row in rows)
    task = {"id": "task", "payload": {
        "output_dataset_version_id": "output", "generation": config.model_dump(),
        "generation_input": {"artifact_hash": "source"},
        "generation_model": {"identity_hash": "model"},
    }}
    identity = generation_checkpoint_identity(task)
    task["payload"]["generation"]["styles"] = ["thematic"]
    assert generation_checkpoint_identity(task) != identity


@pytest.mark.parametrize("version,styles", [
    ("cpt-generation-v4", ["thematic"]),
    ("cpt-generation-v1", ["retrospective"]),
    ("cpt-generation-v5", ["plain"]),
    ("cpt-generation-v5", ["retrospective", "textbook"]),
])
def test_new_styles_reject_incompatible_protocol(version, styles):
    with pytest.raises(ValueError):
        DataGenerationCreate(name="test", input_dataset_version_id="data",
                             model_weight_version_id="model", prompt_version=version, styles=styles)


def test_surface_checks_reject_missing_or_invented_values_and_bad_cards():
    source = "甲村于2001年有81,629人，外文名Alpha Town。作者写过《乙》。"
    anchors = factual_surface_anchors(json.dumps([source]), "thematic")
    cards = json.dumps([source], ensure_ascii=False)
    assert validate_factual_surface(source, anchors) == source
    for bad in [source.replace("2001", "2002"), source.replace("Alpha Town", ""),
                source.replace("《乙》", "《丙》"), source + "2023年建立。"]:
        with pytest.raises(ValueError, match="validation"):
            validate_factual_surface(bad, anchors)
    for bad in ['{}', '[1]', '[]', '[""]']:
        with pytest.raises(ValueError):
            parse_fact_cards(bad)


def test_structured_facts_only_and_whole_processing():
    common = dict(name="facts", input_dataset_version_id="i", model_weight_version_id="m",
                  prompt_version="cpt-generation-v5", styles=["thematic"])
    assert DataGenerationCreate(**common, document_mode="whole")
    with pytest.raises(ValueError):
        DataGenerationCreate(**common)
    cards = [{"主体": "甲", "关系": "出生地", "客体": "乙", "限定": "1901年"}]
    encoded = json.dumps(cards, ensure_ascii=False)
    assert parse_fact_cards(encoded) == cards
    assert "1901" in factual_surface_anchors(encoded, "thematic")["numbers"]
    assert encoded in prompt_for(encoded, "knowledge", "thematic", "cpt-generation-v5")
    for invalid in ["原文段落。", '[{"主体":"甲","客体":"乙"}]', '[{"主体":"甲","关系":"出生","客体":"乙","限定":[]}]']:
        with pytest.raises(ValueError):
            parse_fact_cards(invalid)


def test_english_facts_allow_chinese_translation_but_reject_english_prose():
    facts = json.dumps([{"主体": "bicycle", "关系": "UsedFor", "客体": "transportation"}])
    anchors = factual_surface_anchors(facts, "thematic")
    assert validate_factual_surface("自行车可以用作交通工具。", anchors)
    with pytest.raises(ValueError, match="non_chinese_prose"):
        validate_factual_surface("A bicycle is used for transportation.", anchors)
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("自行车用于 transportation。", anchors)


@pytest.mark.parametrize("relation,good,bad", [
    ("子女包括", "甲某的子女包括乙某。", "甲某是乙某的子女。"),
    ("子女包括", "乙某是甲某的孩子。", "乙某是甲某的母亲。"),
    ("父亲为", "乙某是甲某的父亲。", "甲某是乙某的父亲。"),
    ("母亲为", "甲某的母亲为乙某。", "乙某是甲某的孩子。"),
    ("所属类型", "甲某属于乙某这一类型。", "甲某是乙某的下属机构。"),
])
def test_obvious_relation_errors_are_rejected_without_rejecting_correct_direction(relation, good, bad):
    anchors = factual_surface_anchors(json.dumps([{"主体": "甲某", "关系": relation, "客体": "乙某"}]), "facts")
    assert validate_factual_surface(good, anchors) == good
    with pytest.raises(ValueError, match="obvious_relation_contradiction"):
        validate_factual_surface(bad, anchors)


def test_source_notes_cannot_become_birthplaces_but_literal_topics_remain_allowed():
    anchors = factual_surface_anchors(json.dumps([{"主体": "甲某", "关系": "出生地为", "客体": "北京"}]), "textbook")
    with pytest.raises(ValueError, match="source_boilerplate_as_content"):
        validate_factual_surface("甲某出生于知识库来源记录中。", anchors)
    assert validate_factual_surface("根据来源快照，甲某出生于北京。", anchors)
    anchors = factual_surface_anchors(json.dumps([{"主体": "来源快照", "关系": "用途", "客体": "审计"}]), "facts")
    assert validate_factual_surface("来源快照可用于审计。", anchors)


def test_generic_word_links_are_not_definitions_and_names_cannot_duplicate():
    anchors=factual_surface_anchors(json.dumps([{"主体":"甲词","关系":"有关联（未说明具体关系）","客体":"乙词"}]),"facts")
    assert validate_factual_surface("甲词与乙词有关联。",anchors)
    with pytest.raises(ValueError,match='association_promoted_to_definition'):
        validate_factual_surface("甲词指代乙词。",anchors)
    with pytest.raises(ValueError,match='duplicated_entity_name'):
        validate_factual_surface("亨利·罗斯科·亨利·罗斯科是作家。",{})


def test_single_temporal_direction_keeps_required_word_and_mixed_directions_are_unrestricted():
    facts=[{"主体":"甲","关系":"在其之前","客体":"乙"}]
    anchors=factual_surface_anchors(json.dumps(facts),"facts")
    assert validate_factual_surface("甲在乙之前。",anchors)
    with pytest.raises(ValueError,match='temporal_direction_word_mismatch'):
        validate_factual_surface("甲在乙之后。",anchors)
    facts.append({"主体":"甲","关系":"在其之后","客体":"丙"})
    assert factual_surface_anchors(json.dumps(facts),"facts")['temporal_direction'] is None
