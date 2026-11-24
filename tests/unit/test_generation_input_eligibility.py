import asyncio
import io
import json

import pytest
from llm_platform_core.data_generation import DataGenerationCreate, factual_surface_anchors, validate_factual_surface
from llm_platform_worker.data_generation_runtime import (
    eligible_generation_jobs,
    generate_records,
    generate_records_vllm_async,
    generation_job_key,
    jobs,
    prepare_generation_checkpoint,
)


def fact(subject="darest", relation="词语由其派生", obj="dare"):
    return {"主体": subject, "关系": relation, "客体": obj}


def config(**kwargs):
    return DataGenerationCreate(name="eligibility", input_dataset_version_id="input",
                                model_weight_version_id="model", prompt_version="cpt-generation-v7",
                                document_mode="whole", styles=["facts", "textbook"], **kwargs)


def source(tmp_path, cards, cfg):
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps({"text": json.dumps(card, ensure_ascii=False)}, ensure_ascii=False)
                            + "\n" for card in cards), encoding="utf-8")
    return jobs({"runtime_path": str(path)}, cfg, "knowledge")


@pytest.mark.parametrize("relation", ["FormOf", "/r/FormOf", "DerivedFrom", "EtymologicallyRelatedTo",
                                      "EtymologicallyDerivedFrom", "SpellingOf", "是其词形变化",
                                      "词语由其派生", "词源相关", "词源来自", "拼写变体", "是其拼写变体"])
def test_explicit_latin_lexical_relations_are_detected(relation):
    anchors = factual_surface_anchors(json.dumps([fact("a", relation, "an")]), "facts")
    assert anchors["requires_latin_spelling"] is True
    assert anchors["latin_spelling_facts"] == [
        {"fact_index": 0, "relation": relation, "subject": "a", "object": "an"}]


@pytest.mark.parametrize("card", [
    fact("cup", "可用于", "drinking"), fact("Microsoft", "总部位于", "Redmond"),
    fact("big", "同义或对应译词", "large"), fact("hot", "反义词", "cold"),
    fact("中文词", "词源来自", "另一个中文词"), "English FormOf is merely quoted text",
])
def test_ordinary_english_facts_and_chinese_lexical_relations_are_preserved(card):
    assert factual_surface_anchors(json.dumps([card]), "facts")["requires_latin_spelling"] is False


@pytest.mark.parametrize("relation", ["Synonym", "/r/Synonym", "TranslationOf", "同义或对应译词",
                                      "同义词或对应译词", "对应译词", "翻译为"])
def test_foreign_translation_cards_are_skipped_before_both_style_requests(tmp_path, relation):
    cfg = config(style_order="record_major")
    cards = [[fact("障碍", relation, "obstacle"), fact("杯子", "可用于", "饮水")],
             [fact("big", relation, "large")], [fact("杯子", "可用于", "饮水")]]
    report, completed, skipped = {}, set(), io.StringIO()
    remaining = list(eligible_generation_jobs(source(tmp_path, cards, cfg), {}, report, skipped, completed))
    rejected = [json.loads(line) for line in skipped.getvalue().splitlines()]
    assert [(job["source_ordinal"], job["style"]) for job in remaining] == [(2, "facts"), (2, "textbook")]
    assert len(rejected) == report["preflight_skipped_records"] == 4
    assert all(row["reason"] == "input_latin_translation" and row["model_requests"] == 0 for row in rejected)
    assert rejected[0]["matched_facts"][0]["relation"] == relation
    assert list(eligible_generation_jobs(source(tmp_path, cards[:2], cfg), {}, report, skipped, completed)) == []
    assert report["preflight_skipped_records"] == 4


@pytest.mark.parametrize("card", [fact("Microsoft", "总部位于", "Redmond"),
                                   fact("大", "同义或对应译词", "巨大"),
                                   fact("hot", "反义词", "cold"),
                                   fact("翻译", "有关联（未说明具体关系）", "language"),
                                   "This quoted sentence discusses translation."])
def test_translation_filter_does_not_match_unrelated_facts(card):
    assert not factual_surface_anchors(json.dumps([card]), "facts")["latin_translation_facts"]


@pytest.mark.parametrize("order", ["style_major", "record_major"])
def test_whole_mixed_card_is_skipped_for_both_styles_with_durable_evidence(tmp_path, order):
    cfg = config(style_order=order)
    cards = [[fact()], [fact(), fact("cup", "可用于", "drinking")],
             [fact("cup", "可用于", "drinking")]]
    report, completed = {}, set()
    with (tmp_path / "skips").open("w+", encoding="utf-8") as skipped:
        accepted = list(eligible_generation_jobs(source(tmp_path, cards, cfg), {"input": "frozen"},
                                                report, skipped, completed))
        skipped.seek(0)
        rows = [json.loads(line) for line in skipped]
    assert [(x["source_ordinal"], x["style"]) for x in accepted] == [(2, "facts"), (2, "textbook")]
    assert len(rows) == report["preflight_skipped_records"] == report["skipped_generation_records"] == 4
    assert len(completed) == 4
    assert all(row["model_requests"] == 0 and row["reason"] == "input_latin_spelling" for row in rows)
    assert all(row["matched_facts"][0]["subject"] == "darest" for row in rows)
    assert all(row["metadata"]["input"] == "frozen" for row in rows)


@pytest.mark.parametrize("engine", ["vllm", "transformers"])
@pytest.mark.parametrize("card", [fact(), fact("障碍", "同义或对应译词", "obstacle")])
def test_all_incompatible_input_makes_zero_model_or_probe_requests(tmp_path, engine, card):
    class NoRequests:
        def generate(self, *args, **kwargs):
            pytest.fail("incompatible cards reached the model")

    cfg, report = config(engine=engine), {"probes": [], "oom_backoffs": 0}
    with (tmp_path / "skips").open("w", encoding="utf-8") as skipped:
        inputs = source(tmp_path, [[card]], cfg)
        if engine == "vllm":
            # Even model/tokenizer imports are avoided; no GPU or vLLM is installed in this test env.
            asyncio.run(generate_records_vllm_async("not-a-model", inputs, cfg, io.StringIO(), {},
                                                   report, set(), skipped))
        else:
            generate_records(NoRequests(), inputs, cfg, io.StringIO(), {}, report, skipped)
    assert report["output_records"] == 0
    assert report["preflight_skipped_records"] == 2
    assert report["selection"] == "no_eligible_input"


def test_skip_checkpoint_resumes_without_repeating_a_card_or_count(tmp_path):
    cfg = config()
    provenance = {"input_artifact_hash": "hash"}
    task = {"id": "task", "payload": {"generation": cfg.model_dump(), "output_dataset_version_id": "v",
                                        "generation_input": {"artifact_hash": "hash"},
                                        "generation_model": {"identity_hash": "model"}}}
    first, report = tmp_path / "v-first", {}
    keys, _, _, _, skip_path = prepare_generation_checkpoint(task, tmp_path, first, provenance, report)
    (first / "data.partial").touch()
    original = list(source(tmp_path, [[fact()], [fact("cup", "可用于", "drinking")]], cfg))
    with skip_path.open("a", encoding="utf-8") as skipped:
        accepted = list(eligible_generation_jobs(iter(original), provenance, report, skipped, keys))
    report = {}
    keys, _, _, skipped_count, resumed_path = prepare_generation_checkpoint(
        task, tmp_path, tmp_path / "v-second", provenance, report)
    prefix = resumed_path.read_bytes()
    assert skipped_count == 2
    with resumed_path.open("a", encoding="utf-8") as skipped:
        resumed = list(eligible_generation_jobs(iter(original), provenance, report, skipped, keys))
    assert [generation_job_key(x) for x in resumed] == [generation_job_key(x) for x in accepted]
    assert resumed_path.read_bytes() == prefix
    assert report["preflight_skipped_records"] == 0
    assert report["skipped_generation_records"] == 2


def test_translation_jobs_are_not_subject_to_lexical_filter(tmp_path):
    path = tmp_path / "translation.jsonl"
    path.write_text(json.dumps({"text": json.dumps([fact()])}), encoding="utf-8")
    inputs = jobs({"runtime_path": str(path)}, config(), "translation")
    report = {}
    assert len(list(eligible_generation_jobs(inputs, {}, report, None))) == 1
    assert report["preflight_skipped_records"] == 0


def test_opaque_place_codes_are_skipped_without_rejecting_normal_abbreviations(tmp_path):
    cfg=config(fact_input_selection="chinese_ready");report={}
    with (tmp_path/'opaque-skips').open('w+',encoding='utf-8') as skipped:
        accepted=list(eligible_generation_jobs(source(tmp_path, [
            [fact('甲某','逝世地为','UN/LOCODE:USLDB')],
            [fact('USA','名称','美国')]],cfg),{},report,skipped,config=cfg))
        skipped.seek(0);rows=[json.loads(line) for line in skipped]
    assert len(rows)==2 and all(row['reason']=='input_opaque_label' and row['model_requests']==0 for row in rows)
    assert len(accepted)==2 and all(row['source_ordinal']==1 for row in accepted)


def test_latin_taxon_names_and_wiki_editor_paths_are_not_guessed(tmp_path):
    cfg=config(fact_input_selection="chinese_ready");report={}
    cards=[[fact('arius sciurus','taxon rank','种')],
           [fact('WikiProject Georgia/Person','所属类型','人类')],
           [fact('西方鼠鲶','taxon rank','种')],
           [fact('London','country','英国')]]
    with (tmp_path/'unsafe-labels').open('w+',encoding='utf-8') as skipped:
        accepted=list(eligible_generation_jobs(source(tmp_path,cards,cfg),{},report,skipped,config=cfg))
        skipped.seek(0);rows=[json.loads(line) for line in skipped]
    assert {r['reason'] for r in rows}=={'input_untranslated_taxon','input_opaque_label'}
    assert {r['source_ordinal'] for r in accepted}=={2,3}


def test_explicit_abstention_is_journaled_without_retry_or_dataset_text(tmp_path):
    class Engine:
        calls = 0

        def generate(self, prompts):
            self.calls += len(prompts)
            return [("【无法可靠生成】", 5, False) for _ in prompts]

    engine, report, output = Engine(), {"probes": [], "oom_backoffs": 0}, io.StringIO()
    cfg = config(batch_size=1, max_batch_size=1)
    with (tmp_path / "skips").open("w+", encoding="utf-8") as skipped:
        generate_records(engine, source(tmp_path, [[fact("未知词", "有关联", "其他词")]], cfg),
                         cfg, output, {}, report, skipped)
        skipped.seek(0)
        rows = [json.loads(line) for line in skipped]
    assert engine.calls == len(rows) == 2
    assert {row["reason"] for row in rows} == {"unresolved_source"}
    assert not output.getvalue() and report["output_records"] == 0


@pytest.mark.parametrize("offset", ["+05:30", "-03:30", "+00:00"])
def test_source_backed_timezone_normalization_preserves_sign_and_minutes(offset):
    anchors = factual_surface_anchors(json.dumps([fact("甲城", "时区", "UTC" + offset)]), "facts")
    assert validate_factual_surface("甲城使用UTC" + offset + "时区。", anchors) == "甲城使用协调世界时" + offset + "时区。"


@pytest.mark.parametrize("text", ["甲城使用协调世界时-05:30。", "甲城使用协调世界时+05:00。",
                                  "甲城时区为05、30。", "甲城使用协调世界时+05:300。"])
def test_timezone_cannot_be_replaced_by_matching_unsigned_numbers(text):
    anchors = factual_surface_anchors(json.dumps([fact("甲城", "时区", "UTC+05:30")]), "facts")
    with pytest.raises(ValueError, match="timezone_mismatch"):
        validate_factual_surface(text, anchors)


def test_timezone_does_not_exempt_unrelated_english_or_plain_text_anchors():
    anchors = factual_surface_anchors(json.dumps([fact("甲城", "时区", "UTC+05:30")]), "facts")
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface("甲城使用UTC+05:30时区，city。", anchors)
    anchors = factual_surface_anchors(json.dumps(["甲城使用UTC+05:30时区。"]), "facts")
    assert validate_factual_surface("甲城使用UTC+05:30时区。", anchors) == "甲城使用UTC+05:30时区。"


@pytest.mark.parametrize("number,word", [("5", "第五代"), ("10", "第十届"), ("25", "第二十五届"),
                                         ("005", "第五号"), ("3", "三世"), ("2001", "二〇〇一年")])
def test_clear_chinese_numerals_are_equivalent(number, word):
    anchors = factual_surface_anchors(json.dumps([fact("甲", "编号", number)]), "facts")
    assert validate_factual_surface("甲为" + word + "。", anchors) == "甲为" + word + "。"


@pytest.mark.parametrize("text", ["甲为第四代。", "甲为第五代，也叫第六代。", "甲没有编号。"])
def test_changed_added_and_missing_numbers_still_fail(text):
    anchors = factual_surface_anchors(json.dumps([fact("甲", "编号", "5")]), "facts")
    with pytest.raises(ValueError, match="number_mismatch"):
        validate_factual_surface(text, anchors)


def test_non_numeric_chinese_words_and_dotted_identifiers_do_not_break_validation():
    anchors = factual_surface_anchors(json.dumps([fact("四川的一名将领", "出生年", "1876")]), "facts")
    text = "四川的一名将领出生于1876年，是一代名将。"
    assert validate_factual_surface(text, anchors) == text
    anchors = factual_surface_anchors(json.dumps([fact("软件", "版本", "1.2.3")]), "facts")
    assert validate_factual_surface("软件版本为1.2.3。", anchors) == "软件版本为1.2.3。"
    with pytest.raises(ValueError, match="number_mismatch"):
        validate_factual_surface("软件版本为1.2.4。", anchors)


def test_equivalent_timezone_is_written_without_quality_retry(tmp_path):
    class Engine:
        calls = 0

        def generate(self, prompts):
            self.calls += len(prompts)
            return [("甲城使用UTC+05:30时区。", 15, False) for _ in prompts]

    engine, report, output = Engine(), {"probes": [], "oom_backoffs": 0}, io.StringIO()
    cfg = config(batch_size=1, max_batch_size=1)
    with (tmp_path / "skips").open("w+", encoding="utf-8") as skipped:
        generate_records(engine, source(tmp_path, [[fact("甲城", "时区", "UTC+05:30")]], cfg),
                         cfg, output, {}, report, skipped)
        skipped.seek(0)
        assert not skipped.read()
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert engine.calls == len(rows) == 2
    assert all(row["text"] == "甲城使用协调世界时+05:30时区。" for row in rows)


@pytest.mark.parametrize("file_format", ["jsonl", "parquet"])
@pytest.mark.parametrize("selection", ["standard", "chinese_ready"])
def test_preflight_selection_preserves_source_scope_and_resume(tmp_path, file_format, selection):
    cfg = config(fact_input_selection=selection, style_order="record_major")
    rows = [
        {"source": "wikidata5m", "text": json.dumps([fact("John Doe", "国籍为", "英国")])},
        {"source": "wikidata5m", "text": json.dumps([fact("甲城", "时区", "UTC+05:30")])},
        {"source": "conceptnet", "text": json.dumps([fact("cup", "可用于", "drinking")])},
        {"source": "atomic2020", "text": json.dumps([fact("某人 cooks", "可能意图", "eat")])},
        {"source": "conceptnet", "text": json.dumps([fact("foo", "有关联（未说明具体关系）", "bar")])},
        {"text": json.dumps([fact("John Doe", "国籍为", "英国")])},
    ]
    path = tmp_path / ("cards." + file_format)
    if file_format == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def inputs():
        return jobs({"runtime_path": str(path)}, cfg, "knowledge")

    report, completed, skipped = {}, set(), io.StringIO()
    accepted = list(eligible_generation_jobs(inputs(), {}, report, skipped, completed, config=cfg))
    rejected = [json.loads(line) for line in skipped.getvalue().splitlines()]
    if selection == "chinese_ready":
        assert {job["source_ordinal"] for job in accepted} == {1, 2, 3, 5}
        assert len(rejected) == 4
        assert {row["reason"] for row in rejected} == {"input_untranslated_entity", "input_foreign_association_only"}
        assert all(row["model_requests"] == 0 for row in rejected)
        assert rejected[0]["metadata"]["source_kind"] == "wikidata5m"
        assert "quality_anchors" not in rejected[0]["metadata"]
        assert rejected[0]["matched_fact_count"] == 1
        list(eligible_generation_jobs(inputs(), {}, report, skipped, completed, config=cfg))
        assert report["preflight_skipped_records"] == 4
    else:
        assert len(accepted) == 12 and not rejected
    assert accepted[0]["source_sha256"] == next(job for job in inputs() if job["source_ordinal"] == accepted[0]["source_ordinal"])["source_sha256"]


def test_preflight_selection_and_retry_budget_are_frozen_identity():
    from llm_platform_worker.data_generation_runtime import generation_checkpoint_identity

    task = {"id": "task", "payload": {"generation": config(engine="vllm").model_dump(),
                                        "output_dataset_version_id": "out",
                                        "generation_input": {"artifact_hash": "input"},
                                        "generation_model": {"identity_hash": "model"}}}
    old = generation_checkpoint_identity(task)
    task["payload"]["generation"].update(fact_input_selection="chinese_ready", validation_retry_limit=0)
    assert generation_checkpoint_identity(task) != old


@pytest.mark.parametrize("engine", ["transformers", "vllm"])
def test_selected_foreign_entity_never_loads_or_requests_model(tmp_path, engine):
    class NoRequests:
        def generate(self, *args, **kwargs):
            pytest.fail("preselected card reached model")

    cfg = config(engine=engine, fact_input_selection="chinese_ready")
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"source": "wikidata5m", "text": json.dumps([fact("John Doe", "国籍为", "英国")])}), encoding="utf-8")
    report = {"probes": [], "oom_backoffs": 0}
    with (tmp_path / "skips").open("w", encoding="utf-8") as skipped:
        inputs = jobs({"runtime_path": str(path)}, cfg, "knowledge")
        if engine == "vllm":
            asyncio.run(generate_records_vllm_async("not-a-model", inputs, cfg, io.StringIO(), {}, report, set(), skipped))
        else:
            generate_records(NoRequests(), inputs, cfg, io.StringIO(), {}, report, skipped)
    assert report["selection"] == "no_eligible_input"
    assert report["preflight_skip_reasons"] == {"input_untranslated_entity": 2}


@pytest.mark.parametrize("relation", ["反义词", "Antonym", "/r/Antonym", "类似", "SimilarTo", "不同于", "DistinctFrom"])
def test_foreign_word_comparisons_are_preselected_but_real_fact_mixtures_remain(tmp_path, relation):
    cfg = config(fact_input_selection="chinese_ready")
    cards = [[fact("hot", relation, "cold"), fact("hot", "有关联（未说明具体关系）", "warm")],
             [fact("cup", "可用于", "drinking"), fact("hot", relation, "cold")]]
    report, skipped = {}, io.StringIO()
    accepted = list(eligible_generation_jobs(source(tmp_path, cards, cfg), {}, report, skipped, config=cfg))
    assert {job["source_ordinal"] for job in accepted} == {1}
    rejected = [json.loads(line) for line in skipped.getvalue().splitlines()]
    assert len(rejected) == 2
    assert all(row["reason"] == "input_foreign_lexical_only" and row["model_requests"] == 0 for row in rejected)


@pytest.mark.parametrize("text", ["某人ｒｉｐｓ开放包装。", "某人使用ＳＴＡＲ产品。"])
def test_fullwidth_foreign_words_cannot_bypass_cheap_fallback(text):
    anchors = factual_surface_anchors(json.dumps([fact("某人", "行为", "打开包装")]), "facts")
    with pytest.raises(ValueError, match="untranslated_words"):
        validate_factual_surface(text, anchors)


@pytest.mark.parametrize("explicit_config", [True, False])
def test_standard_retains_foreign_facts_including_taxa_and_identifiers(tmp_path, explicit_config):
    cfg = config()
    cards = [[fact("John Doe", "国籍为", "英国")],
             [fact("arius sciurus", "taxon rank", "种")],
             [fact("甲某", "逝世地为", "UN/LOCODE:USLDB")],
             [fact("WikiProject Georgia/Person", "所属类型", "人类")],
             [fact("hot", "反义词", "cold")],
             [fact("cup", "有关联（未说明具体关系）", "drink")],
             [fact("障碍", "翻译为", "obstacle")], [fact()]]
    report, skipped = {}, io.StringIO()
    accepted = list(eligible_generation_jobs(source(tmp_path, cards, cfg), {}, report, skipped,
                                             config=cfg if explicit_config else None))
    assert {job["source_ordinal"] for job in accepted} == set(range(6))
    assert len(accepted) == 12
    rejected = [json.loads(line) for line in skipped.getvalue().splitlines()]
    assert len(rejected) == 4
    assert {row["reason"] for row in rejected} == {"input_latin_translation", "input_latin_spelling"}
    assert all(row["model_requests"] == 0 for row in rejected)
