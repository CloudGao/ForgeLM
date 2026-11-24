import csv
import json

import pytest
from llm_platform_core.data_generation import DataGenerationCreate
from llm_platform_core.dense_qa import duplicate_kind, planner_prompt, source_jobs, validate_plans, validate_review
from llm_platform_worker.data_generation_runtime import _scan_checkpoint_rows
from llm_platform_worker.dense_qa_runtime import finalize_qa_batches, generate_qa


def config(**kw):
    return DataGenerationCreate(name="pilot", input_dataset_version_id="mother", model_weight_version_id="awq",
        prompt_version="dense-qa-v2", engine="vllm", styles=["textbook"], document_mode="whole",
        text_field="dense_note", batch_size=2, **kw)


def test_sources_preserve_mother_notes_and_do_not_guess_ambiguous_names(tmp_path):
    path = tmp_path / "mother.csv"
    rows = [dict(id="a", concept="甲", atlas_order=1, dense_note="甲的知识正文。", prerequisites="乙 | missing", importance=5),
            dict(id="b", concept="乙", atlas_order=2, dense_note="乙的一种含义。", importance=3),
            dict(id="c", concept="乙", atlas_order=3, dense_note="乙的另一含义。", importance=3)]
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {}
    items = list(source_jobs({"runtime_path": str(path)}, config(qa_primary_limit=1), report))
    assert [n["id"] for n in items[0]["qa_sources"]] == ["a"]
    assert items[0]["qa_sources"][0]["dense_note"] == rows[0]["dense_note"]
    assert len(report["qa_unresolved_links"]) == 2
    rows[0]["dense_note"] = ""
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="mother node"):
        list(source_jobs({"runtime_path": str(path)}, config(), {}))


def test_no_fabricated_source_or_evidence_and_no_duplicate_questions():
    sources = [{"id": "a", "dense_note": "声音传播需要介质，真空不能传声。"}]
    plan = dict(angle="mechanism", difficulty="L2", intent="声音传播", source_ids=["invented"])
    assert validate_plans({"plans": [plan]}, sources) == []
    review = dict(grounded=True, all_sources_needed=True, semantic_duplicate=False,
                  evidence=[dict(node_id="a", quote="真空中的声音传播得更快")])
    with pytest.raises(ValueError, match="verbatim"):
        validate_review(review, sources)
    review["evidence"][0]["quote"] = "声音传播需要介质"
    validate_review(review, sources)
    review["grounded"] = "true"
    with pytest.raises(ValueError, match="rejected"):
        validate_review(review, sources)
    assert duplicate_kind({"question": "为什么真空不传声？"}, [{"question": "为什么真空不传声"}]) == "exact"


def test_single_source_excludes_resolvable_links_and_keeps_original_note(tmp_path):
    path = tmp_path / "mother.csv"
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["id", "concept", "atlas_order", "dense_note", "prerequisites"])
        writer.writeheader()
        writer.writerows([
            dict(id="a", concept="甲", atlas_order=1, dense_note="唯一母文甲。", prerequisites="乙"),
            dict(id="b", concept="乙", atlas_order=2, dense_note="禁止混入的乙正文。", prerequisites=""),
        ])
    linked = list(source_jobs({"runtime_path": str(path)}, config(qa_primary_limit=1), {}))[0]
    report = {}
    single = list(source_jobs({"runtime_path": str(path)}, config(qa_primary_limit=1, qa_source_mode="single"), report))[0]
    assert [n["id"] for n in linked["qa_sources"]] == ["a", "b"]
    assert [n["id"] for n in single["qa_sources"]] == ["a"]
    assert single["source_sha256"] != linked["source_sha256"]
    assert report["qa_source_mode"] == "single"
    prompt = planner_prompt(single["qa_sources"])
    assert "唯一母文甲。" in prompt and "禁止混入的乙正文。" not in prompt
    assert "本次只有一个母节点" in prompt
    assert validate_plans({"plans": [dict(angle="mechanism", difficulty="L2", intent="甲乙有什么联系？", source_ids=["a", "b"])]}, single["qa_sources"]) == []


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.asyncio
async def test_pilot_hard_cap_stages_grounded_and_node_checkpoint(tmp_path, automatic):
    node = dict(id="a", atlas_order=1, concept="声音", importance=5, dense_note="声音传播需要介质，真空不能传声。")
    item = dict(prompt="", qa_sources=[node], source_file="source.csv", source_ordinal=0,
                source_sha256="sha", character_start=0, character_end=20, style="dense_qa")
    calls = []
    async def request(item, **kwargs):
        prompt = item["prompt"]
        assert node["dense_note"] in prompt
        assert "qa_json_schema" in item
        calls.append(prompt)
        if item['qa_json_schema']['properties'].get('plans'):
            value = {"plans": [dict(angle="mechanism", difficulty="L2", intent="声音为何需要介质才能传播？", source_ids=["a"]),
                                 dict(angle="condition", difficulty="L3", intent="真空中能否听到远处的爆炸？", source_ids=["a"])]}
        elif '"grounded"' in prompt:
            value = dict(grounded=True, all_sources_needed=True, semantic_duplicate=False,
                         evidence=[dict(node_id="a", quote="声音传播需要介质")])
        else:
            value = dict(question="真空里能听到远处的爆炸吗？" if "真空中能否" in prompt else "声音是如何传播到耳朵的？",
                         answer="声音传播需要介质，真空不能传声。")
        return item, (json.dumps(value, ensure_ascii=False), 20, False), True, 0
    capacities = []
    async def prepare_capacity(capacity):
        capacities.append(capacity)
    selections = []
    async def selector(cap, trial, report):
        selections.append(cap)
        result = await trial(1)
        assert result["requests"] == 2 and result["raw_tokens"] == 40
        assert result["accepted_records"] == 2
        report.update(submission_window=1, selected_batch_size=1)
        return 1
    run_config = DataGenerationCreate.model_validate({**config(qa_limit=1).model_dump(),
        "batch_size": "auto" if automatic else 2})
    partial = tmp_path / "data.partial"
    provenance = {"input_dataset_version_id": "mother"}
    report = {}
    with partial.open("a", encoding="utf-8") as stream:
        await generate_qa(iter([item]), run_config, request, stream, provenance, report, lambda *a, **k: None, tmp_path, select_window=selector, prepare_capacity=prepare_capacity if automatic else None)
    assert report["qa_count"] == 1 and len(calls) == (9 if automatic else 5)
    assert selections == ([2] if automatic else [])
    assert capacities == ([1, 1] if automatic else [])
    keys, rows, _ = _scan_checkpoint_rows(partial, provenance)
    assert len(keys) == rows == 1
    with partial.open("a", encoding="utf-8") as stream:
        await generate_qa(iter([item]), run_config, request, stream, provenance, report, lambda *a, **k: None, tmp_path, select_window=selector, prepare_capacity=prepare_capacity if automatic else None)
    assert len(calls) == (9 if automatic else 5)  # frozen target already satisfied on resume
    output = tmp_path / "data.jsonl"
    finalize_qa_batches(partial, output, report)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["metadata"]["input_dataset_version_id"] == "mother"
    assert rows[0]["messages"][1]["content"] == rows[0]["answer"]


@pytest.mark.parametrize("change", [{"engine": "transformers"}, {"document_mode": "chunk"}, {"text_field": "text"}])
def test_qa_rejects_unsupported_modes(change):
    with pytest.raises(ValueError):
        DataGenerationCreate.model_validate({**config().model_dump(), **change})


def test_evidence_plans_reject_invented_quotes_and_preserve_distinct_facts():
    from llm_platform_core.dense_qa import validate_evidence_plans, stage_schema
    sources = [dict(id="a", dense_note="声音传播需要介质，真空不能传声。")]
    plan = dict(angle="identity", difficulty="L1", intent="声音传播需要什么条件？", source_ids=["a"],
                evidence=[dict(node_id="a", quote="声音传播需要介质")])
    assert validate_evidence_plans({"plans": [plan]}, sources) == [plan]
    assert validate_evidence_plans({"plans": []}, sources) == []
    assert validate_evidence_plans({"plans": [dict(plan, evidence=[dict(node_id="a", quote="真空中的声音更快")])]}, sources) == []
    assert validate_evidence_plans({"plans": [dict(plan, difficulty="L4")]}, sources) == []
    assert len(validate_evidence_plans({"plans": [plan, dict(plan, intent="真空中能够传播声音吗？")]}, sources)) == 2
    assert stage_schema("planner", sources, True)["properties"]["plans"]["minItems"] == 0


@pytest.mark.asyncio
async def test_v3_empty_node_then_evidence_bound_output(tmp_path):
    node = dict(id="a", atlas_order=1, concept="声音", importance=5, dense_note="声音传播需要介质，真空不能传声。")
    item = dict(prompt="", qa_sources=[node], source_file="source.csv", source_ordinal=0,
                source_sha256="sha", character_start=0, character_end=20, style="dense_qa")
    evidence = [dict(node_id="a", quote="声音传播需要介质")]
    planning_calls = 0
    async def request(job, **kwargs):
        nonlocal planning_calls
        assert node["dense_note"] in job["prompt"]
        schema = job["qa_json_schema"]["properties"]
        if "plans" in schema:
            planning_calls += 1
            value = {"plans": [] if planning_calls == 1 else [dict(evidence=evidence, angle="identity", difficulty="L1", intent="声音传播需要什么条件？", source_ids=["a"])]}
        elif "grounded" in schema:
            value = dict(grounded=True, all_sources_needed=True, semantic_duplicate=False, reason="", evidence=evidence)
        else:
            value = dict(question="声音传播需要什么条件？", answer="声音传播需要介质。")
        return job, (json.dumps(value, ensure_ascii=False), 20, False), True, 0
    cfg = DataGenerationCreate.model_validate({**config(qa_limit=1).model_dump(), "prompt_version": "dense-qa-v3"})
    partial = tmp_path / "partial.jsonl"
    report = {}
    with partial.open("a", encoding="utf-8") as stream:
        await generate_qa(iter([item, {**item, "source_ordinal": 1}]), cfg, request, stream,
                          {"input_dataset_version_id": "mother"}, report, lambda *a, **k: None, tmp_path)
    batches = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    assert batches[0]["qa_records"] == []
    assert batches[1]["qa_records"][0]["planning_evidence"] == evidence
    assert report["qa_target_reached"] is True


@pytest.mark.asyncio
async def test_node_target_rounds_deduplicate_and_checkpoint_once(tmp_path):
    source = [dict(prompt="", qa_sources=[dict(id=k, atlas_order=i, concept=k, importance=3,
              dense_note="声音传播需要介质，真空不能传声。")], source_file="source.csv", source_ordinal=i,
              source_sha256=k, character_start=0, character_end=20, style="dense_qa") for i,k in enumerate(["a", "b"])]
    calls = {"a": 0, "b": 0}
    async def request(job, **kwargs):
        key = job["qa_sources"][0]["id"]
        evidence = [dict(node_id=key, quote="声音传播需要介质")]
        schema = job["qa_json_schema"]["properties"]
        if "plans" in schema:
            calls[key] += 1
            if calls[key] > 1:
                assert "已有问题" in job["prompt"]
            q = "声音传播需要什么物质作为介质？" if key == "a" and calls[key] == 1 else "真空环境里可以听见远方爆炸吗？" if key == "a" else "这些现象体现了哪一项物理条件？"
            value = {"plans": [dict(evidence=evidence, angle="identity", difficulty="L1", intent=q, source_ids=[key])]}
        elif "grounded" in schema:
            value = dict(grounded=True, all_sources_needed=True, semantic_duplicate=False, reason="", evidence=evidence)
        else:
            plan = json.loads(job["prompt"].split("证据与问题规划：", 1)[1])
            value = dict(question=plan["intent"], answer="声音传播需要介质。")
        return job, (json.dumps(value, ensure_ascii=False), 20, False), True, 0
    cfg = DataGenerationCreate.model_validate({**config(qa_limit=4).model_dump(), "prompt_version": "dense-qa-v3", "qa_per_node_target": 2, "qa_node_max_rounds": 3})
    partial = tmp_path / "partial.jsonl"
    report = {}
    with partial.open("a", encoding="utf-8") as stream:
        await generate_qa(iter(source), cfg, request, stream, {"input_dataset_version_id": "mother"}, report, lambda *a, **k: None, tmp_path)
    batches = [json.loads(l) for l in partial.read_text(encoding="utf-8").splitlines()]
    assert len(batches) == 2
    assert [len(b["qa_records"]) for b in batches] == [2, 1]
    assert calls == {"a": 2, "b": 3}
    assert report["qa_target_reached"] is False
    assert report["qa_duplicate_removed"] == 2
    assert report["qa_node_shortfalls"] == [dict(node_id="b", concept="b", target=2, accepted=1, rounds=3)]
    assert [r["node_generation_round"] for r in batches[0]["qa_records"]] == [1, 2]
    keys, count, _ = _scan_checkpoint_rows(partial, {"input_dataset_version_id": "mother"})
    assert len(keys) == count == 2


def test_per_node_target_requires_evidence_protocol():
    with pytest.raises(ValueError, match="dense-qa-v3"):
        config(qa_per_node_target=20)


@pytest.mark.asyncio
@pytest.mark.parametrize("review_case", ["normal", "negative", "invalid_group", "truncated_group", "same_question"])
async def test_one_shot_twenty_candidates_retained_with_review_flags(tmp_path, review_case):
    import hashlib
    node = dict(id="a", atlas_order=1, concept="声音", importance=5, dense_note="声音传播需要介质，真空不能传声。")
    item = dict(prompt="", qa_sources=[node], source_file="source.csv", source_ordinal=0,
                source_sha256="sha", character_start=0, character_end=20, style="dense_qa")
    evidence = [dict(node_id="a", quote="声音传播需要介质")]
    calls = dict(planner=0, draft=0, group=0)
    async def request(job, **kwargs):
        assert node["dense_note"] in job["prompt"]
        schema = job["qa_json_schema"]["properties"]
        if "plans" in schema:
            calls["planner"] += 1
            assert schema["plans"]["maxItems"] == 20
            assert "目标20题" in job["prompt"] and "没有后续补题" in job["prompt"]
            value = {"plans": [dict(evidence=evidence, angle="identity", difficulty="L1",
                intent="问题" + hashlib.sha256(str(i).encode()).hexdigest()[:24] + "？", source_ids=["a"]) for i in range(20)]}
        elif "duplicate_groups" in schema:
            calls["group"] += 1
            group = json.loads(job["prompt"].split("整组问答：")[1])
            assert len(group) == 20
            assert schema["duplicate_groups"]["items"]["properties"]["indices"]["items"]["enum"] == list(range(20))
            value = dict(duplicate_groups=[dict(indices=[0, 20] if review_case == "invalid_group" else [1, 0], reason="相同考点")])
        elif "grounded" in schema:
            value = dict(grounded=review_case != "negative", all_sources_needed=True, semantic_duplicate=review_case == "negative", reason="模型拒绝" if review_case == "negative" else "", evidence=evidence)
        else:
            calls["draft"] += 1
            plan = json.loads(job["prompt"].split("证据与问题规划：", 1)[1])
            value = dict(question="声音传播需要什么条件？" if review_case == "same_question" else plan["intent"], answer="声音传播需要介质。")
        return job, (json.dumps(value, ensure_ascii=False), 20, review_case == "truncated_group" and "duplicate_groups" in schema), True, 0
    cfg = DataGenerationCreate.model_validate({**config(qa_limit=20).model_dump(), "prompt_version":"dense-qa-v3",
        "qa_per_node_target":20, "qa_node_max_rounds":1, "qa_planning_mode":"one_shot"})
    partial = tmp_path / "partial.jsonl"
    report = {}
    with partial.open("a", encoding="utf-8") as stream:
        await generate_qa(iter([item]), cfg, request, stream, {"input_dataset_version_id":"mother"}, report, lambda *a, **k:None, tmp_path)
    batches = [json.loads(l) for l in partial.read_text(encoding="utf-8").splitlines()]
    assert calls == dict(planner=1, draft=20, group=1)
    assert len(batches) == 1 and len(batches[0]["qa_records"]) == 20
    assert report["qa_group_duplicate_removed"] == 0
    assert report["qa_duplicate_removed"] == report["qa_semantic_duplicate_removed"] == 0
    assert report["qa_plans_proposed"] == report["qa_plans_validated"] == 20
    assert report["qa_node_shortfalls"] == []
    assert report["qa_target_reached"] is True
    kept = batches[0]["qa_records"]
    assert len({r["qa_id"] for r in kept}) == 20
    assert all(r["review_status"] == "pending_independent_review" for r in kept)
    assert report["quality_status"] == "candidates_pending_independent_review"
    if review_case in {"invalid_group", "truncated_group"}:
        assert report["qa_group_review_failed"] == 1
        assert all(any(f["type"] == "group_review_failed" for f in r["review_flags"]) for r in kept)
    else:
        assert report["qa_group_duplicate_flagged"] == 2
    if review_case == "negative":
        assert report["qa_review_flagged"] == 20
        assert all(r["grounding_review"]["grounded"] is False for r in kept)
    if review_case == "same_question":
        assert report["qa_duplicate_flagged"] == 19
    final = tmp_path / "data.jsonl"
    finalize_qa_batches(partial, final, report)
    assert len(final.read_text(encoding="utf-8").splitlines()) == 20
    assert report["quality_status"] == "candidates_pending_independent_review"


@pytest.mark.parametrize("groups", [None, [{"indices":[0,20]}], [{"indices":[0,0]}], [{"indices":[True,2]}]])
def test_group_review_invalid_verdict_cannot_pass(groups):
    from llm_platform_core.dense_qa import group_duplicate_indices
    with pytest.raises(ValueError):
        group_duplicate_indices({"duplicate_groups":groups},20)


@pytest.mark.parametrize("change", [{"qa_per_node_target":0}, {"qa_node_max_rounds":2}, {"prompt_version":"dense-qa-v2"}])
def test_one_shot_contract(change):
    values = {**config().model_dump(), "prompt_version":"dense-qa-v3", "qa_planning_mode":"one_shot", "qa_per_node_target":20, "qa_node_max_rounds":1, **change}
    with pytest.raises(ValueError, match="one_shot"):
        DataGenerationCreate.model_validate(values)


def test_group_review_merges_observed_overlap_and_bridge():
    from llm_platform_core.dense_qa import group_duplicate_indices
    groups = [{"indices":[10,11,12]}, {"indices":[13,10,11,12]}]
    assert group_duplicate_indices({"duplicate_groups":groups},14) == {11,12,13}
    groups = [{"indices":[0,1]}, {"indices":[3,4]}, {"indices":[1,3]}]
    assert group_duplicate_indices({"duplicate_groups":groups},5) == {1,3,4}
    assert group_duplicate_indices({"duplicate_groups":list(reversed(groups))},5) == {1,3,4}


def test_grouped_qa_contract_and_validation():
    from llm_platform_core.dense_qa import grouped_qa_prompt_v3, stage_schema, validate_grouped_qa
    nodes = [dict(id="a", dense_note="河流发源于高山，向东注入海洋。", concept="河流")]
    schema = stage_schema("group_generate", nodes, evidence_first=True, plan_limit=20)
    assert schema["properties"]["items"]["maxItems"] == 20
    assert "唯一一次生成" in grouped_qa_prompt_v3(nodes, 20)
    value = {"items":[{"evidence":[{"node_id":"a","quote":"河流发源于高山"}],
        "angle":"identity","difficulty":"L1","source_ids":["a"],
        "question":"这条河流发源于哪里？","answer":"这条河流发源于高山。"}]}
    pairs = validate_grouped_qa(value, nodes, 20)
    assert len(pairs) == 1 and pairs[0][1]["answer"] == "这条河流发源于高山。"


def test_grouped_mode_contract():
    values = {**config().model_dump(), "prompt_version":"dense-qa-v3", "qa_planning_mode":"grouped",
              "qa_per_node_target":20, "qa_node_max_rounds":1}
    assert DataGenerationCreate.model_validate(values).qa_planning_mode == "grouped"
    with pytest.raises(ValueError, match="grouped"):
        DataGenerationCreate.model_validate({**values, "qa_per_node_target":0})


def test_group_schema_binds_actual_thirteen_candidate_ids():
    from llm_platform_core.dense_qa import stage_schema
    schema = stage_schema("group_review", [{"id":"a"}], group_count=13)
    ids = schema["properties"]["duplicate_groups"]["items"]["properties"]["indices"]["items"]["enum"]
    assert ids == list(range(13)) and 13 not in ids


def test_v3_single_node_synthesis_requires_distinct_supported_evidence():
    from llm_platform_core.dense_qa import validate_evidence_plans
    nodes = [dict(id="a", dense_note="上游湿润地区持续提供水量，因此河流能够穿越下游沙漠。")]
    evidence = [dict(node_id="a", quote="上游湿润地区持续提供水量"), dict(node_id="a", quote="河流能够穿越下游沙漠")]
    for angle in ["synthesis", "multi_hop"]:
        plan = dict(angle=angle, difficulty="L2", intent="上游水源与穿越下游沙漠之间有什么联系？", source_ids=["a"], evidence=evidence)
        assert validate_evidence_plans({"plans":[plan]}, nodes) == [plan]
        assert validate_plans({"plans":[plan]}, nodes) == []  # legacy protocol unchanged
        assert validate_evidence_plans({"plans":[dict(plan,evidence=[evidence[0],evidence[0]])]}, nodes) == []
        assert validate_evidence_plans({"plans":[dict(plan,evidence=evidence+[dict(node_id="a",quote="无依据的额外内容")])]}, nodes) == []
        assert validate_evidence_plans({"plans":[dict(plan,difficulty="L4")]}, nodes) == []
