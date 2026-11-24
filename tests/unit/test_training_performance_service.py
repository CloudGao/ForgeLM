from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from llm_platform import models as m
from llm_platform.schemas import TrainingPerformanceCreate
from llm_platform.services import training_performance as service


class Records:
    def __init__(self, *rows):
        self.rows = {row.id: row for row in rows}

    def get(self, model, identity):
        return self.rows.get(identity)

    def scalar(self, query):
        return self.rows["attempt" if query.column_descriptions[0]["entity"] is m.TaskAttempt else "receipt"]


def test_adopting_recommendation_returns_new_draft_without_mutating_source():
    from llm_platform.routers.training import training_performance_draft
    revision = {"tokenizer_version_id": "tokenizer", "effective_config": {"engine_config": {"key": "value"}}, "input_model_ref": {"type": "scratch"}}
    profile = {"partial_rope": "flash_attn", "linear_cross_entropy": "fla"}
    task = NS(id="comparison", kind="training_performance", project_id="project", observed_state="succeeded",
              payload={"performance_probe": {"source_training_run_id": "run", "source_stage": {"type": "pretrain", "micro_batch_size": 1}}, "code": {"version_id": "new-code"}},
              result={"performance_report": {"recommendation": {"micro_batch_size": 2, "optimizations": profile}}})
    source = NS(id="run", project_id="project", config_snapshot={"revision": revision})
    before = deepcopy(task.payload)
    result = training_performance_draft("project", task.id, Records(task, source))
    assert result["code_version_id"] == "new-code"
    assert result["stages"][0]["optimizations"] == profile
    assert result["input_model_ref"] == {"type": "full_state_continuation", "performance_task_id": "comparison"}
    assert task.payload == before and revision["input_model_ref"]["type"] == "scratch"
    with pytest.raises(HTTPException):
        training_performance_draft("other", task.id, Records(task, source))


def records():
    source = NS(id="source", project_id="project", kind="train", observed_state="paused",
                result={"controlled_state": "paused", "step": 10, "safe_pause_checkpoint": "/data/run/trainer/checkpoint-10"},
                payload={"training_run_id": "run", "stages": [{"type": "pretrain", "micro_batch_size": 4,
                         "effective_batch_tokens": 262144, "sequence_length": 4096, "learning_rate": 0.0003}],
                         "training_data_view_inputs": [{"version_id": "view"}], "tokenizer": {"version_id": "tokenizer"},
                         "training_data_range": {"planned_steps": 57221, "sequence_start": 0}, "command": {"control": {}},
                         "pipeline_node_id": "original", "resource_batch_policy": {"mode": "auto"}})
    version = NS(id="code", artifact_id="code-artifact", version=5, git_commit=None, dependency_spec="",
                 entrypoints={"training": "train.py"}, training_contract={"performance_probe": service.PROTOCOL})
    attempt = NS(id="attempt", worker_id="worker", environment_snapshot={"resource_batch_plan": {
        "selected": {"values": {"micro_batch_size": 8, "gradient_accumulation_steps": 8}}}})
    artifact = NS(id="code-artifact", availability_state="available", content_hash="hash")
    receipt = NS(id="receipt", payload={"protocol": "forgelm-command-control-file-v1", "action": "safe_pause",
                                         "step": 10, "checkpoint": "/data/run/trainer/checkpoint-10"})
    return Records(source, version, attempt, artifact, receipt), source, version


def test_withdrawn_engineering_options_are_rejected_instead_of_silently_ignored():
    from llm_platform.schemas import validate_training_stage_names
    from pydantic import ValidationError

    with pytest.raises(ValueError, match="已撤回"):
        validate_training_stage_names([{"name": "pretrain", "type": "pretrain", "engineering": {"shared_rope": True}}])
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingPerformanceCreate(source_task_id="source", engine_version_id="code",
                                  candidates=[{"micro_batch_size": 8, "engineering": {"shared_rope": True}}])



def test_native_comparison_freezes_actual_baseline_and_preserves_source(monkeypatch):
    session, source, version = records()
    before = deepcopy(source.payload)
    created = {}

    def create(_session, **kwargs):
        created.update(kwargs)
        return NS(id="probe")

    monkeypatch.setattr(service, "create_task", create)
    monkeypatch.setattr(service, "commit_or_flush", lambda _: None)
    request = TrainingPerformanceCreate(source_task_id="source", engine_version_id="code",
                                       candidates=[{"micro_batch_size": 16}])
    assert service.create_comparison(session, "project", request, version)["task_id"] == "probe"
    assert source.payload == before
    frozen = created["payload"]
    assert frozen["performance_probe"]["candidates"][0]["micro_batch_size"] == 8
    assert "control" not in frozen["command"]
    assert "training_run_id" not in frozen and "pipeline_node_id" not in frozen
    assert frozen["resource_request"]["worker_selector"] == {"worker_id": "worker"}
    assert frozen["command"]["args"] == ["train.py", "--model-config", "configs/model.json"]
    assert frozen["performance_probe"]["legacy_position_from_frozen_range"] is False
    source.payload["effective_config"] = {"engine_config": {
        "model_implementation": "microlm_306m_v3.modeling:MicroLMForCausalLM"}}
    service.create_comparison(session, "project", request, version)
    assert created["payload"]["performance_probe"]["legacy_position_from_frozen_range"] is True
    source.payload["code"] = {"training_contract": {"performance_probe": service.PROTOCOL}}
    service.create_comparison(session, "project", request, version)
    assert created["payload"]["performance_probe"]["legacy_position_from_frozen_range"] is False
    source.payload["input_model_ref"] = {"config_path": "configs/model-800m-4k.json", "parameter_count": 804634976}
    service.create_comparison(session, "project", request, version)
    assert created["payload"]["command"]["args"] == ["train.py", "--model-config", "configs/model-800m-4k.json"]
    assert created["payload"]["input_model_ref"] == source.payload["input_model_ref"]
    source.payload["resource_request"] = {"gpus": 2}
    with pytest.raises(HTTPException):
        service.create_comparison(session, "project", request, version)
    source.payload.pop("resource_request")
    source.observed_state = "running"
    with pytest.raises(HTTPException, match="409"):
        service.create_comparison(session, "project", request, version)


def test_continuation_retains_original_range_and_rejects_semantic_changes():
    session, source, version = records()
    stage = deepcopy(source.payload["stages"][0])
    recommended = {"micro_batch_size": 8, "activation_checkpointing": True, "checkpoint_every_n_layers": 2}
    probe = NS(id="probe", project_id="project", kind="training_performance", observed_state="succeeded",
               payload={**deepcopy(source.payload), "code": {"version_id": version.id},
                        "performance_probe": {"source_task_id": source.id, "source_step": 10,
                                              "source_stage": stage, "source_training_run_id": "run"}},
               result={"performance_report": {"schema_version": service.PROTOCOL, "recommendation": recommended},
                       "output_artifacts": [{"ref": "full-trainer-state", "artifact_id": "checkpoint"}]})
    checkpoint = NS(id="checkpoint", kind="checkpoint", availability_state="available", content_hash="state-hash")
    session.rows.update(probe=probe, checkpoint=checkpoint)
    revision = NS(input_model_ref={"type": "full_state_continuation", "performance_task_id": "probe"},
                  code_version_id="code", tokenizer_version_id="tokenizer")
    updated = {**stage, **recommended}
    binding = service.freeze_continuation(session, "project", revision, [updated])
    assert binding["sequence_offset"] == 0 and binding["token_count"] == 0 and binding["step"] == 10
    with pytest.raises(HTTPException):
        service.freeze_continuation(session, "project", revision, [{**updated, "learning_rate": 0.0004}])
    with pytest.raises(HTTPException):
        service.freeze_continuation(session, "project", revision, [{**updated, "micro_batch_size": 16}])
    source.observed_state = "running"
    with pytest.raises(HTTPException):
        service.freeze_continuation(session, "project", revision, [updated])


def test_native_performance_node_is_visible_and_typed():
    from llm_platform.services.native_pipeline_nodes import operation_catalog, validate_node

    assert any(op["id"] == "training.performance" for op in operation_catalog("preparation"))
    node = {"id": "probe", "type": "preparation", "config": {"native_operation": "training.performance",
            "parameters": {"source_task_id": "source", "engine_version_id": "code", "candidates": [{"micro_batch_size": 8}]}}}
    assert validate_node(node, []) == []
    node["config"]["parameters"]["candidates"][0]["micro_batch_size"] = 0
    assert validate_node(node, [])


def test_automatic_search_is_default_without_candidates_and_requires_engine_capability(monkeypatch):
    session, source, version = records()
    created = {}
    monkeypatch.setattr(service, "create_task", lambda _s, **kw: (created.update(kw) or NS(id="probe")))
    monkeypatch.setattr(service, "commit_or_flush", lambda _: None)
    request = TrainingPerformanceCreate(source_task_id="source", engine_version_id="code")
    with pytest.raises(HTTPException):
        service.create_comparison(session, "project", request, version)
    version.training_contract["performance_search"] = service.SEARCH_PROTOCOL
    with pytest.raises(HTTPException):
        service.create_comparison(session, "project", request, version)
    source.payload["validation_data_input"] = {"sequence_count": 256}
    service.create_comparison(session, "project", request, version)
    probe = created["payload"]["performance_probe"]
    assert probe["search_mode"] == "auto" and probe["memory_headroom_mb"] is None
    assert any(c["micro_batch_size"] == 64 for c in probe["candidates"])
    assert any(c["activation_checkpointing"] is False for c in probe["candidates"])
    assert probe["max_candidates"] == 12
    manual = request.model_copy(update={"search_mode": "manual"})
    with pytest.raises(HTTPException):
        service.create_comparison(session, "project", manual, version)
