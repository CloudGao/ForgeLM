from __future__ import annotations

import tempfile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    PlanRevision,
    SupervisionMixture,
    SupervisionMixtureVersion,
    Task,
    TokenizerVersion,
    TrainingPlan,
    TrainingRun,
)
from llm_platform.routers.training import _freeze_resume_checkpoint
from llm_platform.services.artifacts import artifact_storage


def test_chat_template_and_supervision_domain_apis(client: TestClient) -> None:
    project = client.post(
        "/api/projects",
        json={
            "name": "Post Training Domain",
            "target_model_name": "PostLM",
            "workspace_uri": tempfile.mkdtemp(prefix="post-training-domain-"),
        },
    ).json()
    template_response = client.post(
        f"/api/projects/{project['id']}/chat-templates",
        json={
            "name": "Default chat",
            "template_body": "{% for message in messages %}{% if message.role == 'assistant' %}{% generation %}{{ message.content }}{% endgeneration %}{% else %}{{ message.content }}{% endif %}{% endfor %}",
            "role_schema": {"roles": ["system", "user", "assistant"]},
            "required_special_tokens": [],
            "assistant_mask_mode": "generation_spans_v1",
        },
    )
    assert template_response.status_code == 201, template_response.text
    template = template_response.json()
    assert template["versions"][0]["validation_report"]["assistant_mask_capable"] is True

    with SessionLocal() as session:
        snapshot = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            payload=b'{"messages":[]}\n',
            filename="records.jsonl",
        )
        manifest = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="record_manifest",
            payload=b'{"format":"test"}\n',
            filename="manifest.jsonl",
        )
        dataset = Dataset(
            project_id=project["id"],
            name="Conversation source",
            source_type="local",
            source_uri="test://conversation",
            status="ready",
        )
        session.add(dataset)
        session.flush()
        version = DatasetVersion(
            dataset_id=dataset.id,
            version=1,
            state="ready",
            tier="bronze",
            artifact_id=snapshot.id,
            record_manifest_artifact_id=manifest.id,
            document_count=1,
            token_estimate=10,
        )
        session.add(version)
        session.commit()
        version_id = version.id

    mapped_response = client.post(
        f"/api/projects/{project['id']}/supervision-datasets",
        json={
            "name": "Conversation mapping",
            "dataset_version_id": version_id,
            "kind": "conversation",
            "mapping": {"messages_path": "messages"},
            "filters": {},
            "start_scan": False,
        },
    )
    assert mapped_response.status_code == 201, mapped_response.text
    mapped = mapped_response.json()
    mixture_response = client.post(
        f"/api/projects/{project['id']}/supervision-mixtures",
        json={
            "name": "SFT mixture",
            "entries": [{"supervision_dataset_version_id": mapped["id"], "weight": 1}],
            "seed": 42,
        },
    )
    assert mixture_response.status_code == 201, mixture_response.text
    mixture_version = mixture_response.json()
    assert mixture_version["kind"] == "conversation"
    assert mixture_version["entries"][0]["supervision_name"] == "Conversation mapping"
    assert mixture_version["entries"][0]["dataset_name"] == "Conversation source"
    assert mixture_version["entries"][0]["normalized_weight"] == 1
    mixture_id = mixture_version["mixture_id"]
    detail = client.get(f"/api/supervision-mixtures/{mixture_id}")
    assert detail.status_code == 200
    assert detail.json()["versions"][0]["id"] == mixture_version["id"]
    updated = client.patch(
        f"/api/supervision-mixtures/{mixture_id}",
        json={"name": "SFT mixture renamed", "description": "中文与代码"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "SFT mixture renamed"
    next_version = client.post(
        f"/api/supervision-mixtures/{mixture_id}/versions",
        json={
            "entries": [{"supervision_dataset_version_id": mapped["id"], "weight": 3}],
            "seed": 43,
        },
    )
    assert next_version.status_code == 201, next_version.text
    assert next_version.json()["version"] == 2
    assert len(client.get(f"/api/projects/{project['id']}/chat-templates").json()) == 1
    assert len(client.get(f"/api/projects/{project['id']}/supervision-datasets").json()) == 1
    assert len(client.get(f"/api/projects/{project['id']}/supervision-mixtures").json()) == 1

    with SessionLocal() as session:
        plan = TrainingPlan(project_id=project["id"], name="Uses SFT mixture")
        session.add(plan)
        session.flush()
        revision = PlanRevision(
            plan_id=plan.id,
            revision=1,
            stages=[{"type": "sft", "data_ref_id": next_version.json()["id"]}],
        )
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        session.commit()
    blocked = client.delete(f"/api/supervision-mixtures/{mixture_id}")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "supervision_mixture_in_use"
    assert "训练计划" in blocked.json()["detail"]["message"]


def test_unused_supervision_mixture_can_be_deleted(client: TestClient) -> None:
    project = client.post(
        "/api/projects",
        json={
            "name": "Disposable Post Training Domain",
            "target_model_name": "DeleteLM",
            "workspace_uri": tempfile.mkdtemp(prefix="post-training-delete-"),
        },
    ).json()
    with SessionLocal() as session:
        mixture = SupervisionMixture(project_id=project["id"], name="Disposable mixture")
        session.add(mixture)
        session.flush()
        version = SupervisionMixtureVersion(
            mixture_id=mixture.id,
            version=1,
            kind="conversation",
            entries=[],
            seed=42,
            identity_hash="d" * 64,
        )
        session.add(version)
        session.flush()
        mixture.current_version_id = version.id
        mixture_id = mixture.id
        session.commit()

    response = client.delete(f"/api/supervision-mixtures/{mixture_id}")
    assert response.status_code == 204, response.text
    assert client.get(f"/api/supervision-mixtures/{mixture_id}").status_code == 404


def test_supervision_lookup_freezes_both_dataset_inputs(client: TestClient) -> None:
    project = client.post(
        "/api/projects",
        json={
            "name": "Supervision Lookup",
            "target_model_name": "LookupLM",
            "workspace_uri": tempfile.mkdtemp(prefix="supervision-lookup-"),
        },
    ).json()
    version_ids: list[str] = []
    with SessionLocal() as session:
        for name, payload in (
            ("Questions", b'{"question":"Q","answer":"A","node_id":"n1"}\n'),
            ("Mother", b'{"id":"n1","dense_note":"Knowledge"}\n'),
        ):
            snapshot = artifact_storage.put_bytes(
                session,
                project_id=project["id"],
                kind="dataset_snapshot",
                payload=payload,
                filename=f"{name.lower()}.jsonl",
            )
            manifest = artifact_storage.put_bytes(
                session,
                project_id=project["id"],
                kind="record_manifest",
                payload=b'{"format":"test"}\n',
                filename=f"{name.lower()}-records.jsonl",
            )
            dataset = Dataset(
                project_id=project["id"],
                name=name,
                source_type="local",
                status="ready",
            )
            session.add(dataset)
            session.flush()
            version = DatasetVersion(
                dataset_id=dataset.id,
                version=1,
                state="ready",
                artifact_id=snapshot.id,
                record_manifest_artifact_id=manifest.id,
                document_count=1,
            )
            session.add(version)
            session.flush()
            version_ids.append(version.id)
        session.commit()

    response = client.post(
        f"/api/projects/{project['id']}/supervision-datasets",
        json={
            "name": "Recall mapping",
            "dataset_version_id": version_ids[0],
            "kind": "conversation",
            "mapping": {
                "prompt_field": "question",
                "completion_parts": [
                    {"field": {"path": "_lookup.mother.dense_note"}},
                    {"field": {"path": "answer"}},
                ],
            },
            "lookups": [
                {
                    "name": "mother",
                    "dataset_version_id": version_ids[1],
                    "source_key": "node_id",
                    "lookup_key": "id",
                    "fields": {"dense_note": "dense_note"},
                }
            ],
            "filters": {},
        },
    )
    assert response.status_code == 201, response.text
    mapped = response.json()
    frozen_lookup = mapped["mapping_config"]["_lookups"][0]
    assert frozen_lookup["dataset_version_id"] == version_ids[1]
    assert frozen_lookup["dataset_artifact_hash"]
    with SessionLocal() as session:
        task = session.get(Task, mapped["task_id"])
        assert task is not None
        assert [item["dataset_version_id"] for item in task.payload["data_inputs"]] == version_ids


def test_native_sft_preflight_requests_supervision_data_not_tokenized_dataset(
    client: TestClient,
) -> None:
    project = client.post(
        "/api/projects",
        json={
            "name": "Native SFT Preflight",
            "target_model_name": "PostLM",
            "workspace_uri": tempfile.mkdtemp(prefix="native-sft-preflight-"),
        },
    ).json()
    template = client.post(
        f"/api/projects/{project['id']}/chat-templates",
        json={
            "name": "SFT chat",
            "template_body": "{% for message in messages %}{% if message.role == 'assistant' %}{% generation %}{{ message.content }}{% endgeneration %}{% else %}{{ message.content }}{% endif %}{% endfor %}",
            "role_schema": {"roles": ["system", "user", "assistant"]},
            "required_special_tokens": [],
            "assistant_mask_mode": "generation_spans_v1",
        },
    ).json()
    template_version_id = template["versions"][0]["id"]

    with SessionLocal() as session:
        tokenizer = TokenizerVersion(
            project_id=project["id"],
            name="Locked tokenizer",
            version=1,
            algorithm="bpe",
            vocab_size=32_000,
            data_ref_type="dataset_version",
            data_ref_id="test-source",
            status="ready",
            locked=True,
        )
        mixture = SupervisionMixture(
            project_id=project["id"], name="Ready conversation mixture"
        )
        session.add_all([tokenizer, mixture])
        session.flush()
        mixture_version = SupervisionMixtureVersion(
            mixture_id=mixture.id,
            version=1,
            kind="conversation",
            entries=[],
            seed=42,
            identity_hash="a" * 64,
            status="ready",
            locked=True,
        )
        session.add(mixture_version)
        session.flush()
        mixture.current_version_id = mixture_version.id
        plan = TrainingPlan(project_id=project["id"], name="Native SFT")
        session.add(plan)
        session.flush()
        revision = PlanRevision(
            plan_id=plan.id,
            revision=1,
            stages=[
                {
                    "name": "SFT",
                    "type": "sft",
                    "data_ref_id": mixture_version.id,
                    "chat_template_version_id": template_version_id,
                    "objective": {
                        "schema_version": 1,
                        "kind": "sft",
                        "loss_scope": "assistant_only",
                    },
                    "budget": {"unit": "examples", "value": 128},
                    "sequence_length": 2048,
                    "micro_batch_size": 4,
                    "gradient_accumulation_steps": 8,
                }
            ],
            tokenizer_version_id=tokenizer.id,
            engine_plugin_id="builtin.command.training",
            launcher_plugin_id="builtin.local-process",
            resource_request={"gpus": 0},
            effective_config={
                "engine_config": {
                    "data_input_mode": "tokenized_required",
                    "executable": "python",
                }
            },
        )
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        session.commit()
        plan_id = plan.id

    response = client.post(f"/api/training-plans/{plan_id}/preflight", json={})
    assert response.status_code == 200, response.text
    issue_codes = {item["code"] for item in response.json()["issues"]}
    assert "supervision_data_preparation_required" in issue_codes
    assert "tokenized_dataset_unavailable" not in issue_codes


def test_native_sft_can_fork_same_project_checkpoint_from_another_plan(
    client: TestClient,
) -> None:
    project = client.post(
        "/api/projects",
        json={
            "name": "Native SFT Checkpoint Fork",
            "target_model_name": "PostLM",
            "workspace_uri": tempfile.mkdtemp(prefix="native-sft-checkpoint-fork-"),
        },
    ).json()
    with SessionLocal() as session:
        source_plan = TrainingPlan(project_id=project["id"], name="Pretraining source")
        session.add(source_plan)
        session.flush()
        source_revision = PlanRevision(
            plan_id=source_plan.id,
            revision=1,
            stages=[{"type": "pretrain", "target_tokens": 1_000}],
            effective_config={"engine_config": {"data_input_mode": "tokenized_required"}},
        )
        session.add(source_revision)
        session.flush()
        source_plan.current_revision_id = source_revision.id
        task = Task(
            project_id=project["id"],
            kind="training",
            title="Pretraining source",
        )
        session.add(task)
        session.flush()
        run = TrainingRun(
            project_id=project["id"],
            plan_revision_id=source_revision.id,
            task_id=task.id,
            status="succeeded",
        )
        session.add(run)
        session.flush()
        checkpoint_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="checkpoint",
            payload=b"checkpoint",
            filename="checkpoint.tar.gz",
        )
        checkpoint = Checkpoint(
            project_id=project["id"],
            training_run_id=run.id,
            step=10,
            token_count=1_000,
            artifact_id=checkpoint_artifact.id,
            metrics={},
            status="available",
        )
        session.add(checkpoint)

        sft_plan = TrainingPlan(project_id=project["id"], name="SFT fork")
        pretrain_plan = TrainingPlan(project_id=project["id"], name="Pretraining fork")
        session.add_all([sft_plan, pretrain_plan])
        session.flush()
        sft_revision = PlanRevision(
            plan_id=sft_plan.id,
            revision=1,
            input_model_ref={
                "type": "checkpoint_finetune",
                "checkpoint_id": checkpoint.id,
            },
            stages=[
                {
                    "type": "sft",
                    "data_ref_id": "supervision-mixture",
                    "chat_template_version_id": "chat-template",
                    "objective": {
                        "schema_version": 1,
                        "kind": "sft",
                        "loss_scope": "assistant_only",
                    },
                    "budget": {"unit": "examples", "value": 128},
                }
            ],
            effective_config={"engine_config": {"data_input_mode": "tokenized_required"}},
        )
        pretrain_revision = PlanRevision(
            plan_id=pretrain_plan.id,
            revision=1,
            stages=[{"type": "pretrain", "target_tokens": 1_000}],
            effective_config={"engine_config": {"data_input_mode": "tokenized_required"}},
        )
        session.add_all([sft_revision, pretrain_revision])
        session.flush()

        frozen_from_plan = _freeze_resume_checkpoint(
            session,
            plan=sft_plan,
            revision=sft_revision,
            selected_stage_indexes=[0],
            checkpoint_id=None,
        )
        assert frozen_from_plan is not None
        assert frozen_from_plan["checkpoint_id"] == checkpoint.id
        assert frozen_from_plan["mode"] == "checkpoint_finetune"

        frozen = _freeze_resume_checkpoint(
            session,
            plan=sft_plan,
            revision=sft_revision,
            selected_stage_indexes=[0],
            checkpoint_id=checkpoint.id,
        )
        assert frozen is not None
        assert frozen["checkpoint_id"] == checkpoint.id
        assert frozen["mode"] == "checkpoint_finetune"
        assert frozen["sequence_offset"] is None
        assert frozen["sample_offset"] is None

        missing_plan = TrainingPlan(project_id=project["id"], name="Missing SFT input")
        session.add(missing_plan)
        session.flush()
        missing_revision = PlanRevision(
            plan_id=missing_plan.id,
            revision=1,
            input_model_ref={"type": "checkpoint_finetune"},
            stages=sft_revision.stages,
            effective_config={"engine_config": {"data_input_mode": "tokenized_required"}},
        )
        session.add(missing_revision)
        session.flush()
        with pytest.raises(HTTPException) as missing_error:
            _freeze_resume_checkpoint(
                session,
                plan=missing_plan,
                revision=missing_revision,
                selected_stage_indexes=[0],
                checkpoint_id=None,
            )
        assert getattr(missing_error.value, "status_code", None) == 409
        assert getattr(missing_error.value, "detail", {}).get("code") == (
            "input_model_checkpoint_required"
        )

        with pytest.raises(HTTPException) as error:
            _freeze_resume_checkpoint(
                session,
                plan=pretrain_plan,
                revision=pretrain_revision,
                selected_stage_indexes=[0],
                checkpoint_id=checkpoint.id,
            )
        assert getattr(error.value, "status_code", None) == 409
        assert getattr(error.value, "detail", {}).get("code") == "invalid_resume_checkpoint"

        # Continued pretraining is an explicit weight fork, not a resume of the
        # source corpus. Preserve model lineage/tokens without reading its View.
        pretrain_revision.input_model_ref = {
            "type": "checkpoint_finetune",
            "checkpoint_id": checkpoint.id,
            "data_cursor_policy": "reset",
        }
        checkpoint.metrics = {"sequence_offset": 999, "sample_offset": 123}
        run.config_snapshot = {"training_data_view_inputs": [{"version_id": "old-view"}]}
        session.flush()
        fork = _freeze_resume_checkpoint(
            session, plan=pretrain_plan, revision=pretrain_revision,
            selected_stage_indexes=[0], checkpoint_id=None,
        )
        assert fork is not None
        assert fork["mode"] == "weights_only_new_data"
        assert fork["checkpoint_id"] == checkpoint.id
        assert fork["token_count"] == 1_000
        assert fork["sequence_offset"] == fork["sample_offset"] == 0
        assert "training_data_view" not in fork

        # A reset declaration cannot bypass project ownership or stage gates.
        checkpoint.project_id = "another-project"
        with pytest.raises(HTTPException) as cross_project:
            _freeze_resume_checkpoint(
                session, plan=pretrain_plan, revision=pretrain_revision,
                selected_stage_indexes=[0], checkpoint_id=checkpoint.id,
            )
        assert cross_project.value.detail["code"] == "invalid_resume_checkpoint"
        checkpoint.project_id = project["id"]
        pretrain_revision.stages = [
            {"type": "pretrain", "target_tokens": 1_000},
            {"type": "pretrain", "target_tokens": 1_000},
        ]
        with pytest.raises(HTTPException) as skipped_stage:
            _freeze_resume_checkpoint(
                session, plan=pretrain_plan, revision=pretrain_revision,
                selected_stage_indexes=[1], checkpoint_id=checkpoint.id,
            )
        assert skipped_stage.value.detail["code"] == "invalid_resume_checkpoint"

        # A checkpoint produced inside the new plan must keep its own cursor.
        run.plan_revision_id = pretrain_revision.id
        run.config_snapshot = {}
        session.flush()
        resumed = _freeze_resume_checkpoint(
            session, plan=pretrain_plan, revision=pretrain_revision,
            selected_stage_indexes=[0], checkpoint_id=checkpoint.id,
        )
        assert resumed is not None
        assert resumed["mode"] == "weights_and_data_cursor"
        assert resumed["sequence_offset"] == 999
        assert resumed["sample_offset"] == 123
