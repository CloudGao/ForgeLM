from copy import deepcopy
from uuid import uuid4

import pytest
from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.services.training_history import ensure_history_instances, source_run_id
from llm_platform.services.training_upstream_lineage import version_refs
from sqlalchemy import select
from test_training_execution_graph import legacy_training
from test_training_upstream_lineage import attempt, dataset, endpoint, project, save


def tokenizer(session, pid, data):
    return save(
        session,
        m.TokenizerVersion(
            project_id=pid,
            name="tokens",
            version=1,
            algorithm="bpe",
            vocab_size=256,
            data_ref_type="dataset_version",
            data_ref_id=data.id,
        ),
    )


def supervision_chain(session, pid):
    raw = dataset(session, pid)[0]
    mapped_set = save(session, m.SupervisionDataset(project_id=pid, name="mapped"))
    mapped = save(
        session,
        m.SupervisionDatasetVersion(
            supervision_dataset_id=mapped_set.id,
            version=1,
            dataset_version_id=raw.id,
            kind="sft",
            identity_hash=uuid4().hex,
        ),
    )
    mixture = save(session, m.SupervisionMixture(project_id=pid, name="mixed"))
    mixed = save(
        session,
        m.SupervisionMixtureVersion(
            mixture_id=mixture.id,
            version=1,
            kind="sft",
            identity_hash=uuid4().hex,
            entries=[{"supervision_dataset_version_id": mapped.id}],
        ),
    )
    tokens = tokenizer(session, pid, raw)
    template = save(session, m.ChatTemplate(project_id=pid, name="chat"))
    template_version = save(
        session,
        m.ChatTemplateVersion(
            template_id=template.id,
            version=1,
            template_body="test",
            content_hash=uuid4().hex,
        ),
    )
    corpus = save(
        session,
        m.SupervisionCorpusVersion(
            project_id=pid,
            mixture_version_id=mixed.id,
            tokenizer_version_id=tokens.id,
            chat_template_version_id=template_version.id,
            objective_kind="sft",
            identity_hash=uuid4().hex,
            format="forgelm-sft-corpus-v1",
            status="ready",
        ),
    )
    view = save(
        session,
        m.SupervisionTrainingViewVersion(
            project_id=pid,
            corpus_version_id=corpus.id,
            objective_kind="sft",
            identity_hash=uuid4().hex,
            format="forgelm-sft-training-index-v1",
            budget_unit="supervised_tokens",
            budget_value=100,
            status="ready",
        ),
    )
    return raw, mapped, mixed, corpus, view


def assert_layered_chain(graph, chain, target):
    nodes = graph.get("artifacts", graph.get("nodes"))
    assets = {n["asset_id"]: n for n in nodes if n["kind"] == "asset"}
    edges = {(e["source"], e["target"]) for e in graph.get("bindings", graph.get("edges"))}
    assert set(chain) <= set(assets)
    assert (assets[chain[-1]]["id"], target) in edges
    assert all((assets[identity]["id"], target) not in edges for identity in chain[:-1])
    for previous, current in zip(chain, chain[1:], strict=False):
        producer = next(source for source, child in edges if child == assets[current]["id"])
        assert (assets[previous]["id"], producer) in edges
    return assets, edges


@pytest.mark.parametrize("historical", [False, True])
def test_supervision_chain_expands_without_flattening_snapshot_inputs(client, historical):
    with SessionLocal() as session:
        pid = project(session).id
        raw, mapped, mixed, corpus, view = supervision_chain(session, pid)
        chain = [item.id for item in (raw, mapped, mixed, corpus, view)]
        config = {
            "stages": [
                {"type": "sft", "data_ref_id": mixed.id, "supervision_training_view_id": view.id}
            ],
            "data_inputs": [{"dataset_version_id": raw.id}],
            "supervision_training_view_inputs": [
                {
                    "version_id": view.id,
                    "supervision_corpus": {"version_id": corpus.id, "mixture_version_id": mixed.id},
                }
            ],
        }
        if historical:
            training, _, checkpoints = legacy_training(session, pid, raw)
            training.config_snapshot = {"revision": {"stages": config["stages"]}, **config}
            snapshot = deepcopy(training.config_snapshot)
            ensure_history_instances(session, pid)
            instance = next(
                i
                for i in session.scalars(select(m.ProjectPipelineInstance))
                if source_run_id(i) == training.id
            )
            target, record_id = "training", training.id
            model = save(
                session,
                m.ModelVersion(
                    project_id=pid,
                    name="output",
                    version=1,
                    source_checkpoint_id=checkpoints[-1].id,
                ),
            )
            checkpoint_id, model_id = checkpoints[-1].id, model.id
        else:
            run, row = attempt(session, pid, [])
            row.config_snapshot = {"training_spec": config}
            instance = session.get(m.ProjectPipelineInstance, run.instance_id)
            target, record_id = row.node_id, row.id
            snapshot = deepcopy(row.config_snapshot)
            upstream_url = endpoint(run)
        frozen_graph, content_hash = deepcopy(instance.nodes), instance.content_hash
        route = f"/api/projects/{pid}/training-instances/{instance.id}/execution-graph"
        instance_id = instance.id
        session.commit()
    response = client.get(route)
    assert response.status_code == 200, response.text
    graph = response.json()
    assets, edges = assert_layered_chain(graph, chain, target)
    assert not assets[chain[-1]]["context_only"]
    assert all(assets[identity]["context_only"] for identity in chain[:-1])
    assert not graph["truncated"] and not graph["warnings"]
    if historical:
        assert (target, assets[checkpoint_id]["id"]) in edges
        assert (target, assets[model_id]["id"]) in edges
        assert (assets[checkpoint_id]["id"], assets[model_id]["id"]) in edges
    else:
        assert_layered_chain(client.get(upstream_url).json(), chain, target)
    with SessionLocal() as session:
        record = session.get(m.TrainingRun if historical else m.ProjectPipelineNodeRun, record_id)
        assert record.config_snapshot == snapshot
        instance = session.get(m.ProjectPipelineInstance, instance_id)
        assert instance.nodes == frozen_graph and instance.content_hash == content_hash


def test_pretraining_keeps_parallel_mixture_and_corpus_inputs_and_independent_data(client):
    with SessionLocal() as session:
        pid = project(session).id
        raw, independent = dataset(session, pid)[0], dataset(session, pid)[0]
        tokens = tokenizer(session, pid, raw)
        mixture = save(session, m.DataMixture(project_id=pid, name="mixture"))
        mixed = save(
            session,
            m.DataMixtureVersion(
                mixture_id=mixture.id,
                version=1,
                entries=[{"dataset_version_id": raw.id}],
            ),
        )
        corpus = save(
            session,
            m.TokenizedCorpusVersion(
                project_id=pid,
                dataset_version_id=raw.id,
                tokenizer_version_id=tokens.id,
                identity_hash=uuid4().hex,
                status="ready",
            ),
        )
        view = save(
            session,
            m.TrainingDataViewVersion(
                project_id=pid,
                mixture_version_id=mixed.id,
                tokenizer_version_id=tokens.id,
                identity_hash=uuid4().hex,
                sequence_length=128,
                requested_tokens=1024,
                config_snapshot={"corpora": [{"version_id": corpus.id}]},
            ),
        )
        run, row = attempt(session, pid, [])
        row.config_snapshot = {
            "training_data_view_inputs": [
                {
                    "version_id": view.id,
                    "mixture_version_id": mixed.id,
                    "source_corpora": [{"version_id": corpus.id}],
                }
            ],
            "data_inputs": [{"dataset_version_id": raw.id}, {"dataset_version_id": independent.id}],
        }
        route = f"/api/projects/{pid}/training-instances/{run.instance_id}/execution-graph"
        ids = {
            "raw": raw.id,
            "mixed": mixed.id,
            "corpus": corpus.id,
            "view": view.id,
            "independent": independent.id,
        }
        shallow_run, shallow_row = attempt(session, pid, [])
        shallow_row.config_snapshot = {
            "training_data_view_inputs": [{"version_id": view.id}],
            "data_inputs": [{"dataset_version_id": raw.id}],
        }
        shallow_route = (
            f"/api/projects/{pid}/training-instances/{shallow_run.instance_id}/execution-graph"
        )
        session.commit()
    graph = client.get(route).json()
    assets = {n["asset_id"]: n["id"] for n in graph["artifacts"] if n["kind"] == "asset"}
    names = {key: assets[identity] for key, identity in ids.items()}
    edges = {(e["source"], e["target"]) for e in graph["bindings"]}
    view_producer = next(source for source, target in edges if target == names["view"])
    assert {source for source, target in edges if target == "train"} == {
        names["view"],
        names["independent"],
    }
    assert {source for source, target in edges if target == view_producer} == {
        names["mixed"],
        names["corpus"],
    }
    corpus_producer = next(source for source, target in edges if target == names["corpus"])
    assert {source for source, target in edges if target == corpus_producer} == {names["raw"]}
    # A depth cutoff cannot justify hiding an input whose alternate path is absent.
    shallow = client.get(shallow_route + "?max_depth=1").json()
    shallow_edges = {(e["source"], e["target"]) for e in shallow["bindings"]}
    assert shallow["truncated"]
    assert (names["raw"], "train") in shallow_edges


def test_parser_recognizes_typed_corpus_containers_but_not_metadata_or_commands():
    refs = version_refs(
        {
            "supervision_training_view_inputs": [
                {
                    "version_id": "view",
                    "supervision_corpus": {
                        "version_id": "sft-corpus",
                        "mixture_version_id": "sft-mix",
                    },
                }
            ],
            "training_data_view_inputs": [
                {"version_id": "pretrain-view", "source_corpora": [{"version_id": "corpus"}]}
            ],
            "stages": [{"supervision_training_view_id": "view"}],
            "metadata": {"corpora": [{"version_id": "unrelated"}]},
            "command": {"supervision_training_view_id": "not-input"},
            "output_dataset_version_id": "output",
        }
    )
    assert set(refs) == {
        ("supervision_training_view_version", "view"),
        ("supervision_corpus_version", "sft-corpus"),
        ("supervision_mixture_version", "sft-mix"),
        ("training_data_view_version", "pretrain-view"),
        ("tokenized_corpus_version", "corpus"),
    }
