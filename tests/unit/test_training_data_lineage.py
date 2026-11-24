import pytest
from llm_platform.services.training_data_lineage import layered_data_edges


def bindings(*pairs):
    return {f"{source}:{target}": {"source": source, "target": target} for source, target in pairs}


def asset(identity, kind="dataset_version"):
    return {"id": identity, "kind": "asset", "asset_type": kind}


def test_data_shortcuts_are_removed_without_changing_checkpoint_backbone_or_control_dependencies():
    nodes = [
        asset("raw"),
        asset("mixed", "mixture_version"),
        asset("view", "training_data_view_version"),
        asset("checkpoint", "checkpoint"),
        asset("output", "checkpoint"),
        asset("model", "model_version"),
    ]
    edges = bindings(
        ("raw", "mix"),
        ("mix", "mixed"),
        ("mixed", "prepare"),
        ("prepare", "view"),
        ("raw", "prepare"),
        ("raw", "train"),
        ("mixed", "train"),
        ("view", "train"),
        ("view", "previous-train"),
        ("previous-train", "checkpoint"),
        ("checkpoint", "train"),
        ("train", "output"),
        ("train", "model"),
        ("output", "model"),
        ("prepare", "train"),
    )
    kept = layered_data_edges(nodes, edges)
    assert set(edges) - set(kept) == {"raw:prepare", "raw:train", "mixed:train"}
    assert len(edges) == 15  # The input graph and persisted facts are not edited.


@pytest.mark.parametrize("bridge", [[], [("prepare", "train")]])
def test_missing_data_derivation_or_control_dependency_does_not_hide_raw_input(bridge):
    nodes = [asset("raw"), asset("view", "training_data_view_version")]
    edges = bindings(("raw", "prepare"), ("raw", "train"), ("view", "train"), *bridge)
    assert layered_data_edges(nodes, edges) == edges


def test_cyclic_data_facts_do_not_remove_all_inputs():
    nodes = [asset("first"), asset("second")]
    edges = bindings(
        ("first", "p1"),
        ("p1", "second"),
        ("second", "p2"),
        ("p2", "first"),
        ("first", "train"),
        ("second", "train"),
    )
    assert layered_data_edges(nodes, edges) == edges
