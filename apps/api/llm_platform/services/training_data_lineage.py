"""Remove redundant data-input shortcuts from display graphs, never execution facts."""

DATA_VERSION_TYPES = {
    "dataset_version",
    "mixture_version",
    "supervision_dataset_version",
    "supervision_mixture_version",
    "eval_dataset_version",
    "training_data_view_version",
    "tokenized_corpus_version",
    "tokenized_dataset_version",
    "supervision_corpus_version",
    "supervision_training_view_version",
}


def layered_data_edges(nodes, edges):
    """Keep nearest data inputs when the displayed graph proves a derivation path.

    Only alternate paths through data versions and their producers count. A path
    through training/Checkpoint, or a node-to-node control dependency, cannot hide
    a training input. Missing/truncated paths never justify removing an edge.
    """
    assets = {node["id"] for node in nodes if node.get("kind") == "asset"}
    data = {node["id"] for node in nodes if node.get("asset_type") in DATA_VERSION_TYPES}
    inputs, producers = {}, {}
    for edge in edges.values():
        source, target = edge["source"], edge["target"]
        if source in data and target not in assets:
            inputs.setdefault(target, set()).add(source)
        elif target in data and source not in assets:
            producers.setdefault(target, set()).add(source)

    redundant = set()
    for target, sources in inputs.items():
        if len(sources) < 2:
            continue
        ancestors = {}
        for source in sources:
            seen, pending = set(), [source]
            while pending:
                current = pending.pop()
                for producer in producers.get(current, ()):
                    if producer == target:
                        continue
                    for predecessor in inputs.get(producer, ()):
                        if predecessor not in seen:
                            seen.add(predecessor)
                            pending.append(predecessor)
            ancestors[source] = seen
        for source in sources:
            if any(
                source in ancestors[other] and other not in ancestors[source]
                for other in sources - {source}
            ):
                redundant.add((source, target))
    return {
        key: edge
        for key, edge in edges.items()
        if (edge["source"], edge["target"]) not in redundant
    }
