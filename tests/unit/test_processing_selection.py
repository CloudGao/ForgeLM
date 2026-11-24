import hashlib

import pytest
from llm_platform_worker.processing_selection import selected_processing_files


def test_processing_selection_reads_original_files_and_rejects_drift(tmp_path):
    source = tmp_path / "selected.jsonl"
    source.write_bytes(b'{"text":"selected"}\n')
    (tmp_path / "excluded.jsonl").write_bytes(b'{"text":"excluded"}\n')
    selection = {"schema": "forgelm-processing-file-selection-v1", "files": [
        {"path": source.name, "size_bytes": source.stat().st_size,
         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}
    before = sorted(p.name for p in tmp_path.iterdir())
    assert selected_processing_files(tmp_path, selection) == [str(source)]
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    source.write_bytes(b'{"text":"tampered"}\n')
    with pytest.raises(ValueError, match="changed"):
        selected_processing_files(tmp_path, selection)


@pytest.mark.parametrize("path", ["../outside", "/outside", "C:/outside", "a\\b"])
def test_processing_selection_rejects_path_escape(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        selected_processing_files(tmp_path, {"schema": "forgelm-processing-file-selection-v1",
                                             "files": [{"path": path}]})


def test_processing_selection_honors_cancel(tmp_path):
    from llm_platform_worker.adapters.data_import import AnalysisInterrupted

    with pytest.raises(AnalysisInterrupted):
        selected_processing_files(tmp_path, {"schema": "forgelm-processing-file-selection-v1",
                                             "files": [{"path": "a"}]}, lambda: "cancelled")
