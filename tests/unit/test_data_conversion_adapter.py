from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from llm_platform_core.conversion import (
    DEFAULT_CONVERSION_OPERATORS,
    validate_conversion_operators,
)
from llm_platform_worker.adapters.data_conversion import (
    DataConversionAdapter,
    convert_dataset,
    detect_document_format,
)
from llm_platform_worker.adapters.data_import import DataImportAdapter


def _write_docx(path: Path) -> None:
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>第一段</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>单元格 A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>单元格 B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)


def _write_epub(path: Path) -> None:
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    package = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="chapter"/></spine>
</package>"""
    chapter = (
        """<html><body><h1>第一章</h1><p>EPUB 正文。</p><script>ignored()</script></body></html>"""
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/package.opf", package)
        archive.writestr("OPS/chapter.xhtml", chapter)


def test_conversion_operator_validation_normalizes_and_checks_order() -> None:
    normalized = validate_conversion_operators(DEFAULT_CONVERSION_OPERATORS)
    assert normalized[2] == {"extract_docx": {"include_tables": True}}
    assert normalized[-1] == {"filter_empty": {"min_characters": 1}}

    with pytest.raises(ValueError, match="detect_format must be the first"):
        validate_conversion_operators([{"extract_pdf": {}}, {"detect_format": {}}])
    with pytest.raises(ValueError, match="smaller than"):
        validate_conversion_operators(
            [
                {"detect_format": {}},
                {"extract_pdf": {}},
                {"chunk_text": {"max_characters": 256, "overlap_characters": 256}},
            ]
        )


def test_raw_document_import_does_not_implicitly_convert(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "manual.pdf").write_bytes(b"%PDF-1.4\nraw-document")

    analysis = DataImportAdapter._analyze(
        source,
        "local",
        str(source),
        "raw-version-1",
        tmp_path / "records.jsonl.gz",
    )

    assert analysis["document_count"] == 0
    assert analysis["train_ready"] is False
    assert analysis["quality_report"]["unsupported_files"] == ["manual.pdf"]


def test_docx_and_epub_are_detected_by_container_and_converted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_docx(source / "manual.bin")
    _write_epub(source / "book.data")
    (source / "notes.txt").write_text("already text\n", encoding="utf-8")

    assert detect_document_format(source / "manual.bin") == "docx"
    assert detect_document_format(source / "book.data") == "epub"
    result = convert_dataset(source, output, DEFAULT_CONVERSION_OPERATORS)

    assert result["format_counts"] == {"docx": 1, "epub": 1, "text": 1}
    assert result["converted_file_count"] == 2
    assert result["passthrough_file_count"] == 1
    assert (output / "passthrough" / "notes.txt").read_text(encoding="utf-8") == "already text\n"
    docx_records = [
        json.loads(line)
        for line in (output / "documents" / "manual.bin.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    epub_records = [
        json.loads(line)
        for line in (output / "documents" / "book.data.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["text"] for record in docx_records] == ["第一段", "单元格 A\t单元格 B"]
    assert epub_records[0]["text"] == "第一章\n\nEPUB 正文。"
    assert epub_records[0]["chapter"] == 1
    assert all(record["source_sha256"] for record in [*docx_records, *epub_records])


def test_chunk_operator_preserves_source_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_docx(source / "long.docx")
    operators = [
        {"detect_format": {}},
        {"extract_docx": {"include_tables": False}},
        {"chunk_text": {"max_characters": 256, "overlap_characters": 16}},
    ]
    # Replace the short paragraph inside the fixture with a deterministic long one.
    long_text = "甲" * 600
    document = f"""<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{long_text}</w:t></w:r></w:p></w:body></w:document>"""
    with zipfile.ZipFile(source / "long.docx", "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)

    convert_dataset(source, output, operators)
    records = [
        json.loads(line)
        for line in (output / "documents" / "long.docx.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 3
    assert records[1]["metadata"]["character_start"] == 240
    assert records[0]["source_file"] == "long.docx"


def test_text_pdf_is_extracted_from_a_single_file_snapshot(tmp_path: Path) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    output = tmp_path / "output"
    pdf_path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    content = StreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (Hello PDF) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    result = convert_dataset(pdf_path, output, DEFAULT_CONVERSION_OPERATORS)

    assert detect_document_format(pdf_path) == "pdf"
    assert result["converted_file_count"] == 1
    record = json.loads(
        (output / "documents" / "paper.pdf.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "Hello PDF" in record["text"]
    assert record["page"] == 1


@pytest.mark.asyncio
async def test_conversion_adapter_creates_analyzed_snapshot_and_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "source"
    attempt = workspace / ".llm-platform" / "runs" / "task-1" / "attempt-1"
    source.mkdir()
    attempt.mkdir(parents=True)
    _write_docx(source / "manual.docx")
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await DataConversionAdapter().run(
        {
            "project_id": "project-1",
            "payload": {
                "conversion_run_id": "conversion-run-1",
                "input_dataset_version_id": "version-1",
                "output_dataset_version_id": "version-2",
                "data_conversion_input": {
                    "runtime_path": str(source),
                    "artifact_id": "artifact-1",
                    "artifact_hash": "a" * 64,
                    "dataset_version_id": "version-1",
                },
                "recipe": {"operators": DEFAULT_CONVERSION_OPERATORS},
            },
            "runtime_context": {
                "run_dir": str(attempt),
                "attempt_dir": str(attempt),
                "project_workspace": str(workspace),
                "worker_id": "worker-1",
            },
        },
        emit,
        lambda: "running",
    )

    assert result["document_count"] == 2
    assert result["train_ready"] is True
    assert result["conversion_summary"]["converted_file_count"] == 1
    outputs = {item["ref"]: item for item in result["outputs"]}
    assert outputs["dataset"]["storage_type"] == "external"
    assert outputs["dataset"]["worker_id"] == "worker-1"
    assert Path(outputs["record_manifest"]["path"]).is_file()
    assert Path(outputs["conversion_report"]["path"]).is_file()
    assert any(payload.get("phase") == "registering_converted_dataset" for _, payload in events)


def test_empty_and_encrypted_pdf_are_reported_without_fake_ocr(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    blank = PdfWriter()
    blank.add_blank_page(width=100, height=100)
    with (source / "scan.pdf").open("wb") as stream:
        blank.write(stream)
    encrypted = PdfWriter()
    encrypted.add_blank_page(width=100, height=100)
    encrypted.encrypt("secret")
    with (source / "locked.pdf").open("wb") as stream:
        encrypted.write(stream)

    result = convert_dataset(source, output, DEFAULT_CONVERSION_OPERATORS)

    files = {item["path"]: item for item in result["files"]}
    assert files["scan.pdf"]["status"] == "empty"
    assert "OCR" in files["scan.pdf"]["warnings"][0]
    assert files["locked.pdf"]["status"] == "failed"
    assert "encrypted" in files["locked.pdf"]["error"]
    assert result["output_record_count"] == 0


def test_unsafe_docx_container_is_not_opened(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.docx"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
        archive.writestr("../escape.txt", "blocked")

    assert detect_document_format(archive_path) == "unknown"
