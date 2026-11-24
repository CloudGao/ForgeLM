from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from llm_platform_core.conversion import (
    CONVERSION_ENGINE_VERSION,
    validate_conversion_operators,
)

from ..project_storage import project_data_asset_root, scoped_runtime_path
from .data_import import (
    PARQUET_SUFFIXES,
    TEXT_SUFFIXES,
    AnalysisInterrupted,
    DataImportAdapter,
)

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
GetDesiredState = Callable[[], str]

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".epub"}
ZIP_FILE_LIMIT = 20_000
ZIP_UNCOMPRESSED_LIMIT = 2 * 1024 * 1024 * 1024
ZIP_COMPRESSION_RATIO_LIMIT = 250


class _HTMLTextExtractor(HTMLParser):
    _SKIPPED = {"script", "style", "svg", "nav"}
    _BLOCKS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in self._SKIPPED:
            self.skip_depth += 1
        elif not self.skip_depth and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._SKIPPED:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif not self.skip_depth and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _safe_zip(path: Path) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(path)
    entries = archive.infolist()
    if len(entries) > ZIP_FILE_LIMIT:
        archive.close()
        raise ValueError(f"archive contains more than {ZIP_FILE_LIMIT} entries")
    total = 0
    for entry in entries:
        logical = PurePosixPath(entry.filename.replace("\\", "/"))
        if logical.is_absolute() or ".." in logical.parts:
            archive.close()
            raise ValueError(f"archive contains an unsafe path: {entry.filename}")
        if entry.flag_bits & 0x1:
            archive.close()
            raise ValueError(f"archive contains an encrypted entry: {entry.filename}")
        total += int(entry.file_size)
        if total > ZIP_UNCOMPRESSED_LIMIT:
            archive.close()
            raise ValueError("archive uncompressed size exceeds 2 GiB")
        if (
            entry.file_size > 1024 * 1024
            and entry.compress_size > 0
            and entry.file_size / entry.compress_size > ZIP_COMPRESSION_RATIO_LIMIT
        ):
            archive.close()
            raise ValueError(f"archive entry compression ratio is unsafe: {entry.filename}")
    return archive


def detect_document_format(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
    except OSError:
        return "unknown"
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if zipfile.is_zipfile(path):
        try:
            with _safe_zip(path) as archive:
                names = set(archive.namelist())
                if "word/document.xml" in names and "[Content_Types].xml" in names:
                    return "docx"
                if "META-INF/container.xml" in names:
                    try:
                        mimetype = archive.read("mimetype").decode("ascii", errors="ignore").strip()
                    except KeyError:
                        mimetype = ""
                    if mimetype == "application/epub+zip" or any(
                        name.lower().endswith(".opf") for name in names
                    ):
                        return "epub"
        except (OSError, ValueError, zipfile.BadZipFile):
            return "unknown"
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    return "unknown"


def _element_text(element: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        local = child.tag.rsplit("}", 1)[-1]
        if local == "t" and child.text:
            parts.append(child.text)
        elif local == "tab":
            parts.append("\t")
        elif local in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _extract_docx(path: Path, *, include_tables: bool) -> Iterator[dict[str, Any]]:
    with _safe_zip(path) as archive:
        try:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        except KeyError as exc:
            raise ValueError("DOCX is missing word/document.xml") from exc
        except ElementTree.ParseError as exc:
            raise ValueError("DOCX document.xml is malformed") from exc
    body = next(
        (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "body"), root
    )
    index = 0
    for child in list(body):
        local = child.tag.rsplit("}", 1)[-1]
        if local == "p":
            text = _element_text(child)
            if text:
                yield {"text": text, "metadata": {"kind": "paragraph", "paragraph": index + 1}}
                index += 1
        elif local == "tbl" and include_tables:
            for row_index, row in enumerate(
                (item for item in child.iter() if item.tag.rsplit("}", 1)[-1] == "tr"),
                start=1,
            ):
                cells = [_element_text(cell) for cell in row if cell.tag.rsplit("}", 1)[-1] == "tc"]
                text = "\t".join(cell for cell in cells if cell)
                if text:
                    yield {
                        "text": text,
                        "metadata": {"kind": "table_row", "row": row_index},
                    }


def _safe_epub_member(base: str, href: str) -> str:
    member = posixpath.normpath(posixpath.join(posixpath.dirname(base), href.split("#", 1)[0]))
    logical = PurePosixPath(member)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError(f"EPUB manifest contains an unsafe path: {href}")
    return logical.as_posix()


def _extract_epub(path: Path) -> Iterator[dict[str, Any]]:
    with _safe_zip(path) as archive:
        names = set(archive.namelist())
        if "META-INF/encryption.xml" in names:
            try:
                encryption = ElementTree.fromstring(archive.read("META-INF/encryption.xml"))
            except ElementTree.ParseError as exc:
                raise ValueError("EPUB encryption metadata is malformed") from exc
            if any(
                element.tag.rsplit("}", 1)[-1] == "EncryptedData" for element in encryption.iter()
            ):
                raise ValueError("EPUB contains encrypted or DRM-protected content")
        try:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        except KeyError as exc:
            raise ValueError("EPUB is missing META-INF/container.xml") from exc
        except ElementTree.ParseError as exc:
            raise ValueError("EPUB container.xml is malformed") from exc
        rootfile = next(
            (
                str(element.attrib.get("full-path") or "")
                for element in container.iter()
                if element.tag.rsplit("}", 1)[-1] == "rootfile" and element.attrib.get("full-path")
            ),
            "",
        )
        if not rootfile or rootfile not in names:
            raise ValueError("EPUB package document is missing")
        try:
            package = ElementTree.fromstring(archive.read(rootfile))
        except ElementTree.ParseError as exc:
            raise ValueError("EPUB package document is malformed") from exc
        manifest: dict[str, tuple[str, str]] = {}
        for element in package.iter():
            if element.tag.rsplit("}", 1)[-1] != "item":
                continue
            item_id = str(element.attrib.get("id") or "")
            href = str(element.attrib.get("href") or "")
            media_type = str(element.attrib.get("media-type") or "")
            if item_id and href:
                manifest[item_id] = (_safe_epub_member(rootfile, href), media_type)
        spine = [
            str(element.attrib.get("idref") or "")
            for element in package.iter()
            if element.tag.rsplit("}", 1)[-1] == "itemref" and element.attrib.get("idref")
        ]
        if not spine:
            spine = [
                item_id
                for item_id, (_member, media_type) in manifest.items()
                if media_type in {"application/xhtml+xml", "text/html"}
            ]
        for chapter_index, item_id in enumerate(spine, start=1):
            member, media_type = manifest.get(item_id, ("", ""))
            if (
                not member
                or member not in names
                or media_type
                not in {
                    "application/xhtml+xml",
                    "text/html",
                }
            ):
                continue
            parser = _HTMLTextExtractor()
            parser.feed(archive.read(member).decode("utf-8", errors="replace"))
            yield {
                "text": parser.text(),
                "chapter": chapter_index,
                "metadata": {"href": member, "manifest_id": item_id},
            }


def _extract_pdf(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF conversion requires pypdf on this Worker") from exc
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is encrypted and cannot be opened") from exc
        if not unlocked:
            raise ValueError("PDF is encrypted and cannot be opened")
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield {"text": text, "page": page_number, "metadata": {"page": page_number}}


def _normalize_whitespace(text: str, max_blank_lines: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.split("\n")]
    return re.sub(
        rf"\n{{{max_blank_lines + 2},}}", "\n" * (max_blank_lines + 1), "\n".join(lines)
    ).strip()


def _apply_record_operators(
    record: dict[str, Any], operators: list[dict[str, dict[str, Any]]]
) -> Iterator[dict[str, Any]]:
    records = [record]
    for operator in operators:
        name, parameters = next(iter(operator.items()))
        if name == "normalize_whitespace":
            for item in records:
                item["text"] = _normalize_whitespace(
                    str(item.get("text") or ""), int(parameters["max_blank_lines"])
                )
        elif name == "filter_empty":
            minimum = int(parameters["min_characters"])
            records = [
                item for item in records if len(str(item.get("text") or "").strip()) >= minimum
            ]
        elif name == "chunk_text":
            maximum = int(parameters["max_characters"])
            overlap = int(parameters["overlap_characters"])
            chunked: list[dict[str, Any]] = []
            step = maximum - overlap
            for item in records:
                text = str(item.get("text") or "")
                if len(text) <= maximum:
                    chunked.append(item)
                    continue
                for chunk_index, start in enumerate(range(0, len(text), step)):
                    piece = text[start : start + maximum]
                    if not piece:
                        break
                    metadata = {**(item.get("metadata") or {})}
                    metadata.update(
                        {
                            "chunk_index": chunk_index,
                            "character_start": start,
                            "character_end": start + len(piece),
                        }
                    )
                    chunked.append({**item, "text": piece, "metadata": metadata})
                    if start + maximum >= len(text):
                        break
            records = chunked
    yield from records


def _file_hash(path: Path, get_desired_state: GetDesiredState | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if get_desired_state and get_desired_state() in {"paused", "cancelled"}:
                raise AnalysisInterrupted(get_desired_state())
            digest.update(chunk)
    return digest.hexdigest()


def _output_path(root: Path, relative: PurePosixPath) -> Path:
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("conversion output escapes its root") from exc
    return target


def convert_dataset(
    input_root: Path,
    output_root: Path,
    operators: list[dict[str, Any]],
    *,
    get_desired_state: GetDesiredState | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    normalized_operators = validate_conversion_operators(operators)
    names = {next(iter(operator)) for operator in normalized_operators}
    extractor_parameters = {
        next(iter(operator)): next(iter(operator.values())) for operator in normalized_operators
    }
    post_operators = [
        operator
        for operator in normalized_operators
        if next(iter(operator)) in {"normalize_whitespace", "filter_empty", "chunk_text"}
    ]
    input_base = input_root.parent if input_root.is_file() else input_root
    files = (
        [input_root]
        if input_root.is_file()
        else sorted(path for path in input_root.rglob("*") if path.is_file())
    )
    output_root.mkdir(parents=True, exist_ok=True)
    format_counts: Counter[str] = Counter()
    file_results: list[dict[str, Any]] = []
    converted_files = passthrough_files = failed_files = unsupported_files = output_records = 0

    for file_index, path in enumerate(files):
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            raise AnalysisInterrupted(desired)
        relative = PurePosixPath(path.relative_to(input_base).as_posix())
        detected = detect_document_format(path)
        format_counts[detected] += 1
        digest = _file_hash(path, get_desired_state)
        result: dict[str, Any] = {
            "path": relative.as_posix(),
            "format": detected,
            "source_sha256": digest,
            "size_bytes": path.stat().st_size,
            "status": "unsupported",
            "record_count": 0,
            "warnings": [],
        }
        if detected in {"text", "parquet"}:
            target = _output_path(output_root / "passthrough", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            result["status"] = "passthrough"
            passthrough_files += 1
        elif detected in {"pdf", "docx", "epub"} and f"extract_{detected}" in names:
            target = _output_path(
                output_root / "documents", PurePosixPath(f"{relative.as_posix()}.jsonl")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            try:
                if detected == "pdf":
                    extracted: Iterable[dict[str, Any]] = _extract_pdf(path)
                elif detected == "docx":
                    extracted = _extract_docx(
                        path,
                        include_tables=bool(
                            extractor_parameters.get("extract_docx", {}).get("include_tables", True)
                        ),
                    )
                else:
                    extracted = _extract_epub(path)
                record_index = 0
                with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                    for raw_record in extracted:
                        for processed in _apply_record_operators(raw_record, post_operators):
                            desired = get_desired_state() if get_desired_state else "running"
                            if desired in {"paused", "cancelled"}:
                                raise AnalysisInterrupted(desired)
                            record = {
                                **processed,
                                "source_file": relative.as_posix(),
                                "source_sha256": digest,
                                "source_type": detected,
                                "record_index": record_index,
                                "extractor": f"builtin.extract-{detected}",
                                "extractor_version": "1",
                                "warnings": list(processed.get("warnings") or []),
                            }
                            stream.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            record_index += 1
                if record_index:
                    os.replace(temporary, target)
                    result["status"] = "converted"
                    result["record_count"] = record_index
                    output_records += record_index
                    converted_files += 1
                else:
                    temporary.unlink(missing_ok=True)
                    result["status"] = "empty"
                    result["warnings"].append(
                        "未提取到正文；扫描 PDF 需要后续 OCR 能力"
                        if detected == "pdf"
                        else "未提取到正文"
                    )
            except AnalysisInterrupted:
                temporary.unlink(missing_ok=True)
                raise
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                failed_files += 1
        else:
            unsupported_files += 1
        file_results.append(result)
        if progress_callback:
            progress_callback(
                {
                    "progress": (file_index + 1) / max(len(files), 1),
                    "file_count_completed": file_index + 1,
                    "file_count_total": len(files),
                    "converted_records": output_records,
                }
            )

    return {
        "engine": "builtin.data-conversion",
        "engine_version": CONVERSION_ENGINE_VERSION,
        "operators": normalized_operators,
        "input_file_count": len(files),
        "format_counts": dict(sorted(format_counts.items())),
        "converted_file_count": converted_files,
        "passthrough_file_count": passthrough_files,
        "failed_file_count": failed_files,
        "unsupported_file_count": unsupported_files,
        "output_record_count": output_records,
        "files": file_results[:2_000],
        "file_results_truncated": len(file_results) > 2_000,
    }


class DataConversionAdapter:
    """Convert a frozen dataset snapshot into a normalized text snapshot."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime_context = task.get("runtime_context") or {}
        run_dir = Path(runtime_context.get("run_dir") or ".").resolve()
        attempt_dir = Path(runtime_context.get("attempt_dir") or run_dir).resolve()
        project_workspace = str(runtime_context.get("project_workspace") or "")
        worker_id = str(runtime_context.get("worker_id") or "") or None
        conversion_run_id = str(payload.get("conversion_run_id") or "").strip()
        output_version_id = str(payload.get("output_dataset_version_id") or "").strip()
        data_input = payload.get("data_conversion_input") or {}
        recipe = payload.get("recipe") or {}
        input_path = Path(str(data_input.get("runtime_path") or "")).resolve()
        operators = validate_conversion_operators(recipe.get("operators") or [])
        if not conversion_run_id or not output_version_id:
            raise ValueError("Data conversion task is missing its frozen run identity")
        if not input_path.exists():
            raise FileNotFoundError("Frozen data conversion input was not materialized")

        asset_root = scoped_runtime_path(
            project_data_asset_root(project_workspace), "data-conversion"
        )
        asset_root.mkdir(parents=True, exist_ok=True)
        output_root = scoped_runtime_path(asset_root, conversion_run_id)
        staging_root = scoped_runtime_path(
            asset_root, f".{conversion_run_id}-{attempt_dir.name}.staging"
        )
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        await emit("progress", {"progress": 0.02, "phase": "detecting_formats"})
        loop = asyncio.get_running_loop()

        def report_conversion(progress: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **progress,
                        "progress": 0.02 + 0.62 * float(progress.get("progress") or 0),
                        "phase": "converting_documents",
                    },
                ),
                loop,
            )
            future.result()

        try:
            conversion = await asyncio.to_thread(
                convert_dataset,
                input_path,
                staging_root,
                operators,
                get_desired_state=get_desired_state,
                progress_callback=report_conversion,
            )
            if output_root.exists():
                shutil.rmtree(output_root)
            staging_root.replace(output_root)
        except AnalysisInterrupted as exc:
            shutil.rmtree(staging_root, ignore_errors=True)
            await emit("progress", {"progress": 0.02, "phase": exc.state})
            return {"controlled_state": exc.state}
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

        record_manifest_path = attempt_dir / "dataset-records.jsonl.gz"
        await emit("progress", {"progress": 0.66, "phase": "analyzing_converted_data"})

        def report_analysis(progress: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **progress,
                        "progress": 0.66 + 0.3 * float(progress.get("progress") or 0),
                        "phase": "analyzing_converted_data",
                    },
                ),
                loop,
            )
            future.result()

        try:
            analysis = await asyncio.to_thread(
                DataImportAdapter._analyze,
                output_root,
                "conversion",
                str(data_input.get("dataset_version_id") or ""),
                output_version_id,
                record_manifest_path,
                {},
                report_analysis,
                get_desired_state,
            )
        except AnalysisInterrupted as exc:
            record_manifest_path.unlink(missing_ok=True)
            await emit("progress", {"progress": 0.66, "phase": exc.state})
            return {"controlled_state": exc.state}
        files = analysis.get("source_snapshot", {}).get("files") or []
        manifest_hash = DataImportAdapter._manifest_hash(files)
        conversion_report_path = attempt_dir / "data-conversion-report.json"
        conversion_report_path.write_text(
            json.dumps(conversion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        analysis["source_snapshot"] = {
            **(analysis.get("source_snapshot") or {}),
            "storage_type": "external",
            "target_dir": str(output_root),
            "conversion": conversion,
            "parent_dataset_version_id": data_input.get("dataset_version_id"),
            "input_artifact_id": data_input.get("artifact_id"),
            "input_artifact_hash": data_input.get("artifact_hash"),
        }
        await emit(
            "progress",
            {
                "progress": 0.98,
                "phase": "registering_converted_dataset",
                "document_count": analysis.get("document_count", 0),
            },
        )
        return {
            **analysis,
            "engine_version": CONVERSION_ENGINE_VERSION,
            "conversion_summary": conversion,
            "outputs": [
                {
                    "ref": "dataset",
                    "kind": "dataset_snapshot",
                    "storage_type": "external",
                    "uri": output_root.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": int(analysis.get("raw_size_bytes") or 0),
                    "media_type": "application/vnd.llm-platform.dataset-directory",
                    "availability_state": "available",
                    "metadata": {
                        "conversion_run_id": conversion_run_id,
                        "input_artifact_id": data_input.get("artifact_id"),
                        "file_count": len(files),
                    },
                    "worker_id": worker_id,
                },
                {
                    "ref": "record_manifest",
                    "path": str(record_manifest_path),
                    "kind": "dataset_record_manifest",
                    "media_type": "application/gzip",
                },
                {
                    "ref": "conversion_report",
                    "path": str(conversion_report_path),
                    "kind": "data_conversion_report",
                    "media_type": "application/json",
                },
            ],
        }
