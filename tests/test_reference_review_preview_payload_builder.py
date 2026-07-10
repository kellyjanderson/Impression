from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from impression.mesh import Mesh
import impression.devtools.reference_review.preview_payload_builder as preview_payload_builder
from impression.devtools.reference_review import (
    EntrypointParameterRecord,
    LoadedPreviewDataset,
    PREVIEW_PAYLOAD_FORMAT,
    PreviewDatasetBuildResult,
    PreviewPayload,
    PreviewPayloadDiagnostic,
    PreviewPayloadRequest,
    ReviewSourceModelRecord,
    build_serialized_preview_payload,
    build_preview_dataset,
    load_source_records_from_file,
    load_preview_source,
    tessellate_preview_source,
    write_preview_payload_file,
)


def _request_from_record(
    record: ReviewSourceModelRecord,
    *,
    request_id: int = 1,
) -> PreviewPayloadRequest:
    return PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-worker",
        request_id=request_id,
        generation=1,
    )


def test_preview_source_loader_invokes_fixture_entrypoint_with_parameters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parameterized.py"
    source.write_text(
        "from impression.modeling import make_box\n"
        "def make(width):\n"
        "    return make_box(size=(width, 1, 1), backend='surface')\n"
    )
    record = ReviewSourceModelRecord(
        fixture_id="fixture/parameterized",
        feature_name="Parameterized",
        source_path=source,
        entrypoint="make",
        parameters=(EntrypointParameterRecord("width", 2.5),),
    )
    request = _request_from_record(record)

    loaded = load_preview_source(request)
    dataset = tessellate_preview_source(request, loaded)

    assert loaded.__class__.__name__ == "SurfaceBody"
    assert dataset.request is request
    assert dataset.dataset_count == 1
    assert dataset.datasets[0].vertices.shape[0] > 0


def test_preview_source_loader_serializes_process_import_state(tmp_path: Path) -> None:
    source = tmp_path / "simple.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord("fixture/import-lock", "Import Lock", source)
    request = _request_from_record(record)
    acquired = []

    class RecordingLock:
        def __enter__(self):
            acquired.append("enter")

        def __exit__(self, exc_type, exc, tb):
            acquired.append("exit")

    original_lock = preview_payload_builder._PREVIEW_SOURCE_IMPORT_LOCK
    preview_payload_builder._PREVIEW_SOURCE_IMPORT_LOCK = RecordingLock()  # type: ignore[assignment]
    try:
        assert load_preview_source(request) is None
    finally:
        preview_payload_builder._PREVIEW_SOURCE_IMPORT_LOCK = original_lock

    assert acquired == ["enter", "exit"]


def test_preview_dataset_builder_tessellates_real_dirty_impress_source_fixture(
    project_root: Path,
) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    summary = load_source_records_from_file(fixture_file)
    record = summary.valid_items[0].record
    request = _request_from_record(record, request_id=12)

    result = build_preview_dataset(request, cwd=project_root)

    assert result.ok
    assert result.dataset is not None
    assert result.dataset.request.identity == request.identity
    assert result.dataset.dataset_count == 1
    assert result.dataset.datasets[0].faces.shape[0] > 0
    assert result.diagnostic is None


def test_preview_dataset_builder_adds_workspace_root_for_fixture_imports(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support"
    support.mkdir()
    (support / "__init__.py").write_text("")
    (support / "fixture_helpers.py").write_text(
        "from impression.modeling import make_box\n"
        "def build_box():\n"
        "    return make_box(size=(1, 1, 1), backend='surface')\n"
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = fixtures / "source.py"
    source.write_text(
        "from support.fixture_helpers import build_box\n"
        "def build():\n"
        "    return build_box()\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'preview-fixture-test'\n")
    root_text = tmp_path.resolve().as_posix()
    original_path = list(sys.path)
    sys.path[:] = [item for item in sys.path if item != root_text]
    record = ReviewSourceModelRecord(
        fixture_id="fixture/workspace-import",
        feature_name="Workspace Import",
        source_path=source,
    )
    request = _request_from_record(record, request_id=13)

    try:
        result = build_preview_dataset(request, cwd=tmp_path)
    finally:
        restored_path = list(sys.path)
        sys.path[:] = original_path

    assert restored_path == [item for item in original_path if item != root_text]
    assert result.ok
    assert result.dataset is not None
    assert result.dataset.dataset_count == 1
    assert result.diagnostic is None


def test_preview_dataset_builder_returns_sanitized_diagnostic_for_bad_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.py"
    source.write_text("def build():\n    raise RuntimeError('private path: %s')\n" % tmp_path)
    record = ReviewSourceModelRecord(
        fixture_id="fixture/bad",
        feature_name="Bad",
        source_path=source,
    )
    request = _request_from_record(record)

    result = build_preview_dataset(request, cwd=tmp_path)

    assert not result.ok
    assert result.diagnostic is not None
    assert result.diagnostic.code == "preview-source-load-failed"
    assert str(tmp_path) not in result.diagnostic.message
    assert "<workspace>" in result.diagnostic.message


def test_preview_payload_writer_creates_file_backed_serialized_payload(
    project_root: Path,
    tmp_path: Path,
) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    record = load_source_records_from_file(fixture_file).valid_items[0].record
    request = _request_from_record(record, request_id=15)
    dataset_result = build_preview_dataset(request, cwd=project_root)
    assert dataset_result.dataset is not None

    payload = write_preview_payload_file(dataset_result.dataset, payload_dir=tmp_path)
    payload_json = json.loads(payload.payload_path.read_text())  # type: ignore[union-attr]

    assert payload.ok
    assert payload.payload_kind == PREVIEW_PAYLOAD_FORMAT
    assert payload.file_metadata is not None
    assert payload.file_metadata.path == payload.payload_path
    assert payload.file_metadata.byte_count > 0
    assert payload.file_metadata.dataset_count == 1
    assert payload_json["format"] == PREVIEW_PAYLOAD_FORMAT
    assert payload_json["request"]["fixture_id"] == request.fixture_id
    assert payload_json["datasets"][0]["kind"] == "mesh"
    assert payload_json["datasets"][0]["vertices"]
    assert payload_json["datasets"][0]["faces"]


def test_serialized_preview_payload_reports_serialization_diagnostic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("from impression.modeling import make_box\n\ndef build():\n    return make_box(backend='surface')\n")
    blocked_dir = tmp_path / "not-a-directory"
    blocked_dir.write_text("already a file")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/blocked",
        feature_name="Blocked",
        source_path=source,
    )
    request = _request_from_record(record)

    payload = build_serialized_preview_payload(
        request,
        payload_dir=blocked_dir,
        cwd=tmp_path,
    )

    assert not payload.ok
    assert payload.diagnostic is not None
    assert payload.diagnostic.code == "preview-payload-serialization-failed"
    assert str(tmp_path) not in payload.diagnostic.message


def test_serialized_preview_payload_orchestrates_injected_builder_and_writer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    request = _request_from_record(
        ReviewSourceModelRecord(
            fixture_id="fixture/orchestrated",
            feature_name="Orchestrated",
            source_path=source,
        )
    )
    loaded = LoadedPreviewDataset(
        request=request,
        datasets=(
            Mesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                faces=np.asarray(((0, 1, 2),)),
            ),
        ),
        source_type="SurfaceBody",
    )
    calls: list[str] = []

    def builder(candidate_request, *, cwd=None, home=None):
        calls.append(f"builder:{candidate_request.fixture_id}:{cwd == tmp_path}:{home is None}")
        return PreviewDatasetBuildResult(candidate_request, dataset=loaded)

    def writer(candidate_dataset, *, payload_dir=None):
        calls.append(f"writer:{candidate_dataset.dataset_count}:{payload_dir == tmp_path}")
        return PreviewPayload.success(
            candidate_dataset.request,
            payload_path=tmp_path / "payload.json",
            payload_kind=PREVIEW_PAYLOAD_FORMAT,
        )

    payload = build_serialized_preview_payload(
        request,
        payload_dir=tmp_path,
        cwd=tmp_path,
        dataset_builder=builder,
        payload_writer=writer,
    )

    assert payload.ok
    assert calls == [
        "builder:fixture/orchestrated:True:True",
        "writer:1:True",
    ]


def test_serialized_preview_payload_skips_writer_after_builder_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    request = _request_from_record(
        ReviewSourceModelRecord(
            fixture_id="fixture/failure",
            feature_name="Failure",
            source_path=source,
        )
    )
    diagnostic = PreviewPayloadDiagnostic(
        code="preview-source-load-failed",
        message="failed",
        fixture_id=request.fixture_id,
        owner=request.owner,
        request_id=request.request_id,
        generation=request.generation,
    )
    writer_called = False

    def builder(candidate_request, *, cwd=None, home=None):
        return PreviewDatasetBuildResult(candidate_request, diagnostic=diagnostic)

    def writer(candidate_dataset, *, payload_dir=None):
        nonlocal writer_called
        writer_called = True
        return PreviewPayload.success(candidate_dataset.request)

    payload = build_serialized_preview_payload(
        request,
        dataset_builder=builder,
        payload_writer=writer,
    )

    assert not payload.ok
    assert payload.diagnostic == diagnostic
    assert not writer_called


def test_preview_payload_builder_has_no_ui_or_renderer_import_boundary() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src/impression/devtools/reference_review/preview_payload_builder.py"
    )
    source = module_path.read_text()

    assert "PySide6" not in source
    assert "pyvista" not in source
    assert "impression.devtools.reference_review.ui" not in source
