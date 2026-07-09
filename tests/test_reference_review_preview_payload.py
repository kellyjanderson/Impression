from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from impression.devtools.reference_review.async_core.messages import (
    ReviewTaskKind,
    ReviewWorkbenchMessage,
)
from impression.devtools.reference_review.preview_payload import (
    PreviewPayload,
    PreviewPayloadDiagnostic,
    PreviewPayloadRequest,
)
from impression.devtools.reference_review.source_registry import ReviewSourceModelRecord


def _source_record(tmp_path: Path) -> ReviewSourceModelRecord:
    source = tmp_path / "model.py"
    artifact = tmp_path / "dirty.impress"
    source.write_text("def build():\n    return None\n")
    artifact.write_text('{"format": "impress"}\n')
    return ReviewSourceModelRecord(
        fixture_id="fixture/box",
        feature_name="Box",
        source_path=source,
        entrypoint="build",
        artifact_paths=(artifact,),
    )


def test_preview_payload_request_from_source_record_is_immutable_and_serializable(
    tmp_path: Path,
) -> None:
    record = _source_record(tmp_path)

    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-pane",
        request_id=4,
        generation=2,
        metadata={"reason": "fixture-selection"},
    )

    assert request.identity == ("preview-pane", 4, "fixture/box", 2)
    assert request.artifact_paths == record.artifact_paths
    assert request.to_json_dict() == {
        "owner": "preview-pane",
        "request_id": 4,
        "fixture_id": "fixture/box",
        "generation": 2,
        "source_path": record.source_path.as_posix(),
        "entrypoint": "build",
        "parameters": [],
        "artifact_paths": [record.artifact_paths[0].as_posix()],
        "metadata": {"reason": "fixture-selection"},
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.generation = 3  # type: ignore[misc]


def test_preview_payload_request_can_be_built_from_preview_workbench_message(
    tmp_path: Path,
) -> None:
    record = _source_record(tmp_path)
    message = ReviewWorkbenchMessage(
        owner="preview-pane",
        kind=ReviewTaskKind.PREVIEW_BUILD,
        request_id=8,
        fixture_id=record.fixture_id,
        payload={"style": "workbench"},
    )

    request = PreviewPayloadRequest.from_workbench_message(
        message,
        record,
        generation=6,
    )

    assert request.identity == ("preview-pane", 8, record.fixture_id, 6)
    assert request.metadata == {"style": "workbench"}


def test_preview_payload_request_refuses_wrong_message_kind(tmp_path: Path) -> None:
    record = _source_record(tmp_path)
    message = ReviewWorkbenchMessage(
        owner="preview-pane",
        kind=ReviewTaskKind.SOURCE_LOAD,
        request_id=1,
        fixture_id=record.fixture_id,
    )

    with pytest.raises(ValueError, match="PREVIEW_BUILD"):
        PreviewPayloadRequest.from_workbench_message(message, record, generation=1)


def test_preview_payload_success_and_failure_preserve_request_identity(
    tmp_path: Path,
) -> None:
    record = _source_record(tmp_path)
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-pane",
        request_id=12,
        generation=1,
    )
    payload_path = tmp_path / "payload.impress"
    success = PreviewPayload.success(
        request,
        payload_path=payload_path,
        metadata={"triangle_count": 14},
    )

    assert success.ok
    assert success.identity == request.identity
    assert success.to_json_dict()["payload_path"] == payload_path.as_posix()
    assert success.to_json_dict()["metadata"] == {"triangle_count": 14}

    diagnostic = PreviewPayloadDiagnostic(
        code="unsupported-source",
        message="source returned unsupported object",
        fixture_id=request.fixture_id,
        owner=request.owner,
        request_id=request.request_id,
        generation=request.generation,
    )
    failure = PreviewPayload.failure(request, diagnostic)

    assert not failure.ok
    assert failure.identity == request.identity
    assert failure.to_json_dict()["diagnostic"] == diagnostic.to_json_dict()


def test_preview_payload_diagnostic_sanitizes_exception_paths(tmp_path: Path) -> None:
    record = _source_record(tmp_path)
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-pane",
        request_id=5,
        generation=3,
    )

    diagnostic = PreviewPayloadDiagnostic.from_exception(
        request,
        RuntimeError(f"failed under {tmp_path}/private/model.py"),
        cwd=tmp_path,
    )

    assert diagnostic.code == "preview-payload-error"
    assert str(tmp_path) not in diagnostic.message
    assert "<workspace>" in diagnostic.message


def test_preview_payload_module_has_no_ui_or_renderer_import_boundary() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src/impression/devtools/reference_review/preview_payload.py"
    )
    source = module_path.read_text()

    assert "PySide6" not in source
    assert "pyvista" not in source
    assert "impression.devtools.reference_review.ui" not in source
