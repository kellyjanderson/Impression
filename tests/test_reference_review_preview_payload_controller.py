from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from threading import Event
from time import monotonic

from impression.devtools.reference_review import (
    PreviewPayload,
    PreviewPayloadDiagnostic,
    PreviewPayloadFailedEvent,
    PreviewPayloadProcessController,
    PreviewPayloadProcessResult,
    PreviewPayloadReadyEvent,
    PreviewPayloadRequest,
    ReviewSourceModelRecord,
)
from impression.devtools.reference_review.async_core import (
    DispatchResult,
    ReviewTaskKind,
    TaskDispatcher,
    ReviewWorkbenchMessage,
    WorkerResultEnvelope,
)


def test_preview_payload_process_controller_launches_and_tracks_active_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("from impression.modeling import make_box\n\ndef build():\n    return make_box(backend='surface')\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/controller",
        feature_name="Controller",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)

    dispatch = controller.launch(record)
    envelope = dispatch.future.result(timeout=2) if dispatch.future else None
    controller.close()

    assert dispatch.accepted
    assert controller.active_identity == ("preview-payload", 1, "fixture/controller", 1)
    assert envelope is not None
    assert envelope.ok
    assert isinstance(envelope.result, PreviewPayloadProcessResult)
    assert envelope.result.ok
    assert envelope.result.payload.payload_path is not None
    assert envelope.result.payload.payload_path.exists()


def test_preview_payload_process_controller_captures_sanitized_stdout_and_stderr(
    tmp_path: Path,
) -> None:
    source = tmp_path / "noisy.py"
    source.write_text(
        "import sys\n"
        "from impression.modeling import make_box\n"
        "def build():\n"
        "    print('stdout under %s')\n"
        "    print('stderr under %s', file=sys.stderr)\n"
        "    return make_box(backend='surface')\n" % (tmp_path, tmp_path)
    )
    record = ReviewSourceModelRecord(
        fixture_id="fixture/noisy",
        feature_name="Noisy",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)

    dispatch = controller.launch(record)
    envelope = dispatch.future.result(timeout=2) if dispatch.future else None
    controller.close()

    assert envelope is not None
    assert isinstance(envelope.result, PreviewPayloadProcessResult)
    assert str(tmp_path) not in envelope.result.stdout
    assert str(tmp_path) not in envelope.result.stderr
    assert "<workspace>" in envelope.result.stdout
    assert "<workspace>" in envelope.result.stderr


def test_preview_payload_process_controller_records_launch_rejection() -> None:
    request = ReviewWorkbenchMessage(
        owner="preview-payload",
        kind=ReviewTaskKind.PREVIEW_BUILD,
        request_id=1,
        fixture_id="fixture/rejected",
    )

    class RejectingDispatcher:
        def dispatch(self, _request, _worker):
            return DispatchResult(False, request, diagnostic="queue_full")

    controller = PreviewPayloadProcessController(dispatcher=RejectingDispatcher())  # type: ignore[arg-type]
    record = ReviewSourceModelRecord(
        fixture_id="fixture/rejected",
        feature_name="Rejected",
        source_path=Path("missing.py"),
    )

    dispatch = controller.launch(record)

    assert not dispatch.accepted
    assert controller.diagnostics[0].code == "preview-payload-launch-rejected"
    assert controller.diagnostics[0].message == "queue_full"


def test_preview_payload_controller_dispatcher_route_builds_payload(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("from impression.modeling import make_box\n\ndef build():\n    return make_box(backend='surface')\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/dispatcher",
        feature_name="Dispatcher",
        source_path=source,
    )
    dispatcher = TaskDispatcher(max_workers=1)
    controller = PreviewPayloadProcessController(
        dispatcher=dispatcher,
        owns_dispatcher=True,
        payload_dir=tmp_path,
        cwd=tmp_path,
    )

    dispatch = controller.launch(record)
    envelope = dispatch.future.result(timeout=2) if dispatch.future else None
    controller.close()

    assert envelope is not None
    assert envelope.ok
    assert isinstance(envelope.result, PreviewPayloadProcessResult)
    assert envelope.result.payload.payload_path is not None
    assert envelope.result.payload.payload_path.exists()


def test_preview_payload_process_launch_does_not_block_on_process_submit(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/nonblocking-launch",
        feature_name="Nonblocking",
        source_path=source,
    )
    entered_submit = Event()
    release_submit = Event()

    class BlockingProcessExecutor:
        def submit(self, _worker, message, *_args, **_kwargs):
            entered_submit.set()
            release_submit.wait(timeout=2)
            future: Future[WorkerResultEnvelope] = Future()
            future.set_result(
                WorkerResultEnvelope(
                    request=message,
                    ok=False,
                    error="controlled-test-submit",
                )
            )
            return future

        def shutdown(self, *, wait=True, cancel_futures=False):
            release_submit.set()

    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    assert controller._process_executor is not None
    controller._process_executor.shutdown(wait=False, cancel_futures=True)
    controller._process_executor = BlockingProcessExecutor()  # type: ignore[assignment]

    start = monotonic()
    dispatch = controller.launch(record)
    elapsed = monotonic() - start

    assert dispatch.accepted
    assert elapsed < 0.2
    assert entered_submit.wait(timeout=1)
    release_submit.set()
    assert dispatch.future is not None
    envelope = dispatch.future.result(timeout=2)
    controller.close()

    assert envelope.error == "controlled-test-submit"


def _payload_for_cleanup(tmp_path: Path, *, request_id: int = 1) -> PreviewPayload:
    source = tmp_path / f"model-{request_id}.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        fixture_id=f"fixture/cleanup-{request_id}",
        feature_name="Cleanup",
        source_path=source,
    )
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=request_id,
        generation=request_id,
    )
    payload_path = tmp_path / f"payload-{request_id}.json"
    payload_path.write_text("{}")
    return PreviewPayload.success(request, payload_path=payload_path)


def test_preview_payload_process_controller_cleans_completed_payload(
    tmp_path: Path,
) -> None:
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    payload = _payload_for_cleanup(tmp_path)
    controller.adopt_payload(payload)

    diagnostic = controller.cleanup_payload(payload, reason="completed")
    controller.close()

    assert diagnostic.code == "preview-payload-cleanup-deleted"
    assert diagnostic.deleted
    assert not payload.payload_path.exists()  # type: ignore[union-attr]


def test_preview_payload_process_controller_cleans_cancelled_payload(
    tmp_path: Path,
) -> None:
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    payload = _payload_for_cleanup(tmp_path, request_id=2)
    controller.adopt_payload(payload)

    diagnostic = controller.cleanup_cancelled(payload.identity)
    controller.close()

    assert diagnostic.code == "preview-payload-cleanup-deleted"
    assert diagnostic.reason == "cancelled"
    assert not payload.payload_path.exists()  # type: ignore[union-attr]


def test_preview_payload_process_controller_cleans_stale_but_not_current_payload(
    tmp_path: Path,
) -> None:
    current_source = tmp_path / "current.py"
    current_source.write_text("def build():\n    return None\n")
    current_record = ReviewSourceModelRecord(
        fixture_id="fixture/current",
        feature_name="Current",
        source_path=current_source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    controller.launch(current_record)
    stale_payload = _payload_for_cleanup(tmp_path, request_id=3)
    current_request = PreviewPayloadRequest.from_source_record(
        current_record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    current_payload_path = tmp_path / "current-payload.json"
    current_payload_path.write_text("{}")
    current_payload = PreviewPayload.success(current_request, payload_path=current_payload_path)
    controller.adopt_payload(stale_payload)
    controller.adopt_payload(current_payload)

    stale_diagnostic = controller.cleanup_stale_payload(stale_payload)
    current_diagnostic = controller.cleanup_stale_payload(current_payload)
    controller.close()

    assert stale_diagnostic.code == "preview-payload-cleanup-deleted"
    assert not stale_payload.payload_path.exists()  # type: ignore[union-attr]
    assert current_diagnostic.code == "preview-payload-cleanup-skipped-current"
    assert current_payload_path.exists()


def test_preview_payload_process_controller_cleanup_handles_missing_and_unowned_payloads(
    tmp_path: Path,
) -> None:
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    owned = _payload_for_cleanup(tmp_path, request_id=4)
    unowned = _payload_for_cleanup(tmp_path, request_id=5)
    controller.adopt_payload(owned)
    owned.payload_path.unlink()  # type: ignore[union-attr]

    missing = controller.cleanup_payload(owned, reason="completed")
    unowned_diagnostic = controller.cleanup_payload(unowned, reason="completed")
    controller.close()

    assert missing.code == "preview-payload-cleanup-missing"
    assert unowned_diagnostic.code == "preview-payload-cleanup-unowned"
    assert unowned.payload_path.exists()  # type: ignore[union-attr]


def test_preview_payload_process_controller_hands_off_current_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("from impression.modeling import make_box\n\ndef build():\n    return make_box(backend='surface')\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/current-handoff",
        feature_name="Current",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    dispatch = controller.launch(record)
    envelope = dispatch.future.result(timeout=2) if dispatch.future else None
    seen: list[PreviewPayloadReadyEvent] = []

    decision = controller.handle_completion(envelope, seen.append)
    controller.close()

    assert decision.accepted
    assert decision.reason == "current_payload_ready"
    assert len(seen) == 1
    assert seen[0].payload.identity == controller.active_identity
    assert controller.active_state is not None
    assert controller.active_state.identity == controller.active_identity


def test_preview_payload_process_controller_rejects_and_cleans_stale_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "current.py"
    source.write_text("def build():\n    return None\n")
    current_record = ReviewSourceModelRecord(
        fixture_id="fixture/current",
        feature_name="Current",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    controller.launch(current_record)
    stale_payload = _payload_for_cleanup(tmp_path, request_id=8)
    seen: list[PreviewPayloadReadyEvent] = []

    decision = controller.handle_process_result(
        PreviewPayloadProcessResult(stale_payload),
        seen.append,
    )
    controller.close()

    assert not decision.accepted
    assert decision.reason == "stale_payload_rejected"
    assert decision.cleanup is not None
    assert decision.cleanup.code == "preview-payload-cleanup-deleted"
    assert seen == []
    assert not stale_payload.payload_path.exists()  # type: ignore[union-attr]


def test_preview_payload_process_controller_hands_off_current_failure_diagnostic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/current-failure",
        feature_name="Current Failure",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    controller.launch(record)
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    diagnostic = PreviewPayloadDiagnostic.from_exception(
        request,
        RuntimeError(f"failed under {tmp_path}/private.py"),
        cwd=tmp_path,
    )
    failure_payload = PreviewPayload.failure(request, diagnostic)
    ready: list[PreviewPayloadReadyEvent] = []
    failures: list[PreviewPayloadFailedEvent] = []

    decision = controller.handle_process_result(
        PreviewPayloadProcessResult(failure_payload, stdout="out", stderr="err"),
        ready.append,
        failures.append,
    )
    controller.close()

    assert not decision.accepted
    assert decision.reason == "payload_failure"
    assert ready == []
    assert len(failures) == 1
    assert failures[0].diagnostic.code == "preview-payload-error"
    assert str(tmp_path) not in failures[0].diagnostic.message
    assert failures[0].stdout == "out"
    assert failures[0].stderr == "err"


def test_preview_payload_process_controller_ignores_stale_failure_diagnostic(
    tmp_path: Path,
) -> None:
    current_source = tmp_path / "current.py"
    current_source.write_text("def build():\n    return None\n")
    current_record = ReviewSourceModelRecord(
        fixture_id="fixture/current",
        feature_name="Current",
        source_path=current_source,
    )
    stale_source = tmp_path / "stale.py"
    stale_source.write_text("def build():\n    return None\n")
    stale_record = ReviewSourceModelRecord(
        fixture_id="fixture/stale",
        feature_name="Stale",
        source_path=stale_source,
    )
    stale_request = PreviewPayloadRequest.from_source_record(
        stale_record,
        owner="preview-payload",
        request_id=99,
        generation=99,
    )
    stale_payload = PreviewPayload.failure(
        stale_request,
        PreviewPayloadDiagnostic(
            code="preview-payload-error",
            message="stale failure",
            fixture_id=stale_request.fixture_id,
            owner=stale_request.owner,
            request_id=stale_request.request_id,
            generation=stale_request.generation,
        ),
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    controller.launch(current_record)
    ready: list[PreviewPayloadReadyEvent] = []
    failures: list[PreviewPayloadFailedEvent] = []

    decision = controller.handle_process_result(
        PreviewPayloadProcessResult(stale_payload),
        ready.append,
        failures.append,
    )
    controller.close()

    assert not decision.accepted
    assert decision.reason == "stale_payload_rejected"
    assert ready == []
    assert failures == []


def test_preview_payload_process_controller_cancelled_completion_does_not_handoff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "current.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        fixture_id="fixture/cancelled",
        feature_name="Cancelled",
        source_path=source,
    )
    controller = PreviewPayloadProcessController(payload_dir=tmp_path, cwd=tmp_path)
    controller.launch(record)
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    payload_path = tmp_path / "cancelled-payload.json"
    payload_path.write_text("{}")
    payload = PreviewPayload.success(request, payload_path=payload_path)
    controller.adopt_payload(payload)

    cancel_diagnostic = controller.cancel_active_request()
    seen: list[PreviewPayloadReadyEvent] = []
    decision = controller.handle_process_result(
        PreviewPayloadProcessResult(payload),
        seen.append,
    )
    controller.close()

    assert cancel_diagnostic is not None
    assert cancel_diagnostic.reason == "cancelled"
    assert not decision.accepted
    assert decision.reason == "stale_payload_rejected"
    assert seen == []


def test_preview_payload_process_controller_has_no_ui_or_renderer_import_boundary() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src/impression/devtools/reference_review/preview_payload_controller.py"
    )
    source = module_path.read_text()

    assert "PySide6" not in source
    assert "pyvista" not in source
    assert "impression.devtools.reference_review.ui" not in source
