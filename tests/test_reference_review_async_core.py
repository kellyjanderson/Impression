from __future__ import annotations

from threading import Event
from pathlib import Path

from impression.devtools.reference_review.async_core import (
    AuditEmitter,
    DurableWriteLane,
    DurableWriteRequest,
    LatestRequestTracker,
    RequestIdAllocator,
    ReviewTaskKind,
    ReviewWorkbenchMessage,
    TaskDispatcher,
    UICompletionBridge,
    WorkerPolicy,
    WorkerResultEnvelope,
    build_audit_event,
)


def _message(
    request_id: int = 1,
    *,
    owner: str = "preview",
    kind: ReviewTaskKind = ReviewTaskKind.PREVIEW_BUILD,
    fixture_id: str = "fixture-a",
) -> ReviewWorkbenchMessage:
    return ReviewWorkbenchMessage(
        owner=owner,
        kind=kind,
        request_id=request_id,
        fixture_id=fixture_id,
    )


def test_request_ids_are_monotonic_per_owner_and_envelopes_are_immutable() -> None:
    allocator = RequestIdAllocator()

    first = allocator.next("preview")
    second = allocator.next("preview")
    other_owner = allocator.next("notes")
    message = ReviewWorkbenchMessage(
        owner="preview",
        kind=ReviewTaskKind.PREVIEW_BUILD,
        request_id=first,
        fixture_id="fixture-a",
        payload={"path": "model.py"},
    )

    assert (first, second, other_owner) == (1, 2, 1)
    assert message.owner_key == ("preview", ReviewTaskKind.PREVIEW_BUILD, "fixture-a")
    assert message.to_audit_payload()["request_id"] == 1
    try:
        message.payload["extra"] = "blocked"
    except TypeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("payload should be immutable")


def test_worker_result_envelope_requires_error_text_for_failures() -> None:
    message = _message()
    failure = WorkerResultEnvelope(request=message, ok=False, error="boom")

    assert failure.owner == "preview"
    assert failure.kind is ReviewTaskKind.PREVIEW_BUILD
    assert failure.error == "boom"


def test_dispatcher_accepts_tasks_and_returns_typed_completion_envelopes() -> None:
    dispatcher = TaskDispatcher()
    request = _message()

    result = dispatcher.dispatch(request, lambda message: {"seen": message.request_id})
    envelope = result.future.result(timeout=2) if result.future else None
    dispatcher.close()

    assert result.accepted
    assert envelope is not None
    assert envelope.ok
    assert envelope.result == {"seen": 1}


def test_dispatcher_rejects_when_bounded_queue_is_full() -> None:
    release = Event()
    hold: list[ReviewWorkbenchMessage] = []

    def worker(message: ReviewWorkbenchMessage) -> str:
        hold.append(message)
        release.wait(timeout=2)
        return "done"

    dispatcher = TaskDispatcher(
        max_workers=1,
        policies={ReviewTaskKind.PREVIEW_BUILD: WorkerPolicy(max_pending=1)},
    )

    first = dispatcher.dispatch(_message(1), worker)
    second = dispatcher.dispatch(_message(2), worker)
    release.set()
    if first.future:
        first.future.result(timeout=2)
    dispatcher.close()

    assert first.accepted
    assert not second.accepted
    assert second.diagnostic == "queue_full"
    assert len(hold) == 1


def test_latest_request_tracker_rejects_stale_completions_and_cancels_old_tokens() -> None:
    tracker = LatestRequestTracker()
    old = _message(1)
    new = _message(2)

    old_token = tracker.register(old)
    tracker.register(new)
    old_decision = tracker.decide(WorkerResultEnvelope(request=old, ok=True))
    new_decision = tracker.decide(WorkerResultEnvelope(request=new, ok=True))

    assert old_token.cancelled
    assert not old_decision.accepted
    assert old_decision.reason == "stale_completion_rejected"
    assert new_decision.accepted


def test_durable_write_lane_serializes_writes_and_removes_lock_file(tmp_path: Path) -> None:
    lane = DurableWriteLane()
    target = tmp_path / "note.md"
    request = DurableWriteRequest(name="note", root=tmp_path, lock_name="note")

    result = lane.run(request, lambda: target.write_text("reviewed"))

    assert result.accepted
    assert target.read_text() == "reviewed"
    assert not request.lock_path.exists()


def test_durable_write_lane_reports_lock_timeout(tmp_path: Path) -> None:
    lane = DurableWriteLane()
    request = DurableWriteRequest(
        name="promotion",
        root=tmp_path,
        lock_name="promotion",
        timeout_seconds=0.01,
    )
    request.lock_path.write_text("held")

    result = lane.run(request, lambda: None)

    assert not result.accepted
    assert result.diagnostic == "lock_timeout"


def test_ui_completion_bridge_sanitizes_paths_before_handoff(tmp_path: Path) -> None:
    request = _message()
    seen: list[WorkerResultEnvelope] = []
    bridge = UICompletionBridge(seen.append, cwd=tmp_path, home=tmp_path.parent)
    envelope = WorkerResultEnvelope(
        request=request,
        ok=False,
        error=f"{tmp_path}/model.py failed with /var/private/token.txt",
    )

    posted = bridge.post(envelope)
    diagnostic = bridge.diagnostic_for(posted)

    assert seen == [posted]
    assert diagnostic is not None
    assert str(tmp_path) not in diagnostic.message
    assert "/var/private" not in diagnostic.message
    assert "<path>" in diagnostic.message or "<workspace>" in diagnostic.message


def test_audit_events_are_redacted_json_compatible_and_sink_failures_do_not_block() -> None:
    def failing_sink(_event) -> None:
        raise RuntimeError("sink unavailable")

    request = _message(kind=ReviewTaskKind.CODEX_REQUEST)
    emitter = AuditEmitter(sink=failing_sink)
    event = build_audit_event(
        "task_failed",
        request,
        details={"api_token": "secret", "nested": {"password": "secret"}, "ok": True},
    )

    emitted = emitter.emit(event)
    payload = event.to_json_dict()

    assert not emitted
    assert emitter.events == [event]
    assert payload["task_kind"] == "codex_request"
    assert payload["details"]["api_token"] == "<redacted>"
    assert payload["details"]["nested"]["password"] == "<redacted>"
