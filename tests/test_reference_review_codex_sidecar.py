from __future__ import annotations

from pathlib import Path

from impression.devtools.reference_review import (
    AuditEmitter,
    CandidateModelStore,
    CandidateNotePatch,
    ReviewNoteRecord,
    ReviewSourceModelRecord,
    ReviewState,
    SidecarProcessLauncher,
    SidecarSessionRecord,
    SidecarTool,
    ToolPolicyBroker,
    ToolPolicyRecord,
    ToolRequestRecord,
    build_codex_context_payload,
    build_review_context_payload,
    default_context_for_source,
    propose_note_patch,
    request_regeneration,
)


def _source_record(tmp_path: Path) -> ReviewSourceModelRecord:
    path = tmp_path / "model.py"
    path.write_text("def build():\n    return None\n")
    return ReviewSourceModelRecord(
        fixture_id="demo/fixture",
        feature_name="demo",
        source_path=path,
    )


def test_codex_context_payload_is_minimal_and_fixture_scoped(tmp_path: Path) -> None:
    record = _source_record(tmp_path)
    context = build_codex_context_payload(
        build_review_context_payload(record),
        note=ReviewNoteRecord("demo/fixture", ReviewState.NEEDS_WORK, "secret token"),
    )

    payload = context.to_json_dict()

    assert payload["fixture_id"] == "demo/fixture"
    assert payload["note_summary"] == "needs-work"
    assert "chat_history" in payload["omissions"]
    assert str(tmp_path) not in str(payload)


def test_codex_context_payload_is_size_bounded() -> None:
    try:
        build_codex_context_payload(
            build_review_context_payload(
                ReviewSourceModelRecord(
                    fixture_id="demo/fixture",
                    feature_name="demo",
                    source_path=Path("model.py"),
                    description="x" * 40_000,
                )
            )
        )
    except ValueError as exc:
        assert str(exc) == "codex_context_too_large"
    else:
        raise AssertionError("expected oversized Codex context to be refused")


def test_tool_policy_broker_denies_by_default_and_audits_refusals(tmp_path: Path) -> None:
    audit = AuditEmitter()
    broker = ToolPolicyBroker(
        ToolPolicyRecord(
            allowed_tools=frozenset({SidecarTool.READ_FIXTURE_CONTEXT}),
            candidate_root=tmp_path,
        ),
        audit=audit,
    )
    broker.register(SidecarTool.READ_FIXTURE_CONTEXT, lambda request: {"fixture": request.fixture_id})

    allowed = broker.handle(
        ToolRequestRecord(SidecarTool.READ_FIXTURE_CONTEXT, fixture_id="demo/fixture")
    )
    refused = broker.handle(
        ToolRequestRecord(SidecarTool.WRITE_CANDIDATE_MODEL, fixture_id="demo/fixture")
    )

    assert allowed.accepted
    assert allowed.result == {"fixture": "demo/fixture"}
    assert not refused.accepted
    assert refused.diagnostic == "tool_not_allowed"
    assert audit.events[-1].event == "tool_refused"


def test_tool_policy_broker_refuses_unknown_shell_git_and_promote_requests(
    tmp_path: Path,
) -> None:
    audit = AuditEmitter()
    broker = ToolPolicyBroker(
        ToolPolicyRecord(allowed_tools=frozenset(), candidate_root=tmp_path),
        audit=audit,
    )

    shell = broker.handle_raw(tool="shell", fixture_id="demo/fixture")
    git = broker.handle_raw(tool="git", fixture_id="demo/fixture")
    promote = broker.handle_raw(tool="promote", fixture_id="demo/fixture")

    assert shell.diagnostic == "tool_unknown"
    assert git.diagnostic == "tool_unknown"
    assert promote.diagnostic == "tool_unknown"
    assert [event.details["reason"] for event in audit.events] == [
        "unknown_tool",
        "unknown_tool",
        "unknown_tool",
    ]


def test_candidate_model_store_writes_only_under_candidate_root(tmp_path: Path) -> None:
    store = CandidateModelStore(tmp_path / "candidates")

    accepted = store.write_candidate(
        fixture_id="demo/fixture",
        feature_name="demo",
        relative_path="demo__fixture_candidate.py",
        source_text="def build():\n    return None\n",
    )
    refused = store.write_candidate(
        fixture_id="demo/fixture",
        feature_name="demo",
        relative_path="../outside.py",
        source_text="def build():\n    return None\n",
    )

    assert accepted.accepted
    assert accepted.record is not None
    assert accepted.record.source_validation.valid
    assert not refused.accepted
    assert refused.diagnostic == "candidate_outside_root"


def test_candidate_model_store_refuses_oversized_and_overwrite(tmp_path: Path) -> None:
    store = CandidateModelStore(tmp_path / "candidates", max_bytes=16)

    accepted = store.write_candidate(
        fixture_id="demo/fixture",
        feature_name="demo",
        relative_path="demo__fixture_candidate.py",
        source_text="def build():\n",
    )
    overwrite = store.write_candidate(
        fixture_id="demo/fixture",
        feature_name="demo",
        relative_path="demo__fixture_candidate.py",
        source_text="def build():\n",
    )
    oversized = store.write_candidate(
        fixture_id="demo/fixture",
        feature_name="demo",
        relative_path="demo__fixture_big.py",
        source_text="x" * 17,
    )

    assert accepted.accepted
    assert overwrite.diagnostic == "candidate_already_exists"
    assert oversized.diagnostic == "candidate_too_large"


def test_candidate_note_patches_require_human_adoption_and_reject_sensitive_dump() -> None:
    accepted = propose_note_patch(fixture_id="demo/fixture", body="Try widening the lip.")
    refused = propose_note_patch(fixture_id="demo/fixture", body="full chat log password")
    stale = propose_note_patch(
        fixture_id="old/fixture",
        selected_fixture_id="demo/fixture",
        body="Try widening the lip.",
    )
    oversized = propose_note_patch(fixture_id="demo/fixture", body="x" * 20, max_bytes=8)

    assert accepted.accepted
    assert isinstance(accepted.result, CandidateNotePatch)
    assert not refused.accepted
    assert refused.diagnostic == "patch_contains_disallowed_detail"
    assert stale.diagnostic == "stale_fixture"
    assert oversized.diagnostic == "patch_too_large"


def test_regeneration_request_is_fixture_current_and_never_promotes(tmp_path: Path) -> None:
    record = _source_record(tmp_path)
    allowed = request_regeneration(
        fixture_id="demo/fixture",
        selected_fixture_id="demo/fixture",
        source_record=record,
    )
    stale = request_regeneration(
        fixture_id="old/fixture",
        selected_fixture_id="demo/fixture",
        source_record=record,
    )

    assert allowed.accepted
    assert allowed.result.source_path == record.source_path
    assert not stale.accepted
    assert stale.diagnostic == "stale_fixture"


def test_sidecar_session_can_cancel_without_direct_project_mutation(tmp_path: Path) -> None:
    record = _source_record(tmp_path)
    session = SidecarSessionRecord(fixture_id=record.fixture_id)
    cancelled = session.cancel()
    context = default_context_for_source(record)

    assert not session.cancelled
    assert cancelled.cancelled
    assert cancelled.session_id == session.session_id
    assert context.fixture_id == record.fixture_id


def test_sidecar_process_launcher_audits_start_and_failure() -> None:
    audit = AuditEmitter()
    launcher = SidecarProcessLauncher(audit=audit)

    session = launcher.start(fixture_id="demo/fixture")
    failure = launcher.fail(session, "process exited")

    assert session.fixture_id == "demo/fixture"
    assert failure.session_id == session.session_id
    assert [event.event for event in audit.events] == [
        "sidecar_session_started",
        "sidecar_session_failed",
    ]
