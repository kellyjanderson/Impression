from __future__ import annotations

import json
from pathlib import Path

from impression.devtools.reference_review import (
    PromotionArtifact,
    PromotionExecutor,
    PromotionProvenanceStore,
    PromotionRequest,
    ReviewNoteRecord,
    ReviewNoteStore,
    ReviewSourceModelRecord,
    ReviewState,
    build_release_gate_report,
    build_review_context_payload,
    classify_review_state,
    make_provenance_record,
    validate_promotion,
)


def _source_record(tmp_path: Path) -> ReviewSourceModelRecord:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    return ReviewSourceModelRecord(
        fixture_id="demo/fixture",
        feature_name="demo",
        source_path=source,
    )


def test_review_note_store_saves_loads_and_redacts_sensitive_text(tmp_path: Path) -> None:
    store = ReviewNoteStore(tmp_path / "notes")
    note = ReviewNoteRecord(
        fixture_id="demo/fixture",
        status=ReviewState.NEEDS_WORK,
        body="needs edge cleanup\napi_token: abc123",
    )

    result = store.save(note)
    loaded = store.load("demo/fixture")

    assert result.saved
    assert loaded is not None
    assert loaded.status is ReviewState.NEEDS_WORK
    assert "abc123" not in result.path.read_text()
    assert "<redacted>" in result.path.read_text()


def test_review_state_classifier_distinguishes_unreviewed_noted_blocked_approved_and_promoted() -> None:
    assert classify_review_state(fixture_id="a", note=None, promoted=False).state is ReviewState.UNREVIEWED
    assert (
        classify_review_state(
            fixture_id="a",
            note=ReviewNoteRecord("a", ReviewState.NEEDS_WORK, "fix it"),
            promoted=False,
        ).state
        is ReviewState.NEEDS_WORK
    )
    assert (
        classify_review_state(
            fixture_id="a",
            note=ReviewNoteRecord("a", ReviewState.BLOCKED, "missing context"),
            promoted=False,
        ).state
        is ReviewState.BLOCKED
    )
    assert (
        classify_review_state(
            fixture_id="a",
            note=ReviewNoteRecord("a", ReviewState.APPROVED_SOURCE, "looks right"),
            promoted=False,
        ).state
        is ReviewState.APPROVED_SOURCE
    )
    assert (
        classify_review_state(
            fixture_id="a",
            note=ReviewNoteRecord("a", ReviewState.NEEDS_WORK, "old note"),
            promoted=True,
        ).state
        is ReviewState.PROMOTED
    )


def test_promotion_validator_requires_source_and_dirty_artifacts(tmp_path: Path) -> None:
    missing = PromotionArtifact(
        kind="png",
        dirty_path=tmp_path / "dirty.png",
        gold_path=tmp_path / "gold.png",
    )

    result = validate_promotion(source_record=None, artifacts=(missing,))

    assert not result.allowed
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "missing-source-record",
        "missing-dirty-artifact",
    }


def test_promotion_executor_copies_dirty_artifacts_to_gold_with_checksums(tmp_path: Path) -> None:
    source_record = _source_record(tmp_path)
    dirty = tmp_path / "dirty" / "fixture.png"
    gold = tmp_path / "gold" / "fixture.png"
    dirty.parent.mkdir()
    dirty.write_text("image-bytes")
    artifact = PromotionArtifact(kind="png", dirty_path=dirty, gold_path=gold)

    result = PromotionExecutor().promote(
        PromotionRequest(
            fixture_id="demo/fixture",
            source_record=source_record,
            artifacts=(artifact,),
            root=tmp_path,
        )
    )

    assert result.promoted
    assert gold.read_text() == "image-bytes"
    assert result.checksums["png"] == artifact.checksum


def test_promotion_provenance_excludes_secret_note_data_and_release_report_orders_failures(tmp_path: Path) -> None:
    source_record = _source_record(tmp_path)
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("artifact")
    artifact = PromotionArtifact(kind="txt", dirty_path=dirty, gold_path=tmp_path / "gold.txt")
    promotion = PromotionExecutor().promote(
        PromotionRequest(
            fixture_id="demo/fixture",
            source_record=source_record,
            artifacts=(artifact,),
            root=tmp_path,
        )
    )
    context = build_review_context_payload(source_record)
    provenance = make_provenance_record(
        fixture_id="demo/fixture",
        source_record=source_record,
        promotion=promotion,
        context=context,
    )
    store = PromotionProvenanceStore(tmp_path / "provenance")

    assert store.write(provenance)
    payload = json.loads(store.path_for("demo/fixture").read_text())
    assert payload["fixture_id"] == "demo/fixture"
    assert "artifact_checksums" in payload

    report = build_release_gate_report(
        (
            classify_review_state(fixture_id="z", note=None, promoted=False),
            classify_review_state(fixture_id="a", note=None, promoted=True),
        )
    )

    assert not report.passed
    assert [item.fixture_id for item in report.assessments] == ["a", "z"]
    assert report.failing[0].fixture_id == "z"

