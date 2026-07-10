from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

from impression.devtools.reference_review import (
    EntrypointParameterRecord,
    ReferenceReviewStatus,
    ReviewSourceLoadMode,
    ReviewSourceModelRecord,
    approve_reference_artifacts,
    build_review_context_payload,
    discover_source_records,
    load_source_records_from_database,
    load_source_records_from_file,
    resolve_generated_review_module,
    update_fixture_notes_in_file,
    update_fixture_review_status_in_database,
    update_fixture_review_status_in_file,
    validate_source_record,
)


def _write_source(root: Path, name: str = "model.py") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def build():\n    return None\n")
    return path


def test_source_model_record_normalizes_mapping_and_exposes_identity(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    artifact = tmp_path / "dirty.impress"
    artifact.write_text('{"format": "impress"}\n')

    record = ReviewSourceModelRecord.from_mapping(
        {
            "fixture_id": "demo/box",
            "feature_name": "box",
            "source_path": source.name,
            "load_mode": "module",
            "entrypoint": "build",
            "expected_output": "png",
            "purpose": "Exercise fixture metadata",
            "methodology": "Compare the canonical rendered STL against the selected artifact.",
            "render_description": "A single centered box with crisp vertical sides.",
            "notes": "Initial reviewer note.",
            "parameters": [{"name": "width", "value": 12}],
            "artifact_paths": [artifact.name],
        },
        base_dir=tmp_path,
    )

    assert record.load_mode is ReviewSourceLoadMode.MODULE
    assert record.identity.key == ("demo/box", source.as_posix(), "build")
    assert record.purpose == "Exercise fixture metadata"
    assert record.methodology == "Compare the canonical rendered STL against the selected artifact."
    assert record.render_description == "A single centered box with crisp vertical sides."
    assert record.notes == "Initial reviewer note."
    assert record.parameters == (EntrypointParameterRecord("width", 12),)
    assert record.artifact_paths == (artifact,)


def test_validation_reports_multiple_blocking_diagnostics_without_importing(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_model.py"
    missing_artifact = tmp_path / "missing.impress"
    record = ReviewSourceModelRecord(
        fixture_id="demo/missing",
        feature_name="missing",
        source_path=outside,
        load_mode=ReviewSourceLoadMode.CALLABLE,
        entrypoint="build",
        artifact_paths=(missing_artifact,),
    )

    result = validate_source_record(record, allowed_root=tmp_path)

    assert not result.valid
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "source-outside-root",
        "missing-source",
        "callable-entrypoint-not-qualified",
        "missing-artifact",
    }


def test_discovery_reads_review_source_manifests_and_reports_duplicates(tmp_path: Path) -> None:
    first_root = tmp_path / "fixtures"
    first_source = _write_source(first_root / "a")
    second_source = _write_source(first_root / "b")
    for folder, source in ((first_root / "a", first_source), (first_root / "b", second_source)):
        (folder / "review-source.json").write_text(
            json.dumps(
                {
                    "fixture_id": "demo/repeated",
                    "feature_name": "demo",
                    "source_path": source.name,
                    "entrypoint": "build",
                }
            )
        )

    summary = discover_source_records((first_root,))

    assert len(summary.items) == 2
    assert len(summary.valid_items) == 2
    assert any(diagnostic.code == "duplicate-fixture-id" for diagnostic in summary.diagnostics)


def test_fixture_file_loads_review_source_records(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "fixture_model.py")
    fixture_file = tmp_path / "review-fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/file",
                        "feature_name": "demo",
                        "source_path": source.name,
                    }
                ]
            }
        )
    )

    summary = load_source_records_from_file(fixture_file)

    assert len(summary.valid_items) == 1
    assert summary.valid_items[0].record.fixture_id == "demo/file"
    assert summary.valid_items[0].record.review_status is ReferenceReviewStatus.UNREVIEWED


def test_fixture_file_persists_review_status_and_gold_artifact_path(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "fixture_model.py")
    dirty = tmp_path / "reference-stl" / "dirty" / "demo" / "fixture.stl"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("solid demo\nendsolid demo\n")
    fixture_file = tmp_path / "review-fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/file",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "artifact_paths": [dirty.relative_to(tmp_path).as_posix()],
                    }
                ]
            }
        )
    )
    record = load_source_records_from_file(fixture_file).valid_items[0].record

    promotion = approve_reference_artifacts(record)
    result = update_fixture_review_status_in_file(
        fixture_file,
        fixture_id=record.fixture_id,
        status=ReferenceReviewStatus.APPROVED,
        artifact_paths=promotion.artifact_paths,
    )
    reloaded = load_source_records_from_file(fixture_file).valid_items[0].record

    assert promotion.updated
    assert result.updated
    assert not dirty.exists()
    assert reloaded.review_status is ReferenceReviewStatus.APPROVED
    assert reloaded.artifact_paths[0].parts[-3:] == ("gold", "demo", "fixture.stl")
    assert reloaded.artifact_paths[0].read_text() == "solid demo\nendsolid demo\n"


def test_fixture_file_persists_review_notes(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "fixture_model.py")
    fixture_file = tmp_path / "review-fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/file",
                        "feature_name": "demo",
                        "source_path": source.name,
                    }
                ]
            }
        )
    )

    result = update_fixture_notes_in_file(
        fixture_file,
        fixture_id="demo/file",
        notes="Approved geometry, but keep an eye on the top edge.",
    )
    payload = json.loads(fixture_file.read_text())
    reloaded = load_source_records_from_file(fixture_file).valid_items[0].record

    assert result.updated
    assert payload["fixtures"][0]["notes"] == "Approved geometry, but keep an eye on the top edge."
    assert reloaded.notes == "Approved geometry, but keep an eye on the top edge."


def test_fixture_file_stores_promoted_artifacts_relative_to_fixture_file_directory(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "tests" / "reference_review_fixtures"
    fixture_dir.mkdir(parents=True)
    source = _write_source(fixture_dir, "fixture_model.py")
    dirty = tmp_path / "project" / "release" / "reference-stl" / "dirty" / "demo" / "fixture.stl"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("solid demo\nendsolid demo\n")
    fixture_file = fixture_dir / "review-fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "allowed_root": "../..",
                "fixtures": [
                    {
                        "fixture_id": "demo/file",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "artifact_paths": ["../../project/release/reference-stl/dirty/demo/fixture.stl"],
                    }
                ],
            }
        )
    )
    record = load_source_records_from_file(fixture_file).valid_items[0].record

    promotion = approve_reference_artifacts(record)
    result = update_fixture_review_status_in_file(
        fixture_file,
        fixture_id=record.fixture_id,
        status=ReferenceReviewStatus.APPROVED,
        artifact_paths=promotion.artifact_paths,
    )
    payload = json.loads(fixture_file.read_text())
    reloaded = load_source_records_from_file(fixture_file)

    assert result.updated
    assert payload["fixtures"][0]["artifact_paths"] == [
        "../../project/release/reference-stl/gold/demo/fixture.stl"
    ]
    assert len(reloaded.valid_items) == 1
    assert reloaded.valid_items[0].record.review_status is ReferenceReviewStatus.APPROVED


def test_fixture_database_loads_review_source_records(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "db_model.py")
    database = tmp_path / "review-fixtures.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table review_sources (fixture_id text, feature_name text, source_path text, entrypoint text)"
        )
        connection.execute(
            "insert into review_sources values (?, ?, ?, ?)",
            ("demo/db", "demo", source.name, "build"),
        )

    summary = load_source_records_from_database(database)

    assert len(summary.valid_items) == 1
    assert summary.valid_items[0].record.fixture_id == "demo/db"


def test_fixture_database_persists_declined_status(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "db_model.py")
    database = tmp_path / "review-fixtures.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table review_sources (fixture_id text, feature_name text, source_path text, entrypoint text)"
        )
        connection.execute(
            "insert into review_sources values (?, ?, ?, ?)",
            ("demo/db", "demo", source.name, "build"),
        )

    result = update_fixture_review_status_in_database(
        database,
        fixture_id="demo/db",
        status=ReferenceReviewStatus.DECLINED,
    )
    reloaded = load_source_records_from_database(database).valid_items[0].record

    assert result.updated
    assert reloaded.review_status is ReferenceReviewStatus.DECLINED


def test_committed_demo_fixture_file_loads(project_root: Path) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/demo-fixtures.json"

    summary = load_source_records_from_file(fixture_file)

    assert not summary.diagnostics
    assert len(summary.valid_items) == 1
    assert summary.valid_items[0].record.fixture_id == "examples/hello-cube"


def test_dirty_impress_fixture_file_covers_dirty_impress_inventory(project_root: Path) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    dirty_impress = sorted((project_root / "tests/reference_review_fixtures/reference-impress/dirty").rglob("*.impress"))

    summary = load_source_records_from_file(fixture_file)
    records = tuple(item.record for item in summary.valid_items)
    artifact_paths = sorted(record.artifact_paths[0].resolve() for record in records)

    assert not summary.diagnostics
    assert len(records) == len(dirty_impress)
    assert artifact_paths == dirty_impress
    assert all(record.source_path.name == "stl_review_sources.py" for record in records)


def test_dirty_impress_fixture_entrypoints_build_reviewable_models(project_root: Path) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    summary = load_source_records_from_file(fixture_file)
    module_path = project_root / "tests/reference_review_fixtures/stl_review_sources.py"
    spec = importlib.util.spec_from_file_location("stl_review_sources_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    built_types = {
        getattr(module, item.record.entrypoint)().__class__.__name__ for item in summary.valid_items
    }

    assert built_types == {"SurfaceBody"}


def test_review_context_payload_is_deterministic_and_omits_absolute_source_path(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "candidate.py")
    record = ReviewSourceModelRecord(
        fixture_id="demo/context",
        feature_name="context",
        source_path=source,
        expected_output="image+stl",
        description="Context fixture",
        purpose="Verify context serialization",
        methodology="Serialize twice and compare stable dictionaries.",
        render_description="A deterministic image and STL pair.",
        parameters=(EntrypointParameterRecord("scale", 2),),
    )

    first = build_review_context_payload(record).to_json_dict()
    second = build_review_context_payload(record).to_json_dict()

    assert first == second
    assert first["source_display_path"] == "candidate.py"
    assert first["purpose"] == "Verify context serialization"
    assert first["methodology"] == "Serialize twice and compare stable dictionaries."
    assert first["render_description"] == "A deterministic image and STL pair."
    assert first["artifact_display_paths"] == []
    assert str(tmp_path) not in json.dumps(first)


def test_generated_review_module_must_live_under_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "candidates"
    source = _write_source(allowed, "generated.py")
    accepted = resolve_generated_review_module(
        source,
        allowed_root=allowed,
        fixture_id="demo/generated",
        feature_name="generated",
    )
    refused = resolve_generated_review_module(
        tmp_path / "outside.py",
        allowed_root=allowed,
        fixture_id="demo/generated",
        feature_name="generated",
    )

    assert accepted.valid
    assert accepted.record is not None
    assert accepted.record.generated
    assert not refused.valid
    assert refused.diagnostics[0].code == "generated-source-outside-root"
