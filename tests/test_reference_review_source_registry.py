from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from impression.devtools.reference_review import (
    EntrypointParameterRecord,
    ReviewSourceLoadMode,
    ReviewSourceModelRecord,
    build_review_context_payload,
    discover_source_records,
    load_source_records_from_database,
    load_source_records_from_file,
    resolve_generated_review_module,
    validate_source_record,
)


def _write_source(root: Path, name: str = "model.py") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def build():\n    return None\n")
    return path


def test_source_model_record_normalizes_mapping_and_exposes_identity(tmp_path: Path) -> None:
    source = _write_source(tmp_path)

    record = ReviewSourceModelRecord.from_mapping(
        {
            "fixture_id": "demo/box",
            "feature_name": "box",
            "source_path": source.name,
            "load_mode": "module",
            "entrypoint": "build",
            "expected_output": "png",
            "parameters": [{"name": "width", "value": 12}],
        },
        base_dir=tmp_path,
    )

    assert record.load_mode is ReviewSourceLoadMode.MODULE
    assert record.identity.key == ("demo/box", source.as_posix(), "build")
    assert record.parameters == (EntrypointParameterRecord("width", 12),)


def test_validation_reports_multiple_blocking_diagnostics_without_importing(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_model.py"
    record = ReviewSourceModelRecord(
        fixture_id="demo/missing",
        feature_name="missing",
        source_path=outside,
        load_mode=ReviewSourceLoadMode.CALLABLE,
        entrypoint="build",
    )

    result = validate_source_record(record, allowed_root=tmp_path)

    assert not result.valid
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "source-outside-root",
        "missing-source",
        "callable-entrypoint-not-qualified",
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


def test_committed_demo_fixture_file_loads(project_root: Path) -> None:
    fixture_file = project_root / "tests/reference_review_fixtures/demo-fixtures.json"

    summary = load_source_records_from_file(fixture_file)

    assert not summary.diagnostics
    assert len(summary.valid_items) == 1
    assert summary.valid_items[0].record.fixture_id == "examples/hello-cube"


def test_review_context_payload_is_deterministic_and_omits_absolute_source_path(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "candidate.py")
    record = ReviewSourceModelRecord(
        fixture_id="demo/context",
        feature_name="context",
        source_path=source,
        expected_output="image+stl",
        description="Context fixture",
        parameters=(EntrypointParameterRecord("scale", 2),),
    )

    first = build_review_context_payload(record).to_json_dict()
    second = build_review_context_payload(record).to_json_dict()

    assert first == second
    assert first["source_display_path"] == "candidate.py"
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
