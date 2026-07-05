"""Source registry contracts for reference review fixtures."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .async_core.qt_handoff import sanitize_error_text


class ReviewSourceLoadMode(str, Enum):
    MODULE = "module"
    CALLABLE = "callable"
    GENERATED_REVIEW_MODULE = "generated-review-module"


@dataclass(frozen=True)
class EntrypointParameterRecord:
    """JSON-compatible parameter passed to a review model entrypoint."""

    name: str
    value: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        try:
            json.dumps(self.value, sort_keys=True)
        except TypeError as exc:
            raise ValueError("parameter value must be JSON-compatible") from exc


@dataclass(frozen=True)
class SourceIdentity:
    """Stable source identity for queue, preview, notes, and Codex context."""

    fixture_id: str
    source_path: Path
    entrypoint: str = "build"

    @property
    def display_path(self) -> str:
        return self.source_path.name

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.fixture_id, self.source_path.as_posix(), self.entrypoint)


@dataclass(frozen=True)
class ReviewSourceModelRecord:
    """Loadable source model record for a reference fixture."""

    fixture_id: str
    feature_name: str
    source_path: Path
    load_mode: ReviewSourceLoadMode = ReviewSourceLoadMode.MODULE
    entrypoint: str = "build"
    expected_output: str | None = None
    description: str | None = None
    parameters: tuple[EntrypointParameterRecord, ...] = ()
    generated: bool = False

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ValueError("fixture_id must not be empty")
        if not self.feature_name:
            raise ValueError("feature_name must not be empty")
        if not self.entrypoint:
            raise ValueError("entrypoint must not be empty")
        object.__setattr__(self, "load_mode", ReviewSourceLoadMode(self.load_mode))
        object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            fixture_id=self.fixture_id,
            source_path=self.source_path,
            entrypoint=self.entrypoint,
        )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ReviewSourceModelRecord":
        source_path = Path(str(data.get("source_path", "")))
        if base_dir is not None and not source_path.is_absolute():
            source_path = base_dir / source_path
        parameters = tuple(
            EntrypointParameterRecord(name=str(item["name"]), value=item.get("value"))
            for item in data.get("parameters", ())
        )
        return cls(
            fixture_id=str(data.get("fixture_id", "")),
            feature_name=str(data.get("feature_name", "")),
            source_path=source_path,
            load_mode=ReviewSourceLoadMode(data.get("load_mode", ReviewSourceLoadMode.MODULE)),
            entrypoint=str(data.get("entrypoint", "build")),
            expected_output=data.get("expected_output"),
            description=data.get("description"),
            parameters=parameters,
            generated=bool(data.get("generated", False)),
        )


@dataclass(frozen=True)
class SourceValidationDiagnostic:
    code: str
    message: str
    fixture_id: str | None = None


@dataclass(frozen=True)
class SourceValidationResult:
    record: ReviewSourceModelRecord | None
    diagnostics: tuple[SourceValidationDiagnostic, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.diagnostics and self.record is not None


def validate_source_record(
    record: ReviewSourceModelRecord,
    *,
    allowed_root: Path | None = None,
) -> SourceValidationResult:
    diagnostics: list[SourceValidationDiagnostic] = []
    source_path = record.source_path
    if allowed_root is not None:
        root = allowed_root.resolve()
        try:
            source_path.resolve().relative_to(root)
        except ValueError:
            diagnostics.append(
                SourceValidationDiagnostic(
                    "source-outside-root",
                    "source path is outside the configured reference root",
                    record.fixture_id,
                )
            )
    if not source_path.exists():
        diagnostics.append(
            SourceValidationDiagnostic(
                "missing-source",
                sanitize_error_text(str(source_path)),
                record.fixture_id,
            )
        )
    elif not source_path.is_file():
        diagnostics.append(
            SourceValidationDiagnostic("source-not-file", source_path.name, record.fixture_id)
        )
    if record.load_mode is ReviewSourceLoadMode.CALLABLE and "." not in record.entrypoint:
        diagnostics.append(
            SourceValidationDiagnostic(
                "callable-entrypoint-not-qualified",
                "callable load mode requires a dotted entrypoint",
                record.fixture_id,
            )
        )
    if record.load_mode is ReviewSourceLoadMode.MODULE and not record.entrypoint:
        diagnostics.append(
            SourceValidationDiagnostic("missing-entrypoint", "entrypoint is required", record.fixture_id)
        )
    return SourceValidationResult(record=record, diagnostics=tuple(diagnostics))


@dataclass(frozen=True)
class DiscoveryItem:
    record: ReviewSourceModelRecord
    validation: SourceValidationResult


@dataclass(frozen=True)
class DiscoverySummary:
    items: tuple[DiscoveryItem, ...] = ()
    diagnostics: tuple[SourceValidationDiagnostic, ...] = ()

    @property
    def valid_items(self) -> tuple[DiscoveryItem, ...]:
        return tuple(item for item in self.items if item.validation.valid)


def discover_source_records(roots: tuple[Path, ...] | list[Path]) -> DiscoverySummary:
    items: list[DiscoveryItem] = []
    diagnostics: list[SourceValidationDiagnostic] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root)
        for manifest in sorted(root.rglob("review-source.json")):
            try:
                payload = json.loads(manifest.read_text())
                record = ReviewSourceModelRecord.from_mapping(payload, base_dir=manifest.parent)
            except Exception as exc:
                diagnostics.append(SourceValidationDiagnostic("invalid-source-record", str(exc)))
                continue
            if record.fixture_id in seen:
                diagnostics.append(
                    SourceValidationDiagnostic(
                        "duplicate-fixture-id",
                        record.fixture_id,
                        record.fixture_id,
                    )
                )
            seen.add(record.fixture_id)
            validation = validate_source_record(record, allowed_root=root)
            items.append(DiscoveryItem(record=record, validation=validation))
    return DiscoverySummary(items=tuple(items), diagnostics=tuple(diagnostics))


def load_source_records_from_file(path: Path) -> DiscoverySummary:
    """Load review source records from a JSON fixture file."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return DiscoverySummary(
            diagnostics=(SourceValidationDiagnostic("invalid-fixture-file", str(exc)),)
        )
    allowed_root = path.parent
    if isinstance(payload, Mapping) and payload.get("allowed_root") is not None:
        allowed_root = path.parent / str(payload["allowed_root"])
    rows = payload.get("fixtures", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        rows = (rows,)
    if not isinstance(rows, list | tuple):
        return DiscoverySummary(
            diagnostics=(
                SourceValidationDiagnostic(
                    "invalid-fixture-file",
                    "fixture file must contain a record, list, or fixtures list",
                ),
            )
        )
    return _records_from_mappings(rows, base_dir=path.parent, allowed_root=allowed_root)


def load_source_records_from_database(path: Path) -> DiscoverySummary:
    """Load review source records from a SQLite review_sources table."""

    path = Path(path)
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("select * from review_sources order by fixture_id").fetchall()
    except Exception as exc:
        return DiscoverySummary(
            diagnostics=(SourceValidationDiagnostic("invalid-fixture-database", str(exc)),)
        )
    records: list[dict[str, Any]] = []
    for row in rows:
        data = dict(row)
        parameters = data.get("parameters")
        if isinstance(parameters, str) and parameters:
            data["parameters"] = json.loads(parameters)
        records.append(data)
    return _records_from_mappings(records, base_dir=path.parent, allowed_root=path.parent)


def _records_from_mappings(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_dir: Path,
    allowed_root: Path,
) -> DiscoverySummary:
    items: list[DiscoveryItem] = []
    diagnostics: list[SourceValidationDiagnostic] = []
    seen: set[str] = set()
    for row in rows:
        try:
            record = ReviewSourceModelRecord.from_mapping(row, base_dir=base_dir)
        except Exception as exc:
            diagnostics.append(SourceValidationDiagnostic("invalid-source-record", str(exc)))
            continue
        if record.fixture_id in seen:
            diagnostics.append(
                SourceValidationDiagnostic(
                    "duplicate-fixture-id",
                    record.fixture_id,
                    record.fixture_id,
                )
            )
        seen.add(record.fixture_id)
        items.append(DiscoveryItem(record, validate_source_record(record, allowed_root=allowed_root)))
    return DiscoverySummary(items=tuple(items), diagnostics=tuple(diagnostics))


@dataclass(frozen=True)
class ReviewContextPayload:
    fixture_id: str
    feature_name: str
    source_display_path: str
    entrypoint: str
    expected_output: str | None
    description: str | None
    parameters: tuple[EntrypointParameterRecord, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "feature_name": self.feature_name,
            "source_display_path": self.source_display_path,
            "entrypoint": self.entrypoint,
            "expected_output": self.expected_output,
            "description": self.description,
            "parameters": [
                {"name": item.name, "value": item.value} for item in self.parameters
            ],
        }


def build_review_context_payload(record: ReviewSourceModelRecord) -> ReviewContextPayload:
    return ReviewContextPayload(
        fixture_id=record.fixture_id,
        feature_name=record.feature_name,
        source_display_path=record.identity.display_path,
        entrypoint=record.entrypoint,
        expected_output=record.expected_output,
        description=record.description,
        parameters=record.parameters,
    )


@dataclass(frozen=True)
class GeneratedSourceReference:
    record: ReviewSourceModelRecord
    lifecycle_state: str


def resolve_generated_review_module(
    path: Path,
    *,
    allowed_root: Path,
    fixture_id: str,
    feature_name: str,
    entrypoint: str = "build",
) -> SourceValidationResult:
    resolved_root = allowed_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        record = ReviewSourceModelRecord(
            fixture_id=fixture_id,
            feature_name=feature_name,
            source_path=path,
            load_mode=ReviewSourceLoadMode.GENERATED_REVIEW_MODULE,
            entrypoint=entrypoint,
            generated=True,
        )
        return SourceValidationResult(
            record=record,
            diagnostics=(
                SourceValidationDiagnostic(
                    "generated-source-outside-root",
                    "generated source is outside allowed roots",
                    fixture_id,
                ),
            ),
        )
    record = ReviewSourceModelRecord(
        fixture_id=fixture_id,
        feature_name=feature_name,
        source_path=resolved_path,
        load_mode=ReviewSourceLoadMode.GENERATED_REVIEW_MODULE,
        entrypoint=entrypoint,
        generated=True,
    )
    return validate_source_record(record, allowed_root=resolved_root)
