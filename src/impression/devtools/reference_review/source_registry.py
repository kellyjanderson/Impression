"""Source registry contracts for reference review fixtures."""

from __future__ import annotations

import json
import os
import shutil
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


class ReferenceReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    DECLINED = "declined"


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
class ReferenceEvidenceArtifactRecord:
    """Typed artifact entry inside a fixture evidence bundle."""

    role: str
    kind: str
    path: Path
    stage: str = "dirty"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("evidence artifact role must not be empty")
        if not self.kind:
            raise ValueError("evidence artifact kind must not be empty")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "stage", str(self.stage or "dirty"))
        object.__setattr__(self, "required", bool(self.required))


@dataclass(frozen=True)
class ReferenceEvidenceBundleRecord:
    """Typed evidence bundle exposed by file and database fixtures."""

    bundle_id: str
    evidence_kind: str
    role_policy: str = "named-artifacts"
    artifacts: tuple[ReferenceEvidenceArtifactRecord, ...] = ()
    section_plane_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("evidence bundle_id must not be empty")
        if not self.evidence_kind:
            raise ValueError("evidence evidence_kind must not be empty")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "section_plane_metadata", dict(self.section_plane_metadata))


@dataclass(frozen=True)
class SectionEvidenceContractRecord:
    """Validation result for section evidence bundle role and plane metadata."""

    bundle_id: str
    required_roles: tuple[str, ...]
    present_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    section_plane_metadata: Mapping[str, Any]
    diagnostics: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.missing_roles and not self.diagnostics


@dataclass(frozen=True)
class EvidencePathSet:
    """Dirty/gold path references for one generated evidence bundle."""

    dirty_paths: Mapping[str, Path]
    gold_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dirty_paths",
            {str(role): Path(path) for role, path in self.dirty_paths.items()},
        )
        object.__setattr__(
            self,
            "gold_paths",
            {str(role): Path(path) for role, path in self.gold_paths.items()},
        )
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
    purpose: str | None = None
    methodology: str | None = None
    render_description: str | None = None
    notes: str = ""
    parameters: tuple[EntrypointParameterRecord, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    evidence_bundles: tuple[ReferenceEvidenceBundleRecord, ...] = ()
    review_status: ReferenceReviewStatus = ReferenceReviewStatus.UNREVIEWED
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
        object.__setattr__(self, "artifact_paths", tuple(Path(path) for path in self.artifact_paths))
        object.__setattr__(self, "evidence_bundles", tuple(self.evidence_bundles))
        object.__setattr__(self, "review_status", ReferenceReviewStatus(self.review_status))

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
        artifact_paths = tuple(Path(str(path)) for path in data.get("artifact_paths", ()))
        if base_dir is not None:
            artifact_paths = tuple(
                path if path.is_absolute() else base_dir / path for path in artifact_paths
            )
        evidence_bundles = tuple(parse_file_evidence_bundles(data, base_dir=base_dir))
        return cls(
            fixture_id=str(data.get("fixture_id", "")),
            feature_name=str(data.get("feature_name", "")),
            source_path=source_path,
            load_mode=ReviewSourceLoadMode(data.get("load_mode", ReviewSourceLoadMode.MODULE)),
            entrypoint=str(data.get("entrypoint", "build")),
            expected_output=data.get("expected_output"),
            description=data.get("description"),
            purpose=data.get("purpose"),
            methodology=data.get("methodology"),
            render_description=data.get("render_description"),
            notes=str(data.get("notes", data.get("review_notes", "")) or ""),
            parameters=parameters,
            artifact_paths=artifact_paths,
            evidence_bundles=evidence_bundles,
            review_status=data.get("review_status", data.get("status", ReferenceReviewStatus.UNREVIEWED)),
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


@dataclass(frozen=True)
class ReferenceReviewStatusWriteResult:
    updated: bool
    diagnostics: tuple[SourceValidationDiagnostic, ...] = ()
    artifact_paths: tuple[Path, ...] = ()


def parse_file_evidence_bundles(
    data: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> tuple[ReferenceEvidenceBundleRecord, ...]:
    """Parse additive typed evidence bundles from a fixture row."""

    raw_bundles = data.get("evidence_bundles", ())
    if raw_bundles in (None, ""):
        return ()
    if not isinstance(raw_bundles, list | tuple):
        raise ValueError("evidence_bundles must be a list")
    bundles: list[ReferenceEvidenceBundleRecord] = []
    for bundle_index, raw_bundle in enumerate(raw_bundles):
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("evidence bundle entries must be objects")
        raw_artifacts = raw_bundle.get("artifacts", ())
        if not isinstance(raw_artifacts, list | tuple):
            raise ValueError("evidence bundle artifacts must be a list")
        artifacts: list[ReferenceEvidenceArtifactRecord] = []
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise ValueError("evidence artifact entries must be objects")
            artifact_path = Path(str(raw_artifact.get("path", "")))
            if base_dir is not None and not artifact_path.is_absolute():
                artifact_path = base_dir / artifact_path
            artifacts.append(
                ReferenceEvidenceArtifactRecord(
                    role=str(raw_artifact.get("role", "")),
                    kind=str(raw_artifact.get("kind", "")),
                    path=artifact_path,
                    stage=str(raw_artifact.get("stage", "dirty")),
                    required=bool(raw_artifact.get("required", True)),
                )
            )
        bundles.append(
            ReferenceEvidenceBundleRecord(
                bundle_id=str(raw_bundle.get("bundle_id", f"bundle-{bundle_index}")),
                evidence_kind=str(raw_bundle.get("evidence_kind", "")),
                role_policy=str(raw_bundle.get("role_policy", "named-artifacts")),
                artifacts=tuple(artifacts),
                section_plane_metadata=raw_bundle.get(
                    "section_plane_metadata",
                    raw_bundle.get("section_plane", {}),
                )
                or {},
            )
        )
    return tuple(bundles)


def load_database_evidence_bundles(
    payload: str | list[object] | tuple[object, ...] | None,
    *,
    base_dir: Path | None = None,
) -> tuple[ReferenceEvidenceBundleRecord, ...]:
    """Hydrate database-stored evidence bundle JSON into file-compatible records."""

    if payload in (None, ""):
        return ()
    raw_payload: object
    if isinstance(payload, str):
        raw_payload = json.loads(payload)
    else:
        raw_payload = payload
    if not isinstance(raw_payload, list | tuple):
        raise ValueError("database evidence_bundles must be a JSON array")
    return parse_file_evidence_bundles({"evidence_bundles": raw_payload}, base_dir=base_dir)


def serialize_database_evidence_bundles(
    bundles: Iterable[ReferenceEvidenceBundleRecord],
    *,
    base_dir: Path,
) -> str:
    """Serialize evidence bundles for database storage using fixture-relative paths."""

    payload = []
    for bundle in bundles:
        payload.append(
            {
                "bundle_id": bundle.bundle_id,
                "evidence_kind": bundle.evidence_kind,
                "role_policy": bundle.role_policy,
                "section_plane_metadata": dict(bundle.section_plane_metadata),
                "artifacts": [
                    {
                        "role": artifact.role,
                        "kind": artifact.kind,
                        "path": _path_for_fixture_storage(artifact.path, base_dir=base_dir),
                        "stage": artifact.stage,
                        "required": artifact.required,
                    }
                    for artifact in bundle.artifacts
                ],
            }
        )
    return json.dumps(payload, sort_keys=True)


def validate_section_evidence_roles(
    bundle: ReferenceEvidenceBundleRecord,
    *,
    required_roles: tuple[str, ...] = ("expected", "actual", "diff"),
) -> SectionEvidenceContractRecord:
    """Validate section evidence roles and section plane metadata without opening artifacts."""

    present_roles = tuple(dict.fromkeys(artifact.role for artifact in bundle.artifacts))
    missing_roles = tuple(role for role in required_roles if role not in present_roles)
    diagnostics: list[str] = []
    plane = dict(bundle.section_plane_metadata)
    if not plane:
        diagnostics.append("missing-section-plane-metadata")
    else:
        if "origin" not in plane:
            diagnostics.append("missing-section-plane-origin")
        if "normal" not in plane:
            diagnostics.append("missing-section-plane-normal")
    return SectionEvidenceContractRecord(
        bundle_id=bundle.bundle_id,
        required_roles=required_roles,
        present_roles=present_roles,
        missing_roles=missing_roles,
        section_plane_metadata=plane,
        diagnostics=tuple(diagnostics),
    )


def resolve_dirty_gold_evidence_paths(
    dirty_root: Path,
    gold_root: Path,
    *,
    fixture_stem: str,
    roles: tuple[str, ...] = ("expected", "actual", "diff"),
    suffix: str = ".png",
) -> EvidencePathSet:
    """Resolve deterministic dirty/gold evidence paths without touching payloads."""

    if not fixture_stem:
        raise ValueError("fixture_stem must not be empty")
    if "/" in fixture_stem or ".." in fixture_stem.split("/"):
        raise ValueError("fixture_stem must be a safe relative stem")
    dirty_paths = {
        role: Path(dirty_root) / f"{fixture_stem}-{role}{suffix}"
        for role in roles
    }
    gold_paths = {
        role: Path(gold_root) / f"{fixture_stem}-{role}{suffix}"
        for role in roles
    }
    return EvidencePathSet(dirty_paths=dirty_paths, gold_paths=gold_paths)


def build_section_bundle_fixture_record(
    *,
    bundle_id: str,
    evidence_kind: str,
    artifact_paths: Mapping[str, Path],
    section_plane_metadata: Mapping[str, Any],
    stage: str = "dirty",
) -> ReferenceEvidenceBundleRecord:
    """Build a typed section evidence bundle from generated artifact paths."""

    artifacts = tuple(
        ReferenceEvidenceArtifactRecord(
            role=role,
            kind="image/png",
            path=path,
            stage=stage,
            required=True,
        )
        for role, path in artifact_paths.items()
    )
    bundle = ReferenceEvidenceBundleRecord(
        bundle_id=bundle_id,
        evidence_kind=evidence_kind,
        artifacts=artifacts,
        section_plane_metadata=section_plane_metadata,
    )
    contract = validate_section_evidence_roles(bundle)
    if not contract.valid:
        missing = ", ".join(contract.missing_roles + contract.diagnostics)
        raise ValueError(f"section evidence bundle is incomplete: {missing}")
    return bundle


def validate_evidence_artifact_path(
    artifact: ReferenceEvidenceArtifactRecord,
    *,
    fixture_id: str | None = None,
    allowed_root: Path | None = None,
) -> SourceValidationDiagnostic | None:
    """Validate one evidence artifact path without loading artifact payloads."""

    artifact_path = artifact.path
    if allowed_root is not None:
        root = allowed_root.resolve()
        try:
            artifact_path.resolve().relative_to(root)
        except ValueError:
            return SourceValidationDiagnostic(
                "evidence-artifact-outside-root",
                "evidence artifact path is outside the configured reference root",
                fixture_id,
            )
    if not artifact_path.exists():
        if not artifact.required:
            return None
        return SourceValidationDiagnostic(
            "missing-evidence-artifact",
            sanitize_error_text(str(artifact_path)),
            fixture_id,
        )
    if not artifact_path.is_file():
        return SourceValidationDiagnostic("evidence-artifact-not-file", artifact_path.name, fixture_id)
    return None


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
    for artifact_path in record.artifact_paths:
        if allowed_root is not None:
            root = allowed_root.resolve()
            try:
                artifact_path.resolve().relative_to(root)
            except ValueError:
                diagnostics.append(
                    SourceValidationDiagnostic(
                        "artifact-outside-root",
                        "artifact path is outside the configured reference root",
                        record.fixture_id,
                    )
                )
        if not artifact_path.exists():
            diagnostics.append(
                SourceValidationDiagnostic(
                    "missing-artifact",
                    sanitize_error_text(str(artifact_path)),
                    record.fixture_id,
                )
            )
        elif not artifact_path.is_file():
            diagnostics.append(
                SourceValidationDiagnostic("artifact-not-file", artifact_path.name, record.fixture_id)
            )
    for bundle in record.evidence_bundles:
        if not bundle.artifacts:
            diagnostics.append(
                SourceValidationDiagnostic(
                    "evidence-bundle-missing-artifacts",
                    bundle.bundle_id,
                    record.fixture_id,
                )
            )
        for artifact in bundle.artifacts:
            diagnostic = validate_evidence_artifact_path(
                artifact,
                fixture_id=record.fixture_id,
                allowed_root=allowed_root,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        if "section" in bundle.evidence_kind:
            section_contract = validate_section_evidence_roles(bundle)
            for missing_role in section_contract.missing_roles:
                diagnostics.append(
                    SourceValidationDiagnostic(
                        "section-evidence-missing-role",
                        f"{bundle.bundle_id}:{missing_role}",
                        record.fixture_id,
                    )
                )
            for section_diagnostic in section_contract.diagnostics:
                diagnostics.append(
                    SourceValidationDiagnostic(
                        "section-evidence-invalid-plane",
                        f"{bundle.bundle_id}:{section_diagnostic}",
                        record.fixture_id,
                    )
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
        artifact_paths = data.get("artifact_paths")
        if isinstance(artifact_paths, str) and artifact_paths:
            data["artifact_paths"] = json.loads(artifact_paths)
        evidence_bundles = data.get("evidence_bundles")
        if evidence_bundles:
            data["evidence_bundles"] = (
                json.loads(evidence_bundles)
                if isinstance(evidence_bundles, str)
                else evidence_bundles
            )
        records.append(data)
    return _records_from_mappings(records, base_dir=path.parent, allowed_root=path.parent)


def approve_reference_artifacts(record: ReviewSourceModelRecord) -> ReferenceReviewStatusWriteResult:
    """Move dirty reference artifacts to matching gold paths."""

    gold_paths: list[Path] = []
    diagnostics: list[SourceValidationDiagnostic] = []
    for artifact_path in record.artifact_paths:
        gold_path = _gold_path_for_dirty_artifact(artifact_path)
        if gold_path is None:
            diagnostics.append(
                SourceValidationDiagnostic(
                    "artifact-not-under-dirty-root",
                    sanitize_error_text(str(artifact_path)),
                    record.fixture_id,
                )
            )
            continue
        if not artifact_path.exists():
            diagnostics.append(
                SourceValidationDiagnostic(
                    "missing-artifact",
                    sanitize_error_text(str(artifact_path)),
                    record.fixture_id,
                )
            )
            continue
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        if gold_path.exists():
            gold_path.unlink()
        shutil.move(str(artifact_path), str(gold_path))
        gold_paths.append(gold_path)
    if diagnostics:
        return ReferenceReviewStatusWriteResult(False, tuple(diagnostics), tuple(gold_paths))
    return ReferenceReviewStatusWriteResult(True, artifact_paths=tuple(gold_paths))


def update_fixture_review_status_in_file(
    path: Path,
    *,
    fixture_id: str,
    status: ReferenceReviewStatus | str,
    artifact_paths: tuple[Path, ...] | None = None,
) -> ReferenceReviewStatusWriteResult:
    """Persist a fixture review status into a JSON fixture file."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("invalid-fixture-file", str(exc), fixture_id),),
        )
    rows = payload.get("fixtures", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("invalid-fixture-file", "fixture rows are not editable", fixture_id),),
        )
    row = next(
        (item for item in rows if isinstance(item, dict) and item.get("fixture_id") == fixture_id),
        None,
    )
    if row is None:
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("missing-fixture-record", fixture_id, fixture_id),),
        )
    row["review_status"] = ReferenceReviewStatus(status).value
    if artifact_paths is not None:
        row["artifact_paths"] = [
            _path_for_fixture_storage(artifact_path, base_dir=path.parent)
            for artifact_path in artifact_paths
        ]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return ReferenceReviewStatusWriteResult(True, artifact_paths=tuple(artifact_paths or ()))


def update_fixture_notes_in_file(
    path: Path,
    *,
    fixture_id: str,
    notes: str,
) -> ReferenceReviewStatusWriteResult:
    """Persist review notes into a JSON fixture file."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("invalid-fixture-file", str(exc), fixture_id),),
        )
    rows = payload.get("fixtures", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("invalid-fixture-file", "fixture rows are not editable", fixture_id),),
        )
    row = next(
        (item for item in rows if isinstance(item, dict) and item.get("fixture_id") == fixture_id),
        None,
    )
    if row is None:
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("missing-fixture-record", fixture_id, fixture_id),),
        )
    row["notes"] = notes
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return ReferenceReviewStatusWriteResult(True)


def update_fixture_review_status_in_database(
    path: Path,
    *,
    fixture_id: str,
    status: ReferenceReviewStatus | str,
    artifact_paths: tuple[Path, ...] | None = None,
) -> ReferenceReviewStatusWriteResult:
    """Persist a fixture review status into a SQLite review_sources table."""

    path = Path(path)
    try:
        with sqlite3.connect(path) as connection:
            columns = {
                row[1]
                for row in connection.execute("pragma table_info(review_sources)").fetchall()
            }
            if "review_status" not in columns:
                connection.execute("alter table review_sources add column review_status text")
            if artifact_paths is not None and "artifact_paths" not in columns:
                connection.execute("alter table review_sources add column artifact_paths text")
            values: list[object] = [ReferenceReviewStatus(status).value]
            assignments = ["review_status = ?"]
            if artifact_paths is not None:
                assignments.append("artifact_paths = ?")
                values.append(
                    json.dumps(
                        [
                            _path_for_fixture_storage(artifact_path, base_dir=path.parent)
                            for artifact_path in artifact_paths
                        ]
                    )
                )
            values.append(fixture_id)
            cursor = connection.execute(
                f"update review_sources set {', '.join(assignments)} where fixture_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                return ReferenceReviewStatusWriteResult(
                    False,
                    (SourceValidationDiagnostic("missing-fixture-record", fixture_id, fixture_id),),
                )
    except Exception as exc:
        return ReferenceReviewStatusWriteResult(
            False,
            (SourceValidationDiagnostic("invalid-fixture-database", str(exc), fixture_id),),
        )
    return ReferenceReviewStatusWriteResult(True, artifact_paths=tuple(artifact_paths or ()))


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
    purpose: str | None
    methodology: str | None
    render_description: str | None
    parameters: tuple[EntrypointParameterRecord, ...] = ()
    artifact_display_paths: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "feature_name": self.feature_name,
            "source_display_path": self.source_display_path,
            "entrypoint": self.entrypoint,
            "expected_output": self.expected_output,
            "description": self.description,
            "purpose": self.purpose,
            "methodology": self.methodology,
            "render_description": self.render_description,
            "parameters": [
                {"name": item.name, "value": item.value} for item in self.parameters
            ],
            "artifact_display_paths": list(self.artifact_display_paths),
        }


def build_review_context_payload(record: ReviewSourceModelRecord) -> ReviewContextPayload:
    return ReviewContextPayload(
        fixture_id=record.fixture_id,
        feature_name=record.feature_name,
        source_display_path=record.identity.display_path,
        entrypoint=record.entrypoint,
        expected_output=record.expected_output,
        description=record.description,
        purpose=record.purpose,
        methodology=record.methodology,
        render_description=record.render_description,
        parameters=record.parameters,
        artifact_display_paths=tuple(path.name for path in record.artifact_paths),
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


def _gold_path_for_dirty_artifact(path: Path) -> Path | None:
    parts = path.parts
    try:
        dirty_index = parts.index("dirty")
    except ValueError:
        return None
    return Path(*parts[:dirty_index], "gold", *parts[dirty_index + 1 :])


def _path_for_fixture_storage(path: Path, *, base_dir: Path) -> str:
    path = Path(path)
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
