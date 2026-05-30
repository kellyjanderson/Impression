from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from impression.modeling.path3d import Path3D
from impression.modeling.surface import (
    BSplineSurfacePatch,
    DisplacementSurfacePatch,
    HeightmapSurfacePatch,
    IMPLICIT_FIELD_NODE_KINDS,
    ImplicitFieldNode,
    ImplicitFieldSafetyPolicy,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    PATCH_FAMILY_CAPABILITY_MATRIX,
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SUPPORTED_SURFACE_PATCH_FAMILIES,
    SubdivisionCrease,
    SubdivisionSurfacePatch,
    SweepSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBoundaryRef,
    SurfaceBody,
    SurfacePatch,
    SurfaceSeam,
    SurfaceShell,
    TrimLoop,
)

IMPRESS_FORMAT = "impress"
CURRENT_IMPRESS_SCHEMA_VERSION = "1.0"
DEFAULT_IMPRESS_LENGTH_UNIT = "unitless"
SUPPORTED_IMPRESS_LENGTH_UNITS = frozenset({"unitless", "mm", "cm", "m", "in", "ft"})
IMPRESS_DIAGNOSTIC_METADATA_FIELDS = frozenset({"code", "message", "path", "severity"})
_PATCH_KIND_FAMILIES = {
    "PlanarSurfacePatch": "planar",
    "RuledSurfacePatch": "ruled",
    "RevolutionSurfacePatch": "revolution",
    "BSplineSurfacePatch": "bspline",
    "NURBSSurfacePatch": "nurbs",
    "SweepSurfacePatch": "sweep",
    "SubdivisionSurfacePatch": "subdivision",
    "ImplicitSurfacePatch": "implicit",
    "HeightmapSurfacePatch": "heightmap",
    "DisplacementSurfacePatch": "displacement",
}
_ANALYTIC_PATCH_PAYLOAD_VERSION = 1
_SPLINE_PATCH_PAYLOAD_VERSION = 1
_SWEEP_PATCH_PAYLOAD_VERSION = 1
_SUBDIVISION_PATCH_PAYLOAD_VERSION = 1
_IMPLICIT_PATCH_PAYLOAD_VERSION = 1
_SAMPLED_PATCH_PAYLOAD_VERSION = 1
_PATCH_FAMILY_PAYLOAD_VERSIONS = {
    "PlanarSurfacePatch": _ANALYTIC_PATCH_PAYLOAD_VERSION,
    "RuledSurfacePatch": _ANALYTIC_PATCH_PAYLOAD_VERSION,
    "RevolutionSurfacePatch": _ANALYTIC_PATCH_PAYLOAD_VERSION,
    "BSplineSurfacePatch": _SPLINE_PATCH_PAYLOAD_VERSION,
    "NURBSSurfacePatch": _SPLINE_PATCH_PAYLOAD_VERSION,
    "SweepSurfacePatch": _SWEEP_PATCH_PAYLOAD_VERSION,
    "SubdivisionSurfacePatch": _SUBDIVISION_PATCH_PAYLOAD_VERSION,
    "ImplicitSurfacePatch": _IMPLICIT_PATCH_PAYLOAD_VERSION,
    "HeightmapSurfacePatch": _SAMPLED_PATCH_PAYLOAD_VERSION,
    "DisplacementSurfacePatch": _SAMPLED_PATCH_PAYLOAD_VERSION,
}


@dataclass(frozen=True)
class ImpressPatchCodecCoverageRecord:
    """Static `.impress` codec coverage for one surface patch family."""

    family: str
    patch_kind: str | None
    encode_supported: bool
    decode_supported: bool
    required_for_available: bool
    diagnostic: str = ""

    @property
    def covered(self) -> bool:
        return self.encode_supported and self.decode_supported


@dataclass(frozen=True)
class ImpressWholeStoreFixtureCoverageReport:
    """Deterministic coverage report for a whole-store `.impress` fixture."""

    required_families: tuple[str, ...]
    covered_families: tuple[str, ...]
    covered_store_areas: tuple[str, ...]
    missing_payloads: tuple[str, ...]
    mesh_truth_present: bool

    @property
    def covered(self) -> bool:
        return not self.missing_payloads and not self.mesh_truth_present

    @property
    def diagnostic(self) -> str:
        problems = list(self.missing_payloads)
        if self.mesh_truth_present:
            problems.append("mesh truth payload")
        if not problems:
            return "Whole-store `.impress` fixture coverage is complete."
        return "Whole-store `.impress` fixture coverage is incomplete: " + ", ".join(problems)


@dataclass(frozen=True)
class SurfacePatchBasePayload:
    """Decoded base fields shared by every `.impress` surface patch family."""

    kind: str
    family: str
    domain: ParameterDomain
    capability_flags: frozenset[str]
    trim_loops: tuple[TrimLoop, ...]
    transform_matrix: np.ndarray
    metadata: dict[str, object]
    geometry: Mapping[str, object]
    stable_identity: str | None = None

    def constructor_kwargs(self) -> dict[str, object]:
        """Return constructor arguments owned by the shared SurfacePatch base class."""

        return {
            "family": self.family,
            "domain": self.domain,
            "capability_flags": self.capability_flags,
            "trim_loops": self.trim_loops,
            "transform_matrix": self.transform_matrix,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SurfacePatchFamilyDispatchRecord:
    """Allow-listed `.impress` patch family dispatch target."""

    kind: str
    family: str
    payload_version: int


@dataclass(frozen=True)
class ImplicitCSGImpressPayloadRecord:
    """Inspectable `.impress` payload facts for an implicit CSG result body."""

    operation: str
    body_id: str
    patch_id: str
    field_root_kind: str
    source_operand_ids: tuple[str, ...]
    no_mesh_fallback: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "body_id": self.body_id,
            "patch_id": self.patch_id,
            "field_root_kind": self.field_root_kind,
            "source_operand_ids": self.source_operand_ids,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class ImplicitCSGImpressRoundTripDiagnostic:
    """Round-trip verifier result for implicit CSG `.impress` payloads."""

    supported: bool
    code: str
    message: str
    before: ImplicitCSGImpressPayloadRecord | None = None
    after: ImplicitCSGImpressPayloadRecord | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "code": self.code,
            "message": self.message,
            "before": None if self.before is None else self.before.canonical_payload(),
            "after": None if self.after is None else self.after.canonical_payload(),
        }


@dataclass(frozen=True)
class HeightmapCSGImpressPayloadRecord:
    """Inspectable `.impress` payload facts for a heightmap CSG result body."""

    operation: str
    body_id: str
    patch_id: str
    sample_shape: tuple[int, int]
    resample_kernel: str
    lossiness: str
    source_operand_ids: tuple[str, ...]
    no_mesh_fallback: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "body_id": self.body_id,
            "patch_id": self.patch_id,
            "sample_shape": self.sample_shape,
            "resample_kernel": self.resample_kernel,
            "lossiness": self.lossiness,
            "source_operand_ids": self.source_operand_ids,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapCSGImpressRoundTripDiagnostic:
    """Round-trip verifier result for heightmap CSG `.impress` payloads."""

    supported: bool
    code: str
    message: str
    before: HeightmapCSGImpressPayloadRecord | None = None
    after: HeightmapCSGImpressPayloadRecord | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "code": self.code,
            "message": self.message,
            "before": None if self.before is None else self.before.canonical_payload(),
            "after": None if self.after is None else self.after.canonical_payload(),
        }


_PATCH_FAMILY_DISPATCH = {
    kind: SurfacePatchFamilyDispatchRecord(
        kind=kind,
        family=family,
        payload_version=_PATCH_FAMILY_PAYLOAD_VERSIONS[kind],
    )
    for kind, family in _PATCH_KIND_FAMILIES.items()
}


class ImpressFormatError(ValueError):
    """Base error for invalid `.impress` document roots."""


class UnsupportedImpressSchemaVersion(ImpressFormatError):
    """Raised when an `.impress` document uses an unsupported schema."""


def inspect_impress_patch_codec_coverage() -> tuple[ImpressPatchCodecCoverageRecord, ...]:
    """Return static codec coverage records for all supported surface families."""

    family_to_kind = {family: kind for kind, family in _PATCH_KIND_FAMILIES.items()}
    records: list[ImpressPatchCodecCoverageRecord] = []
    for family in SUPPORTED_SURFACE_PATCH_FAMILIES:
        kind = family_to_kind.get(family)
        capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(family)
        required = capability is not None and capability.support_phase == "available"
        covered = kind is not None
        diagnostic = "" if covered else f"No `.impress` patch codec is registered for family '{family}'."
        records.append(
            ImpressPatchCodecCoverageRecord(
                family=family,
                patch_kind=kind,
                encode_supported=covered,
                decode_supported=covered,
                required_for_available=required,
                diagnostic=diagnostic,
            )
        )
    return tuple(records)


def assert_impress_patch_codec_coverage_for_available_families() -> tuple[ImpressPatchCodecCoverageRecord, ...]:
    """Return codec coverage or raise if an available family lacks a codec."""

    records = inspect_impress_patch_codec_coverage()
    missing = [record for record in records if record.required_for_available and not record.covered]
    if missing:
        joined = ", ".join(record.family for record in missing)
        raise ImpressFormatError(f"Missing `.impress` patch codec coverage for available families: {joined}")
    return records


def inspect_impress_whole_store_fixture_coverage(
    payload: Mapping[str, object],
) -> ImpressWholeStoreFixtureCoverageReport:
    """Inspect whether a fixture exercises the complete surface-native store shape."""

    required_families = tuple(sorted(record.family for record in inspect_impress_patch_codec_coverage() if record.covered))
    missing: list[str] = []
    covered_store_areas: list[str] = []

    body_store = payload.get("body_store") if isinstance(payload, Mapping) else None
    bodies = payload.get("bodies") if isinstance(payload, Mapping) else None
    patches = payload.get("patches") if isinstance(payload, Mapping) else None
    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}

    if isinstance(body_store, Mapping):
        covered_store_areas.append("body_store")
        body_entries = body_store.get("bodies")
        if not isinstance(body_entries, Sequence) or isinstance(body_entries, (str, bytes)) or not body_entries:
            missing.append("body_store.bodies")
        elif not all(isinstance(entry, Mapping) and entry.get("body_id") and entry.get("stable_identity") for entry in body_entries):
            missing.append("body_store.identity")
    else:
        missing.append("body_store")

    if isinstance(bodies, Mapping) and bodies:
        covered_store_areas.append("bodies")
        if not _payload_contains_body_area(bodies, "shells"):
            missing.append("bodies.shells")
        if not _payload_contains_body_area(bodies, "seams"):
            missing.append("bodies.shells.seams")
        if not _payload_contains_body_area(bodies, "adjacency"):
            missing.append("bodies.shells.adjacency")
    else:
        missing.append("bodies")

    covered_families: tuple[str, ...] = ()
    if isinstance(patches, Mapping) and patches:
        covered_store_areas.append("patches")
        families = {
            str(patch_payload.get("family"))
            for patch_payload in patches.values()
            if isinstance(patch_payload, Mapping) and patch_payload.get("family")
        }
        covered_families = tuple(sorted(families))
        for family in required_families:
            if family not in families:
                missing.append(f"patches.family[{family}]")
        if not any(
            isinstance(patch_payload, Mapping) and patch_payload.get("trim_loops")
            for patch_payload in patches.values()
        ):
            missing.append("patches.trim_loops")
        if not all(
            isinstance(patch_payload, Mapping) and patch_payload.get("stable_identity")
            for patch_payload in patches.values()
        ):
            missing.append("patches.identity")
    else:
        missing.append("patches")

    if not isinstance(metadata, Mapping):
        missing.append("metadata")
    else:
        for key in ("topology_rails", "lifecycle_records", "operation_provenance"):
            if key not in metadata:
                missing.append(f"metadata.{key}")

    mesh_truth_present = _payload_contains_key(payload, "mesh")
    return ImpressWholeStoreFixtureCoverageReport(
        required_families=required_families,
        covered_families=covered_families,
        covered_store_areas=tuple(sorted(set(covered_store_areas))),
        missing_payloads=tuple(sorted(set(missing))),
        mesh_truth_present=mesh_truth_present,
    )


def assert_impress_whole_store_fixture_coverage(
    payload: Mapping[str, object],
) -> ImpressWholeStoreFixtureCoverageReport:
    """Return whole-store coverage or raise with a deterministic diagnostic."""

    report = inspect_impress_whole_store_fixture_coverage(payload)
    if not report.covered:
        raise ImpressFormatError(report.diagnostic)
    return report


def inspect_impress_patch_family_dispatch() -> tuple[SurfacePatchFamilyDispatchRecord, ...]:
    """Return the allow-listed patch dispatch table used by `.impress` loading."""

    return tuple(_PATCH_FAMILY_DISPATCH[kind] for kind in sorted(_PATCH_FAMILY_DISPATCH))


@dataclass(frozen=True)
class InvalidSurfaceWrapperDiagnostic:
    """Diagnostic for mesh-derived wrappers that cannot be persisted as surface truth."""

    patch_identity: str
    patch_kind: str
    family: str
    producer: str
    reason: str

    def to_json_object(self) -> dict[str, object]:
        return {
            "patch_identity": self.patch_identity,
            "patch_kind": self.patch_kind,
            "family": self.family,
            "producer": self.producer,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImpressUnits:
    """Symbolic `.impress` file units.

    Unit conversion is intentionally outside the V1 root validation contract.
    """

    length: str = DEFAULT_IMPRESS_LENGTH_UNIT

    def to_json_object(self) -> dict[str, object]:
        return {"length": self.length}


@dataclass(frozen=True)
class ImpressBodyEntry:
    """One body entry inside a `.impress` surface body store."""

    body_id: str
    stable_identity: str
    body: SurfaceBody | None = None

    def to_json_object(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "body_id": self.body_id,
            "stable_identity": self.stable_identity,
        }
        if self.body is not None:
            payload["body_ref"] = self.body_id
        return payload


@dataclass(frozen=True)
class SurfaceBodyStore:
    """Validated `.impress` persisted surface body store."""

    bodies: tuple[ImpressBodyEntry, ...]

    def __post_init__(self) -> None:
        _validate_body_entries(self.bodies)

    def to_json_object(self) -> dict[str, object]:
        return {"bodies": [entry.to_json_object() for entry in self.bodies]}


@dataclass(frozen=True)
class ImpressDocumentRoot:
    """Minimal `.impress` document envelope."""

    schema_version: str = CURRENT_IMPRESS_SCHEMA_VERSION
    format: str = IMPRESS_FORMAT
    units: ImpressUnits = field(default_factory=ImpressUnits)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "units": self.units.to_json_object(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ImpressSaveOptions:
    """Deterministic `.impress` JSON writer options."""

    indent: int = 2
    ensure_ascii: bool = False
    trailing_newline: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.indent, int) or isinstance(self.indent, bool) or self.indent < 0:
            raise ImpressFormatError("ImpressSaveOptions.indent must be a non-negative integer.")
        if not isinstance(self.ensure_ascii, bool):
            raise ImpressFormatError("ImpressSaveOptions.ensure_ascii must be a boolean.")
        if not isinstance(self.trailing_newline, bool):
            raise ImpressFormatError("ImpressSaveOptions.trailing_newline must be a boolean.")


@dataclass(frozen=True)
class ImpressLoadResult:
    """Decoded `.impress` document result."""

    root: ImpressDocumentRoot
    body_store: SurfaceBodyStore
    bodies: tuple[SurfaceBody, ...]
    payload: Mapping[str, object]
    path: Path | None = None


def make_impress_document_root(
    *,
    schema_version: str = CURRENT_IMPRESS_SCHEMA_VERSION,
    units: ImpressUnits | Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ImpressDocumentRoot:
    """Create a validated minimal `.impress` document root."""

    normalized_units = validate_impress_units(units)
    root = ImpressDocumentRoot(
        schema_version=schema_version,
        units=normalized_units,
        metadata={} if metadata is None else dict(metadata),
    )
    validate_impress_document_root(root.to_json_object())
    return root


def make_impress_document_payload(
    bodies: Sequence[SurfaceBody],
    *,
    units: ImpressUnits | Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a validated `.impress` document payload from authored surface bodies."""

    body_store = make_surface_body_store(bodies)
    root = make_impress_document_root(units=units, metadata=metadata).to_json_object()
    patch_payloads: dict[str, object] = {}
    for body in bodies:
        if not isinstance(body, SurfaceBody):
            raise ImpressFormatError("`.impress` document payload can only contain SurfaceBody instances.")
        for shell in body.shells:
            for patch in shell.patches:
                patch_payloads.setdefault(patch.stable_identity, encode_surface_patch_payload(patch))
    root["body_store"] = body_store.to_json_object()
    root["patches"] = patch_payloads
    root["bodies"] = {
        entry.body_id: encode_surface_body_payload(entry.body)
        for entry in body_store.bodies
        if entry.body is not None
    }
    return root


def implicit_csg_impress_payload_record(body: SurfaceBody) -> ImplicitCSGImpressPayloadRecord:
    """Return persisted implicit CSG facts for a surface-native result body."""

    if not isinstance(body, SurfaceBody):
        raise ImpressFormatError("implicit_csg_impress_payload_record requires a SurfaceBody.")
    patches = tuple(patch for patch in body.iter_patches(world=True) if isinstance(patch, ImplicitSurfacePatch))
    if len(patches) != 1:
        raise ImpressFormatError("Implicit CSG `.impress` payloads require exactly one implicit result patch.")
    patch = patches[0]
    kernel = patch.metadata.get("kernel", {})
    if not isinstance(kernel, Mapping):
        raise ImpressFormatError("Implicit CSG result patch requires kernel metadata.")
    operation = str(kernel.get("operation", "")).strip()
    if not operation.startswith("implicit-csg-"):
        raise ImpressFormatError("Implicit CSG result patch metadata must declare an implicit-csg operation.")
    source_ids = kernel.get("source_operand_ids", ())
    if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
        raise ImpressFormatError("Implicit CSG result patch source_operand_ids must be an array.")
    source_operand_ids = tuple(str(source_id).strip() for source_id in source_ids)
    if any(not source_id for source_id in source_operand_ids):
        raise ImpressFormatError("Implicit CSG result patch source_operand_ids must contain non-empty strings.")
    no_mesh_fallback = bool(kernel.get("no_mesh_fallback", False))
    if not no_mesh_fallback:
        raise ImpressFormatError("Implicit CSG `.impress` payload must declare no_mesh_fallback=true.")
    return ImplicitCSGImpressPayloadRecord(
        operation=operation,
        body_id=body.stable_identity,
        patch_id=patch.stable_identity,
        field_root_kind=patch.field.kind,
        source_operand_ids=source_operand_ids,
        no_mesh_fallback=no_mesh_fallback,
    )


def encode_implicit_csg_impress_payload(
    result_or_body: object,
    *,
    units: ImpressUnits | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Encode an implicit CSG result body as a `.impress` document payload."""

    body = result_or_body.body if hasattr(result_or_body, "body") else result_or_body
    if body is None or not isinstance(body, SurfaceBody):
        raise ImpressFormatError("Implicit CSG `.impress` encoding requires a supported SurfaceBody result.")
    record = implicit_csg_impress_payload_record(body)
    return make_impress_document_payload(
        (body,),
        units=units,
        metadata={
            "surface_csg_payload": {
                "kind": "implicit-csg",
                "operation": record.operation,
                "patch_id": record.patch_id,
                "no_mesh_fallback": True,
            }
        },
    )


def verify_implicit_csg_impress_round_trip(result_or_body: object) -> ImplicitCSGImpressRoundTripDiagnostic:
    """Verify composed implicit CSG survives `.impress` JSON round-trip as surface truth."""

    try:
        body = result_or_body.body if hasattr(result_or_body, "body") else result_or_body
        if body is None or not isinstance(body, SurfaceBody):
            raise ImpressFormatError("Implicit CSG round-trip requires a supported SurfaceBody result.")
        before = implicit_csg_impress_payload_record(body)
        payload = encode_implicit_csg_impress_payload(body)
        loaded = loads_impress_json(dumps_impress_json(payload))
        if len(loaded.bodies) != 1:
            raise ImpressFormatError("Implicit CSG round-trip expected exactly one body.")
        after = implicit_csg_impress_payload_record(loaded.bodies[0])
        if before.canonical_payload() != after.canonical_payload():
            raise ImpressFormatError("Implicit CSG `.impress` round-trip changed persisted payload identity.")
    except Exception as exc:
        return ImplicitCSGImpressRoundTripDiagnostic(
            supported=False,
            code="implicit-csg-impress-roundtrip-failed",
            message=f"{exc}; no mesh fallback was attempted.",
        )
    return ImplicitCSGImpressRoundTripDiagnostic(
        supported=True,
        code="implicit-csg-impress-roundtrip-supported",
        message="Implicit CSG `.impress` payload round-tripped without mesh truth.",
        before=before,
        after=after,
    )


def heightmap_csg_impress_payload_record(body: SurfaceBody) -> HeightmapCSGImpressPayloadRecord:
    """Return persisted heightmap CSG facts for a surface-native result body."""

    if not isinstance(body, SurfaceBody):
        raise ImpressFormatError("heightmap_csg_impress_payload_record requires a SurfaceBody.")
    patches = tuple(patch for patch in body.iter_patches(world=True) if isinstance(patch, HeightmapSurfacePatch))
    if len(patches) != 1:
        raise ImpressFormatError("Heightmap CSG `.impress` payloads require exactly one heightmap result patch.")
    patch = patches[0]
    kernel = patch.metadata.get("kernel", {})
    if not isinstance(kernel, Mapping):
        raise ImpressFormatError("Heightmap CSG result patch requires kernel metadata.")
    composition = kernel.get("heightmap_csg_composition")
    if not isinstance(composition, Mapping):
        raise ImpressFormatError("Heightmap CSG result patch metadata must declare heightmap_csg_composition.")
    operation = str(composition.get("operation", "")).strip()
    if operation not in {"union", "difference", "intersection"}:
        raise ImpressFormatError("Heightmap CSG result patch metadata must declare a supported operation.")
    operand_ids = composition.get("operand_ids", ())
    if not isinstance(operand_ids, Sequence) or isinstance(operand_ids, (str, bytes)):
        raise ImpressFormatError("Heightmap CSG operand_ids must be an array.")
    source_operand_ids = tuple(str(source_id).strip() for source_id in operand_ids)
    if len(source_operand_ids) != 2 or any(not source_id for source_id in source_operand_ids):
        raise ImpressFormatError("Heightmap CSG operand_ids must contain two non-empty strings.")
    resample_kernel = str(composition.get("resample_kernel", "")).strip()
    if resample_kernel not in {"none", "bilinear"}:
        raise ImpressFormatError("Heightmap CSG resample_kernel must be none or bilinear.")
    lossiness = str(composition.get("lossiness", "")).strip()
    if lossiness not in {"lossless", "sampled-reconstruction"}:
        raise ImpressFormatError("Heightmap CSG lossiness must be lossless or sampled-reconstruction.")
    no_mesh_fallback = bool(composition.get("no_mesh_fallback", False))
    if not no_mesh_fallback:
        raise ImpressFormatError("Heightmap CSG `.impress` payload must declare no_mesh_fallback=true.")
    sample_shape = composition.get("sample_shape", patch.height_samples.shape)
    if not isinstance(sample_shape, Sequence) or isinstance(sample_shape, (str, bytes)) or len(sample_shape) != 2:
        raise ImpressFormatError("Heightmap CSG sample_shape must contain two integers.")
    normalized_shape = tuple(int(value) for value in sample_shape)
    if normalized_shape != tuple(int(value) for value in patch.height_samples.shape):
        raise ImpressFormatError("Heightmap CSG sample_shape must match persisted height_samples.")
    alignment = composition.get("alignment")
    if not isinstance(alignment, Mapping) or "clipping" not in alignment:
        raise ImpressFormatError("Heightmap CSG alignment metadata is required.")
    projection_frame = composition.get("projection_frame")
    if not isinstance(projection_frame, Mapping):
        raise ImpressFormatError("Heightmap CSG projection_frame metadata is required.")
    return HeightmapCSGImpressPayloadRecord(
        operation=operation,
        body_id=body.stable_identity,
        patch_id=patch.stable_identity,
        sample_shape=normalized_shape,
        resample_kernel=resample_kernel,
        lossiness=lossiness,
        source_operand_ids=source_operand_ids,
        no_mesh_fallback=no_mesh_fallback,
    )


def encode_heightmap_csg_impress_payload(
    result_or_body: object,
    *,
    units: ImpressUnits | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Encode a heightmap CSG result body as a `.impress` document payload."""

    body = result_or_body.body if hasattr(result_or_body, "body") else result_or_body
    if body is None or not isinstance(body, SurfaceBody):
        raise ImpressFormatError("Heightmap CSG `.impress` encoding requires a supported SurfaceBody result.")
    record = heightmap_csg_impress_payload_record(body)
    return make_impress_document_payload(
        (body,),
        units=units,
        metadata={
            "surface_csg_payload": {
                "kind": "heightmap-csg",
                "operation": record.operation,
                "patch_id": record.patch_id,
                "resample_kernel": record.resample_kernel,
                "lossiness": record.lossiness,
                "no_mesh_fallback": True,
            }
        },
    )


def verify_heightmap_csg_impress_round_trip(result_or_body: object) -> HeightmapCSGImpressRoundTripDiagnostic:
    """Verify heightmap CSG survives `.impress` JSON round-trip as surface truth."""

    try:
        body = result_or_body.body if hasattr(result_or_body, "body") else result_or_body
        if body is None or not isinstance(body, SurfaceBody):
            raise ImpressFormatError("Heightmap CSG round-trip requires a supported SurfaceBody result.")
        before = heightmap_csg_impress_payload_record(body)
        payload = encode_heightmap_csg_impress_payload(body)
        loaded = loads_impress_json(dumps_impress_json(payload))
        if len(loaded.bodies) != 1:
            raise ImpressFormatError("Heightmap CSG round-trip expected exactly one body.")
        after = heightmap_csg_impress_payload_record(loaded.bodies[0])
        if before.canonical_payload() != after.canonical_payload():
            raise ImpressFormatError("Heightmap CSG `.impress` round-trip changed persisted payload identity.")
    except Exception as exc:
        return HeightmapCSGImpressRoundTripDiagnostic(
            supported=False,
            code="heightmap-csg-impress-roundtrip-failed",
            message=f"{exc}; no mesh fallback was attempted.",
        )
    return HeightmapCSGImpressRoundTripDiagnostic(
        supported=True,
        code="heightmap-csg-impress-roundtrip-supported",
        message="Heightmap CSG `.impress` payload round-tripped without mesh truth.",
        before=before,
        after=after,
    )


def dumps_impress_json(
    payload: Mapping[str, object],
    *,
    options: ImpressSaveOptions | None = None,
) -> str:
    """Serialize a `.impress` payload as deterministic JSON."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("`.impress` payload must be an object.")
    validate_impress_document_root(payload)
    save_options = ImpressSaveOptions() if options is None else options
    if not isinstance(save_options, ImpressSaveOptions):
        raise ImpressFormatError("options must be an ImpressSaveOptions instance.")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=save_options.ensure_ascii,
            indent=save_options.indent,
            sort_keys=True,
            separators=(",", ": ") if save_options.indent is not None else (",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"Unable to serialize `.impress` payload: {exc}") from exc
    return f"{encoded}\n" if save_options.trailing_newline else encoded


def write_impress_json(
    payload: Mapping[str, object],
    path: str | Path,
    *,
    options: ImpressSaveOptions | None = None,
) -> Path:
    """Write a deterministic `.impress` JSON payload to a user-selected path."""

    output_path = Path(path)
    text = dumps_impress_json(payload, options=options)
    atomic_write_text(output_path, text)
    return output_path


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text through a sibling temporary file and atomically replace the destination."""

    output_path = Path(path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    except OSError as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ImpressFormatError(f"Unable to write `.impress` file {output_path}: {exc}") from exc
    return output_path


def loads_impress_json(text: str) -> ImpressLoadResult:
    """Read a `.impress` document from JSON text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImpressFormatError(f"Malformed `.impress` JSON: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ImpressFormatError("`.impress` JSON root must decode to an object.")
    return decode_impress_document_payload(payload)


def load_impress(path: str | Path) -> ImpressLoadResult:
    """Load a `.impress` document from a user-selected path."""

    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImpressFormatError(f"Unable to read `.impress` file {input_path}: {exc}") from exc
    result = loads_impress_json(text)
    return ImpressLoadResult(
        root=result.root,
        body_store=result.body_store,
        bodies=result.bodies,
        payload=result.payload,
        path=input_path,
    )


def decode_impress_document_payload(payload: Mapping[str, object]) -> ImpressLoadResult:
    """Decode a `.impress` document payload into surface-native bodies."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("`.impress` document payload must be an object.")
    unknown_keys = set(payload) - {"format", "schema_version", "units", "metadata", "body_store", "bodies", "patches"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported `.impress` document fields: {keys}.")

    root = validate_impress_document_root(payload)
    body_store = validate_surface_body_store(_required_mapping(payload.get("body_store"), "body_store"))
    body_payloads = _required_mapping(payload.get("bodies"), "bodies")
    patch_payloads = _required_mapping(payload.get("patches"), "patches")
    patches = {
        _validate_patch_identity_payload(patch_id): decode_surface_patch_payload(
            _required_mapping(patch_payload, f"patches[{patch_id!r}]")
        )
        for patch_id, patch_payload in patch_payloads.items()
    }
    for patch_id, patch in patches.items():
        if patch.stable_identity != patch_id:
            raise ImpressFormatError(f"Patch payload {patch_id!r} stable_identity does not match decoded patch.")

    expected_body_ids = {entry.body_id for entry in body_store.bodies}
    actual_body_ids = set(body_payloads)
    if actual_body_ids != expected_body_ids:
        raise ImpressFormatError("`.impress` bodies must exactly match SurfaceBodyStore body IDs.")

    referenced_patch_ids: set[str] = set()
    bodies: list[SurfaceBody] = []
    entries: list[ImpressBodyEntry] = []
    for entry in body_store.bodies:
        body_payload = _required_mapping(body_payloads.get(entry.body_id), f"bodies[{entry.body_id!r}]")
        shells = tuple(_decode_surface_shell_payload_from_document(shell_payload, patches, referenced_patch_ids) for shell_payload in _body_shell_payloads(body_payload))
        body = decode_surface_body_payload(body_payload, shells=shells)
        if body.stable_identity != entry.stable_identity:
            raise ImpressFormatError(f"Body payload {entry.body_id!r} stable_identity does not match SurfaceBodyStore.")
        bodies.append(body)
        entries.append(ImpressBodyEntry(body_id=entry.body_id, stable_identity=entry.stable_identity, body=body))

    if set(patches) != referenced_patch_ids:
        raise ImpressFormatError("`.impress` patch payloads must exactly match shell patch references.")

    return ImpressLoadResult(
        root=root,
        body_store=SurfaceBodyStore(tuple(entries)),
        bodies=tuple(bodies),
        payload=dict(payload),
    )


def save_impress(
    bodies: Sequence[SurfaceBody],
    path: str | Path,
    *,
    units: ImpressUnits | Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    options: ImpressSaveOptions | None = None,
) -> Path:
    """Persist authored surface bodies as deterministic `.impress` JSON."""

    payload = make_impress_document_payload(bodies, units=units, metadata=metadata)
    return write_impress_json(payload, path, options=options)


def validate_impress_units(units: ImpressUnits | Mapping[str, object] | None) -> ImpressUnits:
    """Validate and normalize symbolic `.impress` units."""

    if units is None:
        return ImpressUnits()
    if isinstance(units, ImpressUnits):
        unit_record = units
    elif isinstance(units, Mapping):
        unknown_keys = set(units) - {"length"}
        if unknown_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ImpressFormatError(f"Unsupported `.impress` unit fields: {keys}.")
        length = units.get("length", DEFAULT_IMPRESS_LENGTH_UNIT)
        if not isinstance(length, str):
            raise ImpressFormatError("`.impress` length unit must be a string.")
        unit_record = ImpressUnits(length=length)
    else:
        raise ImpressFormatError("`.impress` units must be an object when present.")

    if unit_record.length not in SUPPORTED_IMPRESS_LENGTH_UNITS:
        supported = ", ".join(sorted(SUPPORTED_IMPRESS_LENGTH_UNITS))
        raise ImpressFormatError(
            f"Unsupported `.impress` length unit {unit_record.length!r}; expected one of {supported}."
        )
    return unit_record


def make_surface_body_store(bodies: Sequence[SurfaceBody]) -> SurfaceBodyStore:
    """Create a deterministic store from authored surface bodies."""

    entries = []
    for index, body in enumerate(bodies):
        if not isinstance(body, SurfaceBody):
            raise ImpressFormatError("SurfaceBodyStore can only contain SurfaceBody instances.")
        entries.append(
            ImpressBodyEntry(
                body_id=f"body-{index + 1:04d}",
                stable_identity=body.stable_identity,
                body=body,
            )
        )
    return SurfaceBodyStore(tuple(entries))


def validate_surface_body_store(store: SurfaceBodyStore | Mapping[str, object]) -> SurfaceBodyStore:
    """Validate a `.impress` body store shape and stable identity policy."""

    if isinstance(store, SurfaceBodyStore):
        entries = store.bodies
    elif isinstance(store, Mapping):
        unknown_keys = set(store) - {"bodies"}
        if unknown_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ImpressFormatError(f"Unsupported SurfaceBodyStore fields: {keys}.")
        raw_entries = store.get("bodies")
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            raise ImpressFormatError("SurfaceBodyStore bodies must be an array.")
        entries = tuple(_validate_body_entry_payload(entry) for entry in raw_entries)
    else:
        raise ImpressFormatError("SurfaceBodyStore must be an object.")

    _validate_body_entries(entries)
    return SurfaceBodyStore(tuple(entries)) if not isinstance(store, SurfaceBodyStore) else store


def _validate_body_entries(entries: Sequence[ImpressBodyEntry]) -> None:
    if not entries:
        raise ImpressFormatError("SurfaceBodyStore requires at least one body entry.")

    body_ids: set[str] = set()
    identities: set[str] = set()
    for entry in entries:
        if not isinstance(entry, ImpressBodyEntry):
            raise ImpressFormatError("SurfaceBodyStore entries must be ImpressBodyEntry records.")
        if entry.body_id in body_ids:
            raise ImpressFormatError(f"Duplicate SurfaceBodyStore body_id {entry.body_id!r}.")
        if entry.stable_identity in identities:
            raise ImpressFormatError(f"Duplicate SurfaceBodyStore stable_identity {entry.stable_identity!r}.")
        if entry.body is not None and entry.body.stable_identity != entry.stable_identity:
            raise ImpressFormatError(f"Body entry {entry.body_id!r} stable_identity does not match its SurfaceBody.")
        body_ids.add(entry.body_id)
        identities.add(entry.stable_identity)


def _validate_body_entry_payload(entry: object) -> ImpressBodyEntry:
    if not isinstance(entry, Mapping):
        raise ImpressFormatError("SurfaceBodyStore body entries must be objects.")
    unknown_keys = set(entry) - {"body_id", "stable_identity", "body_ref"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceBodyStore body entry fields: {keys}.")
    body_id = entry.get("body_id")
    stable_identity = entry.get("stable_identity")
    if not isinstance(body_id, str) or not body_id:
        raise ImpressFormatError("SurfaceBodyStore body entries require a non-empty body_id.")
    if not isinstance(stable_identity, str) or not stable_identity:
        raise ImpressFormatError("SurfaceBodyStore body entries require a non-empty stable_identity.")
    body_ref = entry.get("body_ref", body_id)
    if body_ref != body_id:
        raise ImpressFormatError("SurfaceBodyStore body_ref must match body_id when present.")
    return ImpressBodyEntry(body_id=body_id, stable_identity=stable_identity)


def encode_trim_loop_payload(trim_loop: TrimLoop) -> dict[str, object]:
    """Encode a patch-local trim loop for `.impress` persistence."""

    if not isinstance(trim_loop, TrimLoop):
        raise ImpressFormatError("encode_trim_loop_payload requires a TrimLoop.")
    normalized = trim_loop.normalized()
    return {
        "category": normalized.category,
        "points_uv": _array_payload(normalized.points_uv),
        "clockwise": normalized.is_clockwise,
    }


def decode_trim_loop_payload(payload: Mapping[str, object]) -> TrimLoop:
    """Decode a `.impress` trim payload through the public TrimLoop constructor."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("TrimLoop payload must be an object.")
    unknown_keys = set(payload) - {"category", "points_uv", "clockwise"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported TrimLoop payload fields: {keys}.")
    category = payload.get("category")
    if not isinstance(category, str) or not category:
        raise ImpressFormatError("TrimLoop payload requires a non-empty category.")
    clockwise = payload.get("clockwise")
    if clockwise is not None and not isinstance(clockwise, bool):
        raise ImpressFormatError("TrimLoop clockwise must be a boolean when present.")
    try:
        trim_loop = TrimLoop(_validate_points2_payload(payload.get("points_uv"), "points_uv"), category=category)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc
    normalized = trim_loop.normalized()
    if clockwise is not None and normalized.is_clockwise != clockwise:
        raise ImpressFormatError("TrimLoop clockwise does not match the normalized category orientation.")
    return normalized


def encode_surface_boundary_ref_payload(boundary: SurfaceBoundaryRef) -> dict[str, object]:
    """Encode a seam boundary reference."""

    if not isinstance(boundary, SurfaceBoundaryRef):
        raise ImpressFormatError("encode_surface_boundary_ref_payload requires a SurfaceBoundaryRef.")
    return {
        "patch_index": boundary.patch_index,
        "boundary_id": boundary.boundary_id,
    }


def decode_surface_boundary_ref_payload(
    payload: Mapping[str, object],
    *,
    patch_count: int | None = None,
) -> SurfaceBoundaryRef:
    """Decode and optionally validate a seam boundary reference against loaded patches."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfaceBoundaryRef payload must be an object.")
    unknown_keys = set(payload) - {"patch_index", "boundary_id"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceBoundaryRef payload fields: {keys}.")
    patch_index = _validate_nonnegative_int_payload(payload.get("patch_index"), "patch_index")
    if patch_count is not None and patch_index >= patch_count:
        raise ImpressFormatError("SurfaceBoundaryRef references a patch index outside the loaded shell.")
    boundary_id = payload.get("boundary_id")
    if not isinstance(boundary_id, str) or not boundary_id.strip():
        raise ImpressFormatError("SurfaceBoundaryRef boundary_id must be a non-empty string.")
    try:
        return SurfaceBoundaryRef(patch_index, boundary_id)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def encode_surface_seam_payload(seam: SurfaceSeam) -> dict[str, object]:
    """Encode a `.impress` seam payload."""

    if not isinstance(seam, SurfaceSeam):
        raise ImpressFormatError("encode_surface_seam_payload requires a SurfaceSeam.")
    return {
        "seam_id": seam.seam_id,
        "boundaries": [encode_surface_boundary_ref_payload(boundary) for boundary in seam.boundaries],
        "continuity": seam.continuity,
        "metadata": dict(seam.metadata),
    }


def decode_surface_seam_payload(
    payload: Mapping[str, object],
    *,
    patch_count: int | None = None,
) -> SurfaceSeam:
    """Decode a `.impress` seam payload and validate boundary references when patch_count is provided."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfaceSeam payload must be an object.")
    unknown_keys = set(payload) - {"seam_id", "boundaries", "continuity", "metadata"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceSeam payload fields: {keys}.")
    seam_id = payload.get("seam_id")
    if not isinstance(seam_id, str) or not seam_id.strip():
        raise ImpressFormatError("SurfaceSeam seam_id must be a non-empty string.")
    boundary_payloads = payload.get("boundaries")
    if not isinstance(boundary_payloads, Sequence) or isinstance(boundary_payloads, (str, bytes)):
        raise ImpressFormatError("SurfaceSeam boundaries must be an array.")
    boundaries = tuple(
        decode_surface_boundary_ref_payload(boundary_payload, patch_count=patch_count)
        for boundary_payload in boundary_payloads
    )
    continuity = payload.get("continuity", "C0")
    if not isinstance(continuity, str) or not continuity.strip():
        raise ImpressFormatError("SurfaceSeam continuity must be a non-empty string.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ImpressFormatError("SurfaceSeam metadata must be an object.")
    try:
        return SurfaceSeam(seam_id, boundaries, continuity=continuity, metadata=dict(metadata))
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def encode_surface_adjacency_payload(record: SurfaceAdjacencyRecord) -> dict[str, object]:
    """Encode a `.impress` adjacency payload."""

    if not isinstance(record, SurfaceAdjacencyRecord):
        raise ImpressFormatError("encode_surface_adjacency_payload requires a SurfaceAdjacencyRecord.")
    return {
        "source": encode_surface_boundary_ref_payload(record.source),
        "target": None if record.target is None else encode_surface_boundary_ref_payload(record.target),
        "seam_id": record.seam_id,
        "continuity": record.continuity,
    }


def decode_surface_adjacency_payload(
    payload: Mapping[str, object],
    *,
    patch_count: int | None = None,
    seam_ids: Sequence[str] | None = None,
) -> SurfaceAdjacencyRecord:
    """Decode a `.impress` adjacency payload with optional loaded-shell reference validation."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfaceAdjacencyRecord payload must be an object.")
    unknown_keys = set(payload) - {"source", "target", "seam_id", "continuity"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceAdjacencyRecord payload fields: {keys}.")
    source = decode_surface_boundary_ref_payload(payload.get("source"), patch_count=patch_count)
    target_payload = payload.get("target")
    if target_payload is None:
        target = None
    else:
        target = decode_surface_boundary_ref_payload(target_payload, patch_count=patch_count)
    seam_id = payload.get("seam_id")
    if seam_id is not None:
        if not isinstance(seam_id, str) or not seam_id.strip():
            raise ImpressFormatError("SurfaceAdjacencyRecord seam_id must be a non-empty string when present.")
        if seam_ids is not None and seam_id not in set(seam_ids):
            raise ImpressFormatError("SurfaceAdjacencyRecord references an unknown seam_id.")
    continuity = payload.get("continuity", "C0")
    if not isinstance(continuity, str) or not continuity.strip():
        raise ImpressFormatError("SurfaceAdjacencyRecord continuity must be a non-empty string.")
    try:
        return SurfaceAdjacencyRecord(source=source, target=target, seam_id=seam_id, continuity=continuity)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def encode_surface_patch_payload(patch: SurfacePatch) -> dict[str, object]:
    """Encode base patch fields and supported family geometry for `.impress` persistence."""

    payload = encode_surface_patch_base_payload(patch)
    payload["geometry"] = _encode_patch_geometry_payload(patch)
    return payload


def encode_surface_patch_base_payload(patch: SurfacePatch) -> dict[str, object]:
    """Encode the base fields shared by every `.impress` surface patch family."""

    if not isinstance(patch, SurfacePatch):
        raise ImpressFormatError("encode_surface_patch_base_payload requires a SurfacePatch.")
    diagnostic = validate_surface_patch_serialization_guard(patch)
    if diagnostic is not None:
        raise ImpressFormatError(
            "Refusing to serialize mesh-derived surface wrapper as `.impress` surface truth: "
            f"{diagnostic.reason}."
        )
    _validate_patch_kind_family(type(patch).__name__, patch.family)
    return {
        "stable_identity": patch.stable_identity,
        "kind": type(patch).__name__,
        "family": patch.family,
        "domain": _encode_parameter_domain_payload(patch.domain),
        "capability_flags": sorted(patch.capability_flags),
        "transform_matrix": patch.transform_matrix.tolist(),
        "metadata": dict(patch.metadata),
        "trim_loops": [encode_trim_loop_payload(trim_loop) for trim_loop in patch.trim_loops],
    }


def decode_surface_patch_base_payload(payload: Mapping[str, object]) -> SurfacePatchBasePayload:
    """Decode shared patch fields and leave family geometry opaque for dispatch."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfacePatch payload must be an object.")
    unknown_keys = set(payload) - {
        "stable_identity",
        "kind",
        "family",
        "domain",
        "capability_flags",
        "transform_matrix",
        "metadata",
        "trim_loops",
        "geometry",
    }
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfacePatch payload fields: {keys}.")

    stable_identity = payload.get("stable_identity")
    if stable_identity is not None and (not isinstance(stable_identity, str) or not stable_identity):
        raise ImpressFormatError("SurfacePatch stable_identity must be a non-empty string when present.")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ImpressFormatError("SurfacePatch payload requires a non-empty kind.")
    family = payload.get("family")
    if not isinstance(family, str) or not family:
        raise ImpressFormatError("SurfacePatch payload requires a non-empty family.")
    _validate_patch_kind_family(kind, family)

    flags = payload.get("capability_flags", [])
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        raise ImpressFormatError("SurfacePatch capability_flags must be an array.")
    if not all(isinstance(flag, str) and flag for flag in flags):
        raise ImpressFormatError("SurfacePatch capability_flags must contain non-empty strings.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ImpressFormatError("SurfacePatch metadata must be an object.")

    trim_payloads = payload.get("trim_loops", [])
    if not isinstance(trim_payloads, Sequence) or isinstance(trim_payloads, (str, bytes)):
        raise ImpressFormatError("SurfacePatch trim_loops must be an array.")
    trim_loops = tuple(decode_trim_loop_payload(trim_payload) for trim_payload in trim_payloads)

    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ImpressFormatError("SurfacePatch geometry must be an object.")

    domain = _decode_parameter_domain_payload(payload.get("domain"))
    transform_matrix = _validate_matrix4_payload(payload.get("transform_matrix"))
    return SurfacePatchBasePayload(
        stable_identity=stable_identity,
        kind=kind,
        family=family,
        domain=domain,
        capability_flags=frozenset(flags),
        trim_loops=trim_loops,
        transform_matrix=transform_matrix,
        metadata=dict(metadata),
        geometry=geometry,
    )


def decode_surface_patch_payload(payload: Mapping[str, object]) -> SurfacePatch:
    """Decode a `.impress` patch payload through allow-listed public patch constructors."""

    base = decode_surface_patch_base_payload(payload)
    common = base.constructor_kwargs()
    geometry = base.geometry
    try:
        if base.kind == "PlanarSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={"payload_version", "origin", "u_axis", "v_axis"},
            )
            patch = PlanarSurfacePatch(
                **common,
                origin=_validate_vec3_payload(geometry.get("origin"), "origin"),
                u_axis=_validate_vec3_payload(geometry.get("u_axis"), "u_axis"),
                v_axis=_validate_vec3_payload(geometry.get("v_axis"), "v_axis"),
            )
        elif base.kind == "RuledSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={"payload_version", "start_curve", "end_curve"},
            )
            patch = RuledSurfacePatch(
                **common,
                start_curve=_validate_points3_payload(geometry.get("start_curve"), "start_curve"),
                end_curve=_validate_points3_payload(geometry.get("end_curve"), "end_curve"),
            )
        elif base.kind == "RevolutionSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "profile_curve",
                    "axis_origin",
                    "axis_direction",
                    "start_angle_deg",
                    "sweep_angle_deg",
                },
            )
            patch = RevolutionSurfacePatch(
                **common,
                profile_curve=_validate_points3_payload(geometry.get("profile_curve"), "profile_curve"),
                axis_origin=_validate_vec3_payload(geometry.get("axis_origin"), "axis_origin"),
                axis_direction=_validate_vec3_payload(geometry.get("axis_direction"), "axis_direction"),
                start_angle_deg=_validate_float_payload(geometry.get("start_angle_deg"), "start_angle_deg"),
                sweep_angle_deg=_validate_float_payload(geometry.get("sweep_angle_deg"), "sweep_angle_deg"),
            )
        elif base.kind == "BSplineSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={"payload_version", "degree_u", "degree_v", "knots_u", "knots_v", "control_net"},
                expected_payload_version=_SPLINE_PATCH_PAYLOAD_VERSION,
            )
            patch = BSplineSurfacePatch(
                **common,
                degree_u=_validate_positive_int_payload(geometry.get("degree_u"), "degree_u"),
                degree_v=_validate_positive_int_payload(geometry.get("degree_v"), "degree_v"),
                knots_u=_validate_float_tuple_payload(geometry.get("knots_u"), "knots_u"),
                knots_v=_validate_float_tuple_payload(geometry.get("knots_v"), "knots_v"),
                control_net=_validate_control_net3_payload(geometry.get("control_net"), "control_net"),
            )
        elif base.kind == "NURBSSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "degree_u",
                    "degree_v",
                    "knots_u",
                    "knots_v",
                    "control_net",
                    "weights",
                },
                expected_payload_version=_SPLINE_PATCH_PAYLOAD_VERSION,
            )
            patch = NURBSSurfacePatch(
                **common,
                degree_u=_validate_positive_int_payload(geometry.get("degree_u"), "degree_u"),
                degree_v=_validate_positive_int_payload(geometry.get("degree_v"), "degree_v"),
                knots_u=_validate_float_tuple_payload(geometry.get("knots_u"), "knots_u"),
                knots_v=_validate_float_tuple_payload(geometry.get("knots_v"), "knots_v"),
                control_net=_validate_control_net3_payload(geometry.get("control_net"), "control_net"),
                weights=_validate_weight_net_payload(geometry.get("weights"), "weights"),
            )
        elif base.kind == "SweepSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "profile_points_uv",
                    "path_points",
                    "frame_policy",
                    "profile_reference",
                    "path_reference",
                },
                expected_payload_version=_SWEEP_PATCH_PAYLOAD_VERSION,
            )
            patch = SweepSurfacePatch(
                **common,
                profile_points_uv=_validate_profile_points2_payload(geometry.get("profile_points_uv"), "profile_points_uv"),
                path=Path3D.from_points(_validate_points3_payload(geometry.get("path_points"), "path_points")),
                frame_policy=_validate_string_payload(geometry.get("frame_policy"), "frame_policy"),
                profile_reference=_validate_optional_string_payload(geometry.get("profile_reference"), "profile_reference"),
                path_reference=_validate_optional_string_payload(geometry.get("path_reference"), "path_reference"),
            )
        elif base.kind == "SubdivisionSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "scheme",
                    "subdivision_level",
                    "control_points",
                    "faces",
                    "creases",
                },
                expected_payload_version=_SUBDIVISION_PATCH_PAYLOAD_VERSION,
            )
            patch = SubdivisionSurfacePatch(
                **common,
                scheme=_validate_string_payload(geometry.get("scheme"), "scheme"),
                subdivision_level=_validate_nonnegative_int_payload(
                    geometry.get("subdivision_level"),
                    "subdivision_level",
                ),
                control_points=_validate_control_points3_payload(geometry.get("control_points"), "control_points"),
                faces=_validate_subdivision_faces_payload(geometry.get("faces"), "faces"),
                creases=tuple(_decode_subdivision_crease_payload(crease) for crease in _validate_array_payload(geometry.get("creases"), "creases")),
            )
        elif base.kind == "ImplicitSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={"payload_version", "field", "bounds"},
                expected_payload_version=_IMPLICIT_PATCH_PAYLOAD_VERSION,
            )
            patch = ImplicitSurfacePatch(
                **common,
                field=_decode_implicit_field_node_payload(geometry.get("field")),
                bounds=_validate_bounds3_payload(geometry.get("bounds"), "bounds"),
            )
        elif base.kind == "HeightmapSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "height_samples",
                    "alpha_mask",
                    "alpha_mode",
                    "xy_scale",
                    "center",
                    "height_scale",
                },
                expected_payload_version=_SAMPLED_PATCH_PAYLOAD_VERSION,
            )
            patch = HeightmapSurfacePatch(
                **common,
                height_samples=_validate_numeric_grid_payload(geometry.get("height_samples"), "height_samples"),
                alpha_mask=_validate_bool_grid_payload(geometry.get("alpha_mask"), "alpha_mask"),
                alpha_mode=_validate_string_payload(geometry.get("alpha_mode"), "alpha_mode"),
                xy_scale=_validate_positive_pair_payload(geometry.get("xy_scale"), "xy_scale"),
                center=_validate_vec3_payload(geometry.get("center"), "center"),
                height_scale=_validate_float_payload(geometry.get("height_scale"), "height_scale"),
            )
        elif base.kind == "DisplacementSurfacePatch":
            _validate_patch_geometry_fields(
                geometry,
                kind=base.kind,
                allowed_fields={
                    "payload_version",
                    "source_patch",
                    "displacement_samples",
                    "alpha_mask",
                    "alpha_mode",
                    "height_scale",
                    "direction",
                    "projection",
                    "plane",
                    "projection_bounds",
                },
                expected_payload_version=_SAMPLED_PATCH_PAYLOAD_VERSION,
            )
            patch = DisplacementSurfacePatch(
                **common,
                source_patch=decode_surface_patch_payload(_required_mapping(geometry.get("source_patch"), "source_patch")),
                displacement_samples=_validate_numeric_grid_payload(
                    geometry.get("displacement_samples"),
                    "displacement_samples",
                ),
                alpha_mask=_validate_bool_grid_payload(geometry.get("alpha_mask"), "alpha_mask"),
                alpha_mode=_validate_string_payload(geometry.get("alpha_mode"), "alpha_mode"),
                height_scale=_validate_float_payload(geometry.get("height_scale"), "height_scale"),
                direction=_validate_direction_payload(geometry.get("direction"), "direction"),
                projection=_validate_string_payload(geometry.get("projection"), "projection"),
                plane=_validate_string_payload(geometry.get("plane"), "plane"),
                projection_bounds=_validate_bounds2_payload(geometry.get("projection_bounds"), "projection_bounds"),
            )
        else:
            raise ImpressFormatError(f"Unsupported SurfacePatch kind {base.kind!r}.")
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc
    if base.stable_identity is not None and patch.stable_identity != base.stable_identity:
        raise ImpressFormatError("SurfacePatch stable_identity does not match decoded patch payload.")
    return patch


def validate_surface_patch_serialization_guard(patch: SurfacePatch) -> InvalidSurfaceWrapperDiagnostic | None:
    """Return a diagnostic when a patch is a mesh-derived wrapper, otherwise None."""

    if not isinstance(patch, SurfacePatch):
        raise ImpressFormatError("validate_surface_patch_serialization_guard requires a SurfacePatch.")
    metadata = patch.metadata
    kernel = metadata.get("kernel") if isinstance(metadata, Mapping) else None
    if not isinstance(kernel, Mapping):
        return None
    producer = kernel.get("producer")
    if producer == "heightmap" and "triangle_face_index" in kernel:
        return InvalidSurfaceWrapperDiagnostic(
            patch_identity=patch.stable_identity,
            patch_kind=type(patch).__name__,
            family=patch.family,
            producer=str(producer),
            reason="heightmap triangle wrappers are mesh compatibility data, not native surface payloads",
        )
    return None


def encode_surface_shell_payload(shell: SurfaceShell) -> dict[str, object]:
    """Encode container-level shell fields for `.impress` persistence."""

    if not isinstance(shell, SurfaceShell):
        raise ImpressFormatError("encode_surface_shell_payload requires a SurfaceShell.")
    return {
        "connected": shell.connected,
        "patch_count": shell.patch_count,
        "patches": [patch.stable_identity for patch in shell.patches],
        "transform_matrix": shell.transform_matrix.tolist(),
        "metadata": dict(shell.metadata),
        "seams": [encode_surface_seam_payload(seam) for seam in shell.seams],
        "adjacency": [encode_surface_adjacency_payload(record) for record in shell.adjacency],
    }


def decode_surface_shell_payload(payload: Mapping[str, object], *, patches: Sequence[SurfacePatch]) -> SurfaceShell:
    """Decode container-level shell fields through the public SurfaceShell constructor."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfaceShell payload must be an object.")
    unknown_keys = set(payload) - {"connected", "patch_count", "patches", "transform_matrix", "metadata", "seams", "adjacency"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceShell payload fields: {keys}.")

    connected = payload.get("connected", True)
    if not isinstance(connected, bool):
        raise ImpressFormatError("SurfaceShell connected must be a boolean.")

    patch_ids = payload.get("patches")
    if not isinstance(patch_ids, Sequence) or isinstance(patch_ids, (str, bytes)):
        raise ImpressFormatError("SurfaceShell patches must be an array of patch identities.")
    if not all(isinstance(patch_id, str) and patch_id for patch_id in patch_ids):
        raise ImpressFormatError("SurfaceShell patch identities must be non-empty strings.")

    patch_count = payload.get("patch_count", len(patch_ids))
    if not isinstance(patch_count, int) or patch_count < 1:
        raise ImpressFormatError("SurfaceShell patch_count must be a positive integer.")
    if patch_count != len(patch_ids) or patch_count != len(patches):
        raise ImpressFormatError("SurfaceShell patch_count must match encoded and decoded patch counts.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ImpressFormatError("SurfaceShell metadata must be an object.")

    seams = payload.get("seams", [])
    adjacency = payload.get("adjacency", [])
    if not isinstance(seams, Sequence) or isinstance(seams, (str, bytes)):
        raise ImpressFormatError("SurfaceShell seams must be an array.")
    if not isinstance(adjacency, Sequence) or isinstance(adjacency, (str, bytes)):
        raise ImpressFormatError("SurfaceShell adjacency must be an array.")
    decoded_seams = tuple(decode_surface_seam_payload(seam, patch_count=len(patches)) for seam in seams)
    seam_ids = tuple(seam.seam_id for seam in decoded_seams)
    decoded_adjacency = tuple(
        decode_surface_adjacency_payload(record, patch_count=len(patches), seam_ids=seam_ids) for record in adjacency
    )

    transform_matrix = _validate_matrix4_payload(payload.get("transform_matrix"))
    try:
        return SurfaceShell(
            tuple(patches),
            connected=connected,
            seams=decoded_seams,
            adjacency=decoded_adjacency,
            transform_matrix=transform_matrix,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def encode_surface_body_payload(body: SurfaceBody) -> dict[str, object]:
    """Encode container-level body fields for `.impress` persistence."""

    if not isinstance(body, SurfaceBody):
        raise ImpressFormatError("encode_surface_body_payload requires a SurfaceBody.")
    return {
        "shell_count": body.shell_count,
        "shells": [encode_surface_shell_payload(shell) for shell in body.shells],
        "transform_matrix": body.transform_matrix.tolist(),
        "metadata": dict(body.metadata),
    }


def decode_surface_body_payload(payload: Mapping[str, object], *, shells: Sequence[SurfaceShell]) -> SurfaceBody:
    """Decode container-level body fields through the public SurfaceBody constructor."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfaceBody payload must be an object.")
    unknown_keys = set(payload) - {"shell_count", "shells", "transform_matrix", "metadata"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfaceBody payload fields: {keys}.")

    shell_payloads = payload.get("shells")
    if not isinstance(shell_payloads, Sequence) or isinstance(shell_payloads, (str, bytes)):
        raise ImpressFormatError("SurfaceBody shells must be an array.")
    shell_count = payload.get("shell_count", len(shell_payloads))
    if not isinstance(shell_count, int) or shell_count < 1:
        raise ImpressFormatError("SurfaceBody shell_count must be a positive integer.")
    if shell_count != len(shell_payloads) or shell_count != len(shells):
        raise ImpressFormatError("SurfaceBody shell_count must match encoded and decoded shell counts.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ImpressFormatError("SurfaceBody metadata must be an object.")
    transform_matrix = _validate_matrix4_payload(payload.get("transform_matrix"))
    try:
        return SurfaceBody(tuple(shells), transform_matrix=transform_matrix, metadata=dict(metadata))
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def _validate_matrix4_payload(payload: object) -> np.ndarray:
    if payload is None:
        return np.eye(4, dtype=float)
    try:
        matrix = np.asarray(payload, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError("transform_matrix must be a 4x4 numeric matrix.") from exc
    if not np.all(np.isfinite(matrix)):
        raise ImpressFormatError("transform_matrix must contain only finite values.")
    return matrix


def _required_mapping(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ImpressFormatError(f"`.impress` {name} must be an object.")
    non_string_keys = [key for key in payload if not isinstance(key, str)]
    if non_string_keys:
        raise ImpressFormatError(f"`.impress` {name} keys must be strings.")
    return payload


def _payload_contains_key(payload: object, key: str) -> bool:
    if isinstance(payload, Mapping):
        return key in payload or any(_payload_contains_key(value, key) for value in payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return any(_payload_contains_key(value, key) for value in payload)
    return False


def _payload_contains_body_area(bodies: Mapping[str, object], area: str) -> bool:
    for body_payload in bodies.values():
        if not isinstance(body_payload, Mapping):
            continue
        shell_payloads = body_payload.get("shells")
        if area == "shells" and isinstance(shell_payloads, Sequence) and not isinstance(shell_payloads, (str, bytes)) and shell_payloads:
            return True
        if not isinstance(shell_payloads, Sequence) or isinstance(shell_payloads, (str, bytes)):
            continue
        for shell_payload in shell_payloads:
            if isinstance(shell_payload, Mapping) and shell_payload.get(area):
                return True
    return False


def _validate_patch_identity_payload(patch_id: object) -> str:
    if not isinstance(patch_id, str) or not patch_id:
        raise ImpressFormatError("`.impress` patch IDs must be non-empty strings.")
    return patch_id


def _body_shell_payloads(body_payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    shell_payloads = body_payload.get("shells")
    if not isinstance(shell_payloads, Sequence) or isinstance(shell_payloads, (str, bytes)):
        raise ImpressFormatError("SurfaceBody shells must be an array.")
    return tuple(_required_mapping(shell_payload, "SurfaceBody shell payload") for shell_payload in shell_payloads)


def _decode_surface_shell_payload_from_document(
    shell_payload: Mapping[str, object],
    patches: Mapping[str, SurfacePatch],
    referenced_patch_ids: set[str],
) -> SurfaceShell:
    patch_ids = shell_payload.get("patches")
    if not isinstance(patch_ids, Sequence) or isinstance(patch_ids, (str, bytes)):
        raise ImpressFormatError("SurfaceShell patches must be an array of patch identities.")
    decoded_patches: list[SurfacePatch] = []
    for patch_id in patch_ids:
        patch_key = _validate_patch_identity_payload(patch_id)
        try:
            decoded_patches.append(patches[patch_key])
        except KeyError as exc:
            raise ImpressFormatError(f"SurfaceShell references missing patch payload {patch_key!r}.") from exc
        referenced_patch_ids.add(patch_key)
    return decode_surface_shell_payload(shell_payload, patches=decoded_patches)


def _encode_parameter_domain_payload(domain: ParameterDomain) -> dict[str, object]:
    return {
        "u_range": list(domain.u_range),
        "v_range": list(domain.v_range),
        "normalized": domain.normalized,
    }


def _decode_parameter_domain_payload(payload: object) -> ParameterDomain:
    if payload is None:
        return ParameterDomain()
    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfacePatch domain must be an object.")
    unknown_keys = set(payload) - {"u_range", "v_range", "normalized"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported SurfacePatch domain fields: {keys}.")
    normalized = payload.get("normalized", True)
    if not isinstance(normalized, bool):
        raise ImpressFormatError("SurfacePatch domain normalized must be a boolean.")
    try:
        return ParameterDomain(
            u_range=_validate_range_payload(payload.get("u_range"), "u_range"),
            v_range=_validate_range_payload(payload.get("v_range"), "v_range"),
            normalized=normalized,
        )
    except ValueError as exc:
        raise ImpressFormatError(str(exc)) from exc


def _encode_patch_geometry_payload(patch: SurfacePatch) -> dict[str, object]:
    geometry = patch.geometry_payload()
    if isinstance(patch, PlanarSurfacePatch):
        return {
            "payload_version": _ANALYTIC_PATCH_PAYLOAD_VERSION,
            "origin": _array_payload(geometry["origin"]),
            "u_axis": _array_payload(geometry["u_axis"]),
            "v_axis": _array_payload(geometry["v_axis"]),
        }
    if isinstance(patch, RuledSurfacePatch):
        return {
            "payload_version": _ANALYTIC_PATCH_PAYLOAD_VERSION,
            "start_curve": _array_payload(geometry["start_curve"]),
            "end_curve": _array_payload(geometry["end_curve"]),
        }
    if isinstance(patch, RevolutionSurfacePatch):
        return {
            "payload_version": _ANALYTIC_PATCH_PAYLOAD_VERSION,
            "profile_curve": _array_payload(geometry["profile_curve"]),
            "axis_origin": _array_payload(geometry["axis_origin"]),
            "axis_direction": _array_payload(geometry["axis_direction"]),
            "start_angle_deg": float(geometry["start_angle_deg"]),
            "sweep_angle_deg": float(geometry["sweep_angle_deg"]),
        }
    if isinstance(patch, BSplineSurfacePatch):
        return {
            "payload_version": _SPLINE_PATCH_PAYLOAD_VERSION,
            "degree_u": int(geometry["degree_u"]),
            "degree_v": int(geometry["degree_v"]),
            "knots_u": _array_payload(geometry["knots_u"]),
            "knots_v": _array_payload(geometry["knots_v"]),
            "control_net": _array_payload(geometry["control_net"]),
        }
    if isinstance(patch, NURBSSurfacePatch):
        return {
            "payload_version": _SPLINE_PATCH_PAYLOAD_VERSION,
            "degree_u": int(geometry["degree_u"]),
            "degree_v": int(geometry["degree_v"]),
            "knots_u": _array_payload(geometry["knots_u"]),
            "knots_v": _array_payload(geometry["knots_v"]),
            "control_net": _array_payload(geometry["control_net"]),
            "weights": _array_payload(geometry["weights"]),
        }
    if isinstance(patch, SweepSurfacePatch):
        return {
            "payload_version": _SWEEP_PATCH_PAYLOAD_VERSION,
            "profile_points_uv": _array_payload(geometry["profile_points_uv"]),
            "path_points": _array_payload(geometry["path_points"]),
            "frame_policy": str(geometry["frame_policy"]),
            "profile_reference": geometry["profile_reference"],
            "path_reference": geometry["path_reference"],
        }
    if isinstance(patch, SubdivisionSurfacePatch):
        return {
            "payload_version": _SUBDIVISION_PATCH_PAYLOAD_VERSION,
            "scheme": str(geometry["scheme"]),
            "subdivision_level": int(geometry["subdivision_level"]),
            "control_points": _array_payload(geometry["control_points"]),
            "faces": [list(face) for face in geometry["faces"]],
            "creases": [
                {
                    "edge": list(crease["edge"]),
                    "sharpness": float(crease["sharpness"]),
                }
                for crease in geometry["creases"]
            ],
        }
    if isinstance(patch, ImplicitSurfacePatch):
        return {
            "payload_version": _IMPLICIT_PATCH_PAYLOAD_VERSION,
            "field": geometry["field"],
            "bounds": _array_payload(geometry["bounds"]),
        }
    if isinstance(patch, HeightmapSurfacePatch):
        return {
            "payload_version": _SAMPLED_PATCH_PAYLOAD_VERSION,
            "height_samples": _array_payload(geometry["height_samples"]),
            "alpha_mask": np.asarray(geometry["alpha_mask"], dtype=bool).tolist(),
            "alpha_mode": str(geometry["alpha_mode"]),
            "xy_scale": _array_payload(geometry["xy_scale"]),
            "center": _array_payload(geometry["center"]),
            "height_scale": float(geometry["height_scale"]),
        }
    if isinstance(patch, DisplacementSurfacePatch):
        direction = geometry["direction"]
        return {
            "payload_version": _SAMPLED_PATCH_PAYLOAD_VERSION,
            "source_patch": encode_surface_patch_payload(patch.source_patch),
            "displacement_samples": _array_payload(geometry["displacement_samples"]),
            "alpha_mask": np.asarray(geometry["alpha_mask"], dtype=bool).tolist(),
            "alpha_mode": str(geometry["alpha_mode"]),
            "height_scale": float(geometry["height_scale"]),
            "direction": _array_payload(direction) if not isinstance(direction, str) else direction,
            "projection": str(geometry["projection"]),
            "plane": str(geometry["plane"]),
            "projection_bounds": _array_payload(geometry["projection_bounds"]),
        }
    raise ImpressFormatError(f"Unsupported SurfacePatch kind {type(patch).__name__!r}.")


def _validate_patch_geometry_fields(
    geometry: Mapping[str, object],
    *,
    kind: str,
    allowed_fields: set[str],
    expected_payload_version: int = _ANALYTIC_PATCH_PAYLOAD_VERSION,
) -> None:
    unknown_keys = set(geometry) - allowed_fields
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported {kind} geometry fields: {keys}.")
    payload_version = geometry.get("payload_version")
    if payload_version != expected_payload_version:
        raise ImpressFormatError(
            f"{kind} geometry payload_version must be {expected_payload_version}."
        )


def _validate_patch_kind_family(kind: str, family: str) -> None:
    dispatch = _PATCH_FAMILY_DISPATCH.get(kind)
    if dispatch is None:
        raise ImpressFormatError(f"Unsupported SurfacePatch kind {kind!r}.")
    if family != dispatch.family:
        raise ImpressFormatError(
            f"SurfacePatch kind {kind!r} requires family {dispatch.family!r}; got {family!r}."
        )


def _array_payload(value: object) -> list[object]:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ImpressFormatError("SurfacePatch geometry arrays must contain only finite values.")
    return array.tolist()


def _validate_range_payload(payload: object, name: str) -> tuple[float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 2:
        raise ImpressFormatError(f"SurfacePatch domain {name} must be a two-value numeric array.")
    return (_validate_float_payload(payload[0], f"{name}[0]"), _validate_float_payload(payload[1], f"{name}[1]"))


def _validate_bounds3_payload(payload: object, name: str) -> tuple[float, float, float, float, float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 6:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a six-value numeric bounds array.")
    bounds = tuple(_validate_float_payload(value, f"{name}[]") for value in payload)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must have positive span on every axis.")
    return bounds


def _validate_bounds2_payload(payload: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 4:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a four-value numeric bounds array.")
    bounds = tuple(_validate_float_payload(value, f"{name}[]") for value in payload)
    umin, umax, vmin, vmax = bounds
    if np.isclose(umax, umin) or np.isclose(vmax, vmin):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must have non-degenerate spans.")
    return bounds


def _validate_nonnegative_int_payload(payload: object, name: str) -> int:
    if not isinstance(payload, int) or isinstance(payload, bool) or payload < 0:
        raise ImpressFormatError(f"{name} must be a non-negative integer.")
    return payload


def _validate_positive_int_payload(payload: object, name: str) -> int:
    if not isinstance(payload, int) or isinstance(payload, bool) or payload < 1:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a positive integer.")
    return payload


def _validate_array_payload(payload: object, name: str) -> Sequence[object]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be an array.")
    return payload


def _validate_vec3_payload(payload: object, name: str) -> np.ndarray:
    try:
        vector = np.asarray(payload, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a 3D numeric vector.") from exc
    if not np.all(np.isfinite(vector)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return vector


def _validate_points3_payload(payload: object, name: str) -> np.ndarray:
    try:
        points = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric point array.") from exc
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain at least two 3D points.")
    if not np.all(np.isfinite(points)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return points


def _validate_control_net3_payload(payload: object, name: str) -> np.ndarray:
    try:
        control_net = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric 3D control net.") from exc
    if control_net.ndim != 3 or control_net.shape[2] != 3 or control_net.shape[0] < 2 or control_net.shape[1] < 2:
        raise ImpressFormatError(
            f"SurfacePatch geometry {name} must be a 3D control net with shape (u_count, v_count, 3)."
        )
    if not np.all(np.isfinite(control_net)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return control_net


def _validate_control_points3_payload(payload: object, name: str) -> np.ndarray:
    try:
        points = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric control-point array.") from exc
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain at least three 3D control points.")
    if not np.all(np.isfinite(points)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return points


def _validate_weight_net_payload(payload: object, name: str) -> np.ndarray:
    try:
        weights = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric 2D weight net.") from exc
    if weights.ndim != 2 or weights.shape[0] < 2 or weights.shape[1] < 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a 2D weight net.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite positive values.")
    return weights


def _validate_numeric_grid_payload(payload: object, name: str) -> np.ndarray:
    try:
        grid = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric 2D grid.") from exc
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a 2D grid with at least 2x2 samples.")
    if not np.all(np.isfinite(grid)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return grid


def _validate_bool_grid_payload(payload: object, name: str) -> np.ndarray:
    values = np.asarray(payload, dtype=object)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a 2D boolean grid with at least 2x2 samples.")
    for value in values.flat:
        if not isinstance(value, (bool, np.bool_)):
            raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only booleans.")
    return values.astype(bool)


def _validate_positive_pair_payload(payload: object, name: str) -> tuple[float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a two-value numeric array.")
    pair = (_validate_float_payload(payload[0], f"{name}[0]"), _validate_float_payload(payload[1], f"{name}[1]"))
    if pair[0] <= 0.0 or pair[1] <= 0.0:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain two positive values.")
    return pair


def _validate_direction_payload(payload: object, name: str) -> str | tuple[float, float, float]:
    if isinstance(payload, str):
        return _validate_string_payload(payload, name)
    vector = _validate_vec3_payload(payload, name)
    return tuple(float(value) for value in vector)


def _validate_points2_payload(payload: object, name: str) -> np.ndarray:
    try:
        points = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"TrimLoop {name} must be a numeric point array.") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ImpressFormatError(f"TrimLoop {name} must contain at least three 2D points.")
    if not np.all(np.isfinite(points)):
        raise ImpressFormatError(f"TrimLoop {name} must contain only finite values.")
    return points


def _validate_profile_points2_payload(payload: object, name: str) -> np.ndarray:
    try:
        points = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric 2D point array.") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain at least two 2D points.")
    if not np.all(np.isfinite(points)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must contain only finite values.")
    return points


def _validate_float_payload(payload: object, name: str) -> float:
    try:
        value = float(payload)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be finite.")
    return value


def _validate_float_tuple_payload(payload: object, name: str) -> tuple[float, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a numeric array.")
    values = tuple(_validate_float_payload(value, f"{name}[]") for value in payload)
    if not values:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must not be empty.")
    return values


def _validate_string_payload(payload: object, name: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be a non-empty string.")
    return payload


def _validate_optional_string_payload(payload: object, name: str) -> str | None:
    if payload is None:
        return None
    return _validate_string_payload(payload, name)


def _validate_subdivision_faces_payload(payload: object, name: str) -> tuple[tuple[int, ...], ...]:
    faces_payload = _validate_array_payload(payload, name)
    faces: list[tuple[int, ...]] = []
    for face_index, face_payload in enumerate(faces_payload):
        face_values = _validate_array_payload(face_payload, f"{name}[{face_index}]")
        face: list[int] = []
        for value in face_values:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ImpressFormatError(f"SurfacePatch geometry {name} indices must be integers.")
            face.append(value)
        faces.append(tuple(face))
    if not faces:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must not be empty.")
    return tuple(faces)


def _decode_subdivision_crease_payload(payload: object) -> SubdivisionCrease:
    if not isinstance(payload, Mapping):
        raise ImpressFormatError("Subdivision crease payload must be an object.")
    unknown_keys = set(payload) - {"edge", "sharpness"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported Subdivision crease fields: {keys}.")
    edge_payload = _validate_array_payload(payload.get("edge"), "crease.edge")
    if len(edge_payload) != 2:
        raise ImpressFormatError("Subdivision crease edge must contain exactly two indices.")
    edge: list[int] = []
    for value in edge_payload:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ImpressFormatError("Subdivision crease edge indices must be integers.")
        edge.append(value)
    try:
        return SubdivisionCrease(
            edge=(edge[0], edge[1]),
            sharpness=_validate_float_payload(payload.get("sharpness", 1.0), "crease.sharpness"),
        )
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc


def _decode_implicit_field_node_payload(
    payload: object,
    *,
    path: str = "field",
    policy: ImplicitFieldSafetyPolicy | None = None,
    state: dict[str, int] | None = None,
    depth: int = 1,
) -> ImplicitFieldNode:
    policy = ImplicitFieldSafetyPolicy() if policy is None else policy
    state = {"node_count": 0} if state is None else state
    state["node_count"] += 1
    if state["node_count"] > policy.max_nodes:
        raise ImpressFormatError(
            f"Unsafe implicit field payload at {path}: field tree exceeds max_nodes "
            f"{policy.max_nodes}."
        )
    if depth > policy.max_depth:
        raise ImpressFormatError(
            f"Unsafe implicit field payload at {path}: field tree exceeds max_depth "
            f"{policy.max_depth}."
        )
    if not isinstance(payload, Mapping):
        raise ImpressFormatError(f"Implicit field payload at {path} must be an object.")
    unknown_keys = set(payload) - {"kind", "parameters", "children"}
    if unknown_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ImpressFormatError(f"Unsupported implicit field node fields at {path}: {keys}.")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ImpressFormatError(f"Implicit field node kind at {path} must be a non-empty string.")
    if kind not in IMPLICIT_FIELD_NODE_KINDS:
        allowed = ", ".join(sorted(IMPLICIT_FIELD_NODE_KINDS))
        raise ImpressFormatError(
            f"Unsupported implicit field node kind {kind!r} at {path}; allowed nodes: {allowed}."
        )
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ImpressFormatError(f"Implicit field node parameters at {path} must be an object.")
    if len(parameters) > policy.max_parameters_per_node:
        raise ImpressFormatError(
            f"Unsafe implicit field payload at {path}.parameters: field node exceeds "
            f"max_parameters_per_node {policy.max_parameters_per_node}."
        )
    for key, value in parameters.items():
        parameter_path = f"{path}.parameters.{key}"
        if _implicit_parameter_has_unsafe_payload(str(key), value, policy):
            raise ImpressFormatError(
                f"Unsafe implicit field payload at {parameter_path}: executable payloads are not allowed; "
                f"allowed nodes: {', '.join(sorted(IMPLICIT_FIELD_NODE_KINDS))}."
            )
    children_payload = payload.get("children", [])
    children = _validate_array_payload(children_payload, "field.children")
    if len(children) > policy.max_children_per_node:
        raise ImpressFormatError(
            f"Unsafe implicit field payload at {path}.children: field node exceeds "
            f"max_children_per_node {policy.max_children_per_node}."
        )
    try:
        return ImplicitFieldNode(
            kind=kind,
            parameters=dict(parameters),
            children=tuple(
                _decode_implicit_field_node_payload(
                    child,
                    path=f"{path}.children[{index}]",
                    policy=policy,
                    state=state,
                    depth=depth + 1,
                )
                for index, child in enumerate(children)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"Invalid implicit field payload at {path}: {exc}") from exc


def _implicit_parameter_has_unsafe_payload(
    name: str,
    value: object,
    policy: ImplicitFieldSafetyPolicy,
) -> bool:
    normalized_name = name.strip().lower()
    if normalized_name.startswith("__") or normalized_name in {
        "__builtins__",
        "callable",
        "code",
        "eval",
        "exec",
        "function",
        "globals",
        "import",
        "lambda",
        "locals",
        "module",
    }:
        return True
    if isinstance(value, str):
        text = value.strip().lower()
        if len(value) > policy.max_string_length:
            return True
        return any(token in text for token in ("__", "import ", "eval(", "exec(", "lambda "))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_implicit_parameter_has_unsafe_payload(name, item, policy) for item in value)
    if isinstance(value, Mapping):
        return any(_implicit_parameter_has_unsafe_payload(str(key), nested_value, policy) for key, nested_value in value.items())
    return False


def validate_impress_document_root(root: Mapping[str, object]) -> ImpressDocumentRoot:
    """Validate a minimal `.impress` root envelope and return its typed form."""

    if not isinstance(root, Mapping):
        raise ImpressFormatError("`.impress` root must be an object.")
    non_string_keys = [key for key in root if not isinstance(key, str)]
    if non_string_keys:
        raise ImpressFormatError("`.impress` root keys must be strings.")

    format_value = root.get("format")
    if format_value != IMPRESS_FORMAT:
        raise ImpressFormatError("`.impress` root must declare format='impress'.")

    schema_version = root.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ImpressFormatError("`.impress` root must declare a non-empty schema_version.")
    if schema_version != CURRENT_IMPRESS_SCHEMA_VERSION:
        raise UnsupportedImpressSchemaVersion(
            f"Unsupported `.impress` schema_version {schema_version!r}; "
            f"expected {CURRENT_IMPRESS_SCHEMA_VERSION!r}."
        )

    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ImpressFormatError("`.impress` root metadata must be an object when present.")
    _validate_impress_diagnostic_metadata(metadata)

    units = validate_impress_units(root.get("units"))
    return ImpressDocumentRoot(schema_version=schema_version, units=units, metadata=dict(metadata))


def _validate_impress_diagnostic_metadata(metadata: Mapping[str, object]) -> None:
    diagnostics = metadata.get("diagnostics")
    if diagnostics is None:
        return
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        raise ImpressFormatError("`.impress` diagnostic metadata must be an array.")
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, Mapping):
            raise ImpressFormatError(f"`.impress` diagnostic metadata at diagnostics[{index}] must be an object.")
        unknown_keys = set(diagnostic) - IMPRESS_DIAGNOSTIC_METADATA_FIELDS
        if unknown_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ImpressFormatError(f"Unsupported `.impress` diagnostic metadata fields at diagnostics[{index}]: {keys}.")
        code = diagnostic.get("code")
        message = diagnostic.get("message")
        severity = diagnostic.get("severity", "error")
        if not isinstance(code, str) or not code.strip():
            raise ImpressFormatError(f"`.impress` diagnostic metadata at diagnostics[{index}].code must be a non-empty string.")
        if not isinstance(message, str) or not message.strip():
            raise ImpressFormatError(f"`.impress` diagnostic metadata at diagnostics[{index}].message must be a non-empty string.")
        if not isinstance(severity, str) or not severity.strip():
            raise ImpressFormatError(f"`.impress` diagnostic metadata at diagnostics[{index}].severity must be a non-empty string.")
        path = diagnostic.get("path")
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ImpressFormatError(f"`.impress` diagnostic metadata at diagnostics[{index}].path must be a non-empty string.")
