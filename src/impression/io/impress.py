from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from impression.modeling.surface import (
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
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
_PATCH_KIND_FAMILIES = {
    "PlanarSurfacePatch": "planar",
    "RuledSurfacePatch": "ruled",
    "RevolutionSurfacePatch": "revolution",
}


class ImpressFormatError(ValueError):
    """Base error for invalid `.impress` document roots."""


class UnsupportedImpressSchemaVersion(ImpressFormatError):
    """Raised when an `.impress` document uses an unsupported schema."""


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
    try:
        output_path.write_text(dumps_impress_json(payload, options=options), encoding="utf-8")
    except OSError as exc:
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

    if not isinstance(patch, SurfacePatch):
        raise ImpressFormatError("encode_surface_patch_payload requires a SurfacePatch.")
    _validate_patch_kind_family(type(patch).__name__, patch.family)
    return {
        "kind": type(patch).__name__,
        "family": patch.family,
        "domain": _encode_parameter_domain_payload(patch.domain),
        "capability_flags": sorted(patch.capability_flags),
        "transform_matrix": patch.transform_matrix.tolist(),
        "metadata": dict(patch.metadata),
        "trim_loops": [encode_trim_loop_payload(trim_loop) for trim_loop in patch.trim_loops],
        "geometry": _encode_patch_geometry_payload(patch),
    }


def decode_surface_patch_payload(payload: Mapping[str, object]) -> SurfacePatch:
    """Decode a `.impress` patch payload through the public patch constructors."""

    if not isinstance(payload, Mapping):
        raise ImpressFormatError("SurfacePatch payload must be an object.")
    unknown_keys = set(payload) - {
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
    common = {
        "family": family,
        "domain": domain,
        "capability_flags": frozenset(flags),
        "trim_loops": trim_loops,
        "transform_matrix": transform_matrix,
        "metadata": dict(metadata),
    }
    try:
        if kind == "PlanarSurfacePatch":
            return PlanarSurfacePatch(
                **common,
                origin=_validate_vec3_payload(geometry.get("origin"), "origin"),
                u_axis=_validate_vec3_payload(geometry.get("u_axis"), "u_axis"),
                v_axis=_validate_vec3_payload(geometry.get("v_axis"), "v_axis"),
            )
        if kind == "RuledSurfacePatch":
            return RuledSurfacePatch(
                **common,
                start_curve=_validate_points3_payload(geometry.get("start_curve"), "start_curve"),
                end_curve=_validate_points3_payload(geometry.get("end_curve"), "end_curve"),
            )
        if kind == "RevolutionSurfacePatch":
            return RevolutionSurfacePatch(
                **common,
                profile_curve=_validate_points3_payload(geometry.get("profile_curve"), "profile_curve"),
                axis_origin=_validate_vec3_payload(geometry.get("axis_origin"), "axis_origin"),
                axis_direction=_validate_vec3_payload(geometry.get("axis_direction"), "axis_direction"),
                start_angle_deg=_validate_float_payload(geometry.get("start_angle_deg"), "start_angle_deg"),
                sweep_angle_deg=_validate_float_payload(geometry.get("sweep_angle_deg"), "sweep_angle_deg"),
            )
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(str(exc)) from exc
    raise ImpressFormatError(f"Unsupported SurfacePatch kind {kind!r}.")


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
            "origin": _array_payload(geometry["origin"]),
            "u_axis": _array_payload(geometry["u_axis"]),
            "v_axis": _array_payload(geometry["v_axis"]),
        }
    if isinstance(patch, RuledSurfacePatch):
        return {
            "start_curve": _array_payload(geometry["start_curve"]),
            "end_curve": _array_payload(geometry["end_curve"]),
        }
    if isinstance(patch, RevolutionSurfacePatch):
        return {
            "profile_curve": _array_payload(geometry["profile_curve"]),
            "axis_origin": _array_payload(geometry["axis_origin"]),
            "axis_direction": _array_payload(geometry["axis_direction"]),
            "start_angle_deg": float(geometry["start_angle_deg"]),
            "sweep_angle_deg": float(geometry["sweep_angle_deg"]),
        }
    raise ImpressFormatError(f"Unsupported SurfacePatch kind {type(patch).__name__!r}.")


def _validate_patch_kind_family(kind: str, family: str) -> None:
    expected_family = _PATCH_KIND_FAMILIES.get(kind)
    if expected_family is None:
        raise ImpressFormatError(f"Unsupported SurfacePatch kind {kind!r}.")
    if family != expected_family:
        raise ImpressFormatError(
            f"SurfacePatch kind {kind!r} requires family {expected_family!r}; got {family!r}."
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


def _validate_nonnegative_int_payload(payload: object, name: str) -> int:
    if not isinstance(payload, int) or isinstance(payload, bool) or payload < 0:
        raise ImpressFormatError(f"{name} must be a non-negative integer.")
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


def _validate_float_payload(payload: object, name: str) -> float:
    try:
        value = float(payload)
    except (TypeError, ValueError) as exc:
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ImpressFormatError(f"SurfacePatch geometry {name} must be finite.")
    return value


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

    units = validate_impress_units(root.get("units"))
    return ImpressDocumentRoot(schema_version=schema_version, units=units, metadata=dict(metadata))
