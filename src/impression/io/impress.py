from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from impression.modeling.surface import SurfaceBody, SurfacePatch, SurfaceShell

IMPRESS_FORMAT = "impress"
CURRENT_IMPRESS_SCHEMA_VERSION = "1.0"
DEFAULT_IMPRESS_LENGTH_UNIT = "unitless"
SUPPORTED_IMPRESS_LENGTH_UNITS = frozenset({"unitless", "mm", "cm", "m", "in", "ft"})


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


def encode_surface_shell_payload(shell: SurfaceShell) -> dict[str, object]:
    """Encode container-level shell fields for `.impress` persistence."""

    if not isinstance(shell, SurfaceShell):
        raise ImpressFormatError("encode_surface_shell_payload requires a SurfaceShell.")
    if shell.seams or shell.adjacency:
        raise ImpressFormatError("SurfaceShell seam and adjacency payloads are encoded by later `.impress` codecs.")
    return {
        "connected": shell.connected,
        "patch_count": shell.patch_count,
        "patches": [patch.stable_identity for patch in shell.patches],
        "transform_matrix": shell.transform_matrix.tolist(),
        "metadata": dict(shell.metadata),
        "seams": [],
        "adjacency": [],
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
    if seams != [] or adjacency != []:
        raise ImpressFormatError("SurfaceShell seam and adjacency payloads require their dedicated codecs.")

    transform_matrix = _validate_matrix4_payload(payload.get("transform_matrix"))
    try:
        return SurfaceShell(
            tuple(patches),
            connected=connected,
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
