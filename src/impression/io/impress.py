from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from impression.modeling.surface import SurfaceBody

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
