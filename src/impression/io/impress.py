from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
