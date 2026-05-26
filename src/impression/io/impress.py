from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

IMPRESS_FORMAT = "impress"
CURRENT_IMPRESS_SCHEMA_VERSION = "1.0"


class ImpressFormatError(ValueError):
    """Base error for invalid `.impress` document roots."""


class UnsupportedImpressSchemaVersion(ImpressFormatError):
    """Raised when an `.impress` document uses an unsupported schema."""


@dataclass(frozen=True)
class ImpressDocumentRoot:
    """Minimal `.impress` document envelope."""

    schema_version: str = CURRENT_IMPRESS_SCHEMA_VERSION
    format: str = IMPRESS_FORMAT
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }


def make_impress_document_root(
    *,
    schema_version: str = CURRENT_IMPRESS_SCHEMA_VERSION,
    metadata: Mapping[str, object] | None = None,
) -> ImpressDocumentRoot:
    """Create a validated minimal `.impress` document root."""

    root = ImpressDocumentRoot(
        schema_version=schema_version,
        metadata={} if metadata is None else dict(metadata),
    )
    validate_impress_document_root(root.to_json_object())
    return root


def validate_impress_document_root(root: Mapping[str, object]) -> ImpressDocumentRoot:
    """Validate a minimal `.impress` root envelope and return its typed form."""

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

    return ImpressDocumentRoot(schema_version=schema_version, metadata=dict(metadata))
