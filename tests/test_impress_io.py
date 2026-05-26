from __future__ import annotations

import pytest

from impression.io import (
    CURRENT_IMPRESS_SCHEMA_VERSION,
    DEFAULT_IMPRESS_LENGTH_UNIT,
    IMPRESS_FORMAT,
    ImpressFormatError,
    ImpressUnits,
    UnsupportedImpressSchemaVersion,
    make_impress_document_root,
    validate_impress_units,
    validate_impress_document_root,
)


def test_make_impress_document_root_declares_format_and_schema() -> None:
    root = make_impress_document_root(metadata={"author": "test"})

    assert root.to_json_object() == {
        "format": IMPRESS_FORMAT,
        "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION,
        "units": {"length": DEFAULT_IMPRESS_LENGTH_UNIT},
        "metadata": {"author": "test"},
    }


def test_validate_impress_document_root_returns_typed_root() -> None:
    root = validate_impress_document_root(
        {
            "format": "impress",
            "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION,
            "units": {"length": "mm"},
            "metadata": {"purpose": "roundtrip"},
        }
    )

    assert root.schema_version == CURRENT_IMPRESS_SCHEMA_VERSION
    assert root.units == ImpressUnits(length="mm")
    assert root.metadata == {"purpose": "roundtrip"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "mesh", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION},
        {"format": "impress"},
        {"format": "impress", "schema_version": ""},
        {"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION, "metadata": []},
        {"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION, "units": []},
        {"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION, "units": {"angle": "degree"}},
        {"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION, "units": {"length": "parsec"}},
    ],
)
def test_validate_impress_document_root_rejects_invalid_roots(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        validate_impress_document_root(payload)


def test_validate_impress_document_root_rejects_unsupported_schema() -> None:
    with pytest.raises(UnsupportedImpressSchemaVersion, match="Unsupported `.impress` schema_version"):
        validate_impress_document_root({"format": "impress", "schema_version": "999.0"})


def test_validate_impress_document_root_defaults_missing_units() -> None:
    root = validate_impress_document_root(
        {
            "format": "impress",
            "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION,
            "metadata": {},
        }
    )

    assert root.units == ImpressUnits()


def test_make_impress_document_root_accepts_symbolic_units() -> None:
    root = make_impress_document_root(units={"length": "in"})

    assert root.units == ImpressUnits(length="in")
    assert root.to_json_object()["units"] == {"length": "in"}


def test_validate_impress_units_rejects_invalid_unit() -> None:
    with pytest.raises(ImpressFormatError, match="Unsupported `.impress` length unit"):
        validate_impress_units({"length": "px"})


def test_validate_impress_document_root_rejects_unsafe_root_shape() -> None:
    with pytest.raises(ImpressFormatError, match="root must be an object"):
        validate_impress_document_root([])  # type: ignore[arg-type]

    with pytest.raises(ImpressFormatError, match="root keys must be strings"):
        validate_impress_document_root({1: "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION})  # type: ignore[dict-item]
