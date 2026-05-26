from __future__ import annotations

import pytest

from impression.io import (
    CURRENT_IMPRESS_SCHEMA_VERSION,
    DEFAULT_IMPRESS_LENGTH_UNIT,
    IMPRESS_FORMAT,
    ImpressBodyEntry,
    ImpressFormatError,
    ImpressUnits,
    SurfaceBodyStore,
    UnsupportedImpressSchemaVersion,
    decode_surface_body_payload,
    decode_surface_patch_payload,
    decode_surface_shell_payload,
    encode_surface_body_payload,
    encode_surface_patch_payload,
    encode_surface_shell_payload,
    make_impress_document_root,
    make_surface_body_store,
    validate_impress_units,
    validate_impress_document_root,
    validate_surface_body_store,
)
import numpy as np
from impression.modeling.surface import (
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
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


def _single_patch_body(offset: float = 0.0):
    shell = make_surface_shell([PlanarSurfacePatch(family="planar", origin=(offset, 0.0, 0.0))])
    return make_surface_body([shell])


def test_make_surface_body_store_creates_ordered_identity_entries() -> None:
    body_a = _single_patch_body(0.0)
    body_b = _single_patch_body(2.0)

    store = make_surface_body_store([body_a, body_b])

    assert [entry.body_id for entry in store.bodies] == ["body-0001", "body-0002"]
    assert [entry.stable_identity for entry in store.bodies] == [body_a.stable_identity, body_b.stable_identity]
    assert store.to_json_object() == {
        "bodies": [
            {"body_id": "body-0001", "stable_identity": body_a.stable_identity, "body_ref": "body-0001"},
            {"body_id": "body-0002", "stable_identity": body_b.stable_identity, "body_ref": "body-0002"},
        ]
    }


def test_validate_surface_body_store_accepts_payload_store() -> None:
    store = validate_surface_body_store(
        {
            "bodies": [
                {"body_id": "body-a", "stable_identity": "identity-a"},
                {"body_id": "body-b", "stable_identity": "identity-b", "body_ref": "body-b"},
            ]
        }
    )

    assert isinstance(store, SurfaceBodyStore)
    assert store.bodies == (
        ImpressBodyEntry(body_id="body-a", stable_identity="identity-a"),
        ImpressBodyEntry(body_id="body-b", stable_identity="identity-b"),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"bodies": []},
        {"bodies": "body-a"},
        {"bodies": [{"stable_identity": "identity-a"}]},
        {"bodies": [{"body_id": "body-a"}]},
        {"bodies": [{"body_id": "body-a", "stable_identity": "identity-a", "body_ref": "other"}]},
        {
            "bodies": [
                {"body_id": "body-a", "stable_identity": "identity-a"},
                {"body_id": "body-a", "stable_identity": "identity-b"},
            ]
        },
        {
            "bodies": [
                {"body_id": "body-a", "stable_identity": "identity-a"},
                {"body_id": "body-b", "stable_identity": "identity-a"},
            ]
        },
    ],
)
def test_validate_surface_body_store_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        validate_surface_body_store(payload)


def test_surface_body_store_rejects_entry_identity_mismatch() -> None:
    body = _single_patch_body()

    with pytest.raises(ImpressFormatError, match="stable_identity does not match"):
        SurfaceBodyStore((ImpressBodyEntry(body_id="body-a", stable_identity="not-the-body", body=body),))


def test_encode_decode_surface_shell_payload_round_trips_container_fields() -> None:
    patch = PlanarSurfacePatch(family="planar", metadata={"kernel": {"name": "face"}})
    shell = make_surface_shell([patch], connected=False, metadata={"consumer": {"label": "shell"}})

    payload = encode_surface_shell_payload(shell)
    decoded = decode_surface_shell_payload(payload, patches=[patch])

    assert payload["patches"] == [patch.stable_identity]
    assert decoded.connected is False
    assert decoded.metadata == {"consumer": {"label": "shell"}}
    assert decoded.patches == (patch,)


def test_encode_decode_surface_body_payload_round_trips_container_fields() -> None:
    body = _single_patch_body().with_transform(
        [
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    payload = encode_surface_body_payload(body)
    decoded = decode_surface_body_payload(payload, shells=body.shells)

    assert payload["shell_count"] == 1
    assert len(payload["shells"]) == 1
    assert decoded.metadata == body.metadata
    assert np.allclose(decoded.transform_matrix, body.transform_matrix)
    assert decoded.shells == body.shells


def test_encode_decode_planar_surface_patch_payload_round_trips_base_fields() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        domain=ParameterDomain(u_range=(-1.0, 1.0), v_range=(2.0, 4.0), normalized=False),
        capability_flags=frozenset({"evaluatable", "tessellatable"}),
        transform_matrix=[
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        metadata={"kernel": {"name": "face-1"}},
        origin=(1.0, 2.0, 3.0),
        u_axis=(2.0, 0.0, 0.0),
        v_axis=(0.0, 3.0, 0.0),
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["kind"] == "PlanarSurfacePatch"
    assert payload["family"] == "planar"
    assert payload["domain"] == {"u_range": [-1.0, 1.0], "v_range": [2.0, 4.0], "normalized": False}
    assert payload["capability_flags"] == ["evaluatable", "tessellatable"]
    assert decoded.stable_identity == patch.stable_identity
    assert decoded.metadata == patch.metadata


def test_encode_decode_ruled_and_revolution_surface_patch_payloads_round_trip() -> None:
    ruled = RuledSurfacePatch(
        family="ruled",
        start_curve=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)],
        end_curve=[(2.0, 0.0, 0.5), (2.0, 1.0, 0.5), (2.0, 2.0, 0.5)],
    )
    revolution = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=[(1.0, 0.0, 0.0), (1.5, 0.0, 2.0)],
        axis_origin=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        start_angle_deg=15.0,
        sweep_angle_deg=180.0,
    )

    assert decode_surface_patch_payload(encode_surface_patch_payload(ruled)).stable_identity == ruled.stable_identity
    assert decode_surface_patch_payload(encode_surface_patch_payload(revolution)).stable_identity == revolution.stable_identity


def test_decode_surface_patch_payload_rejects_invalid_family_dispatch() -> None:
    patch = PlanarSurfacePatch(family="planar")
    payload = encode_surface_patch_payload(patch)
    payload["family"] = "ruled"

    with pytest.raises(ImpressFormatError, match="requires family"):
        decode_surface_patch_payload(payload)


def test_decode_surface_patch_payload_uses_constructor_validation() -> None:
    patch = PlanarSurfacePatch(family="planar")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    geometry["v_axis"] = [2.0, 0.0, 0.0]
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match="linearly independent"):
        decode_surface_patch_payload(payload)


def test_surface_patch_payload_codec_refuses_trim_payload_until_trim_codec() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        trim_loops=(TrimLoop(points_uv=[(0.1, 0.1), (0.8, 0.1), (0.1, 0.8)], category="outer"),),
    )

    with pytest.raises(ImpressFormatError, match="trim payloads require"):
        encode_surface_patch_payload(patch)

    payload = encode_surface_patch_payload(PlanarSurfacePatch(family="planar"))
    payload["trim_loops"] = [{"category": "outer", "points_uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}]
    with pytest.raises(ImpressFormatError, match="trim payloads require"):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"kind": "UnknownSurfacePatch", "family": "unknown", "geometry": {}},
        {"kind": "PlanarSurfacePatch", "family": "planar", "geometry": []},
        {"kind": "PlanarSurfacePatch", "family": "planar", "geometry": {}, "capability_flags": "flag"},
        {"kind": "PlanarSurfacePatch", "family": "planar", "geometry": {}, "metadata": []},
        {"kind": "PlanarSurfacePatch", "family": "planar", "geometry": {}, "domain": {"u_range": [0.0, 0.0]}},
        {
            "kind": "PlanarSurfacePatch",
            "family": "planar",
            "geometry": {"origin": [float("nan"), 0.0, 0.0]},
        },
    ],
)
def test_decode_surface_patch_payload_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"patch_count": 1, "patches": "patch-a"},
        {"patch_count": 0, "patches": []},
        {"patch_count": 2, "patches": ["patch-a"]},
        {"patch_count": 1, "patches": ["patch-a"], "metadata": []},
        {"patch_count": 1, "patches": ["patch-a"], "seams": [{"id": "seam"}]},
        {"patch_count": 1, "patches": ["patch-a"], "transform_matrix": [[float("nan")]]},
    ],
)
def test_decode_surface_shell_payload_rejects_invalid_container_payloads(payload: dict[str, object]) -> None:
    patch = PlanarSurfacePatch(family="planar")
    with pytest.raises(ImpressFormatError):
        decode_surface_shell_payload(payload, patches=[patch])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"shell_count": 1, "shells": "shell-a"},
        {"shell_count": 0, "shells": []},
        {"shell_count": 2, "shells": [{}]},
        {"shell_count": 1, "shells": [{}], "metadata": []},
        {"shell_count": 1, "shells": [{}], "transform_matrix": [[float("inf")]]},
    ],
)
def test_decode_surface_body_payload_rejects_invalid_container_payloads(payload: dict[str, object]) -> None:
    body = _single_patch_body()
    with pytest.raises(ImpressFormatError):
        decode_surface_body_payload(payload, shells=body.shells)
