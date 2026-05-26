from __future__ import annotations

import pytest

from impression.io import (
    CURRENT_IMPRESS_SCHEMA_VERSION,
    DEFAULT_IMPRESS_LENGTH_UNIT,
    IMPRESS_FORMAT,
    ImpressBodyEntry,
    ImpressFormatError,
    ImpressSaveOptions,
    ImpressUnits,
    SurfaceBodyStore,
    UnsupportedImpressSchemaVersion,
    decode_surface_adjacency_payload,
    decode_surface_boundary_ref_payload,
    decode_surface_body_payload,
    decode_surface_patch_payload,
    decode_surface_seam_payload,
    decode_surface_shell_payload,
    decode_trim_loop_payload,
    decode_impress_document_payload,
    dumps_impress_json,
    encode_surface_adjacency_payload,
    encode_surface_boundary_ref_payload,
    encode_surface_body_payload,
    encode_surface_patch_payload,
    encode_surface_seam_payload,
    encode_surface_shell_payload,
    encode_trim_loop_payload,
    make_impress_document_payload,
    make_impress_document_root,
    make_surface_body_store,
    load_impress,
    loads_impress_json,
    validate_impress_units,
    validate_impress_document_root,
    validate_surface_body_store,
    save_impress,
    write_impress_json,
)
import numpy as np
from impression.modeling.surface import (
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBoundaryRef,
    SurfaceSeam,
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


def test_make_impress_document_payload_serializes_surface_bodies() -> None:
    body = _single_patch_body()

    payload = make_impress_document_payload([body], units={"length": "mm"}, metadata={"author": "test"})

    assert payload["format"] == IMPRESS_FORMAT
    assert payload["schema_version"] == CURRENT_IMPRESS_SCHEMA_VERSION
    assert payload["units"] == {"length": "mm"}
    assert payload["metadata"] == {"author": "test"}
    assert payload["body_store"] == {
        "bodies": [{"body_id": "body-0001", "stable_identity": body.stable_identity, "body_ref": "body-0001"}]
    }
    assert list(payload["bodies"]) == ["body-0001"]  # type: ignore[arg-type]
    assert len(payload["patches"]) == 1  # type: ignore[arg-type]


def test_dumps_impress_json_is_byte_stable_for_equivalent_payloads() -> None:
    body = _single_patch_body()
    payload_a = make_impress_document_payload([body], metadata={"b": 2, "a": 1})
    payload_b = {
        "metadata": {"a": 1, "b": 2},
        "bodies": payload_a["bodies"],
        "format": payload_a["format"],
        "body_store": payload_a["body_store"],
        "patches": payload_a["patches"],
        "units": payload_a["units"],
        "schema_version": payload_a["schema_version"],
    }

    encoded_a = dumps_impress_json(payload_a)
    encoded_b = dumps_impress_json(payload_b)

    assert encoded_a == encoded_b
    assert encoded_a.endswith("\n")
    assert '"schema_version": "1.0"' in encoded_a


def test_write_and_save_impress_write_deterministic_json(tmp_path) -> None:
    body = _single_patch_body()
    payload = make_impress_document_payload([body])
    payload_path = tmp_path / "payload.impress"
    saved_path = tmp_path / "saved.impress"

    assert write_impress_json(payload, payload_path) == payload_path
    assert save_impress([body], saved_path) == saved_path

    assert payload_path.read_text(encoding="utf-8") == dumps_impress_json(payload)
    assert saved_path.read_text(encoding="utf-8") == dumps_impress_json(make_impress_document_payload([body]))


def test_load_impress_round_trips_saved_surface_bodies(tmp_path) -> None:
    patches = [
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0)),
        PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0)),
    ]
    seam = SurfaceSeam("join", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")))
    adjacency = SurfaceAdjacencyRecord(
        source=SurfaceBoundaryRef(0, "right"),
        target=SurfaceBoundaryRef(1, "left"),
        seam_id="join",
    )
    body = make_surface_body([make_surface_shell(patches, seams=(seam,), adjacency=(adjacency,))])
    path = tmp_path / "roundtrip.impress"

    save_impress([body], path, units={"length": "mm"}, metadata={"purpose": "load"})
    loaded = load_impress(path)

    assert loaded.path == path
    assert loaded.root.units == ImpressUnits(length="mm")
    assert loaded.root.metadata == {"purpose": "load"}
    assert [loaded_body.stable_identity for loaded_body in loaded.bodies] == [body.stable_identity]
    assert loaded.body_store.bodies[0].body is not None
    assert loaded.body_store.bodies[0].body.stable_identity == body.stable_identity


def test_loads_impress_json_returns_load_result_without_path() -> None:
    body = _single_patch_body()

    loaded = loads_impress_json(dumps_impress_json(make_impress_document_payload([body])))

    assert loaded.path is None
    assert [loaded_body.stable_identity for loaded_body in loaded.bodies] == [body.stable_identity]


def test_decode_impress_document_payload_rejects_invalid_body_and_patch_references() -> None:
    body = _single_patch_body()
    payload = make_impress_document_payload([body])
    body_payload = dict(payload["bodies"])  # type: ignore[arg-type]
    body_payload["extra-body"] = body_payload["body-0001"]
    payload["bodies"] = body_payload

    with pytest.raises(ImpressFormatError, match="bodies must exactly match"):
        decode_impress_document_payload(payload)

    payload = make_impress_document_payload([body])
    patches = dict(payload["patches"])  # type: ignore[arg-type]
    patches["missing-patch"] = next(iter(patches.values()))
    payload["patches"] = patches
    with pytest.raises(ImpressFormatError, match="stable_identity does not match"):
        decode_impress_document_payload(payload)


def test_load_impress_rejects_malformed_json_schema_and_invalid_payload(tmp_path) -> None:
    malformed = tmp_path / "malformed.impress"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ImpressFormatError, match="Malformed"):
        load_impress(malformed)

    unsupported = tmp_path / "unsupported.impress"
    unsupported.write_text('{"format":"impress","schema_version":"999.0"}', encoding="utf-8")
    with pytest.raises(UnsupportedImpressSchemaVersion):
        load_impress(unsupported)

    invalid = tmp_path / "invalid.impress"
    invalid.write_text(dumps_impress_json({"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(ImpressFormatError, match="body_store"):
        load_impress(invalid)


def test_load_impress_reports_read_errors(tmp_path) -> None:
    with pytest.raises(ImpressFormatError, match="Unable to read"):
        load_impress(tmp_path / "missing.impress")


def test_dumps_impress_json_accepts_save_options() -> None:
    payload = make_impress_document_payload([_single_patch_body()])

    encoded = dumps_impress_json(payload, options=ImpressSaveOptions(indent=0, trailing_newline=False))

    assert not encoded.endswith("\n")
    assert encoded.startswith("{")


def test_impress_json_writer_rejects_invalid_payload_and_write_errors(tmp_path) -> None:
    with pytest.raises(ImpressFormatError, match="root must declare format"):
        dumps_impress_json({"schema_version": CURRENT_IMPRESS_SCHEMA_VERSION})

    with pytest.raises(ImpressFormatError, match="Unable to serialize"):
        dumps_impress_json({"format": "impress", "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION, "bad": object()})

    with pytest.raises(ImpressFormatError, match="Unable to write"):
        write_impress_json(make_impress_document_payload([_single_patch_body()]), tmp_path / "missing" / "file.impress")


def test_encode_decode_surface_shell_payload_round_trips_container_fields() -> None:
    patch = PlanarSurfacePatch(family="planar", metadata={"kernel": {"name": "face"}})
    shell = make_surface_shell([patch], connected=False, metadata={"consumer": {"label": "shell"}})

    payload = encode_surface_shell_payload(shell)
    decoded = decode_surface_shell_payload(payload, patches=[patch])

    assert payload["patches"] == [patch.stable_identity]
    assert decoded.connected is False
    assert decoded.metadata == {"consumer": {"label": "shell"}}
    assert decoded.patches == (patch,)


def test_encode_decode_surface_seam_payload_round_trips_boundary_refs() -> None:
    seam = SurfaceSeam(
        "front-right",
        (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
        continuity="G1",
        metadata={"kernel": {"source": "fixture"}},
    )

    payload = encode_surface_seam_payload(seam)
    decoded = decode_surface_seam_payload(payload, patch_count=2)

    assert encode_surface_boundary_ref_payload(seam.boundaries[0]) == {"patch_index": 0, "boundary_id": "right"}
    assert payload == {
        "seam_id": "front-right",
        "boundaries": [
            {"patch_index": 0, "boundary_id": "right"},
            {"patch_index": 1, "boundary_id": "left"},
        ],
        "continuity": "G1",
        "metadata": {"kernel": {"source": "fixture"}},
    }
    assert decoded == seam


def test_encode_decode_surface_shell_payload_round_trips_seams() -> None:
    patches = [
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0)),
        PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0)),
    ]
    seam = SurfaceSeam("join", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")))
    shell = make_surface_shell(patches, seams=(seam,))

    payload = encode_surface_shell_payload(shell)
    decoded = decode_surface_shell_payload(payload, patches=patches)

    assert payload["seams"] == [encode_surface_seam_payload(seam)]
    assert decoded.seams == (seam,)


def test_encode_decode_surface_adjacency_payload_round_trips_refs() -> None:
    record = SurfaceAdjacencyRecord(
        source=SurfaceBoundaryRef(0, "right"),
        target=SurfaceBoundaryRef(1, "left"),
        seam_id="join",
        continuity="G1",
    )

    payload = encode_surface_adjacency_payload(record)
    decoded = decode_surface_adjacency_payload(payload, patch_count=2, seam_ids=("join",))

    assert payload == {
        "source": {"patch_index": 0, "boundary_id": "right"},
        "target": {"patch_index": 1, "boundary_id": "left"},
        "seam_id": "join",
        "continuity": "G1",
    }
    assert decoded == record


def test_encode_decode_surface_shell_payload_round_trips_adjacency() -> None:
    patches = [
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0)),
        PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0)),
    ]
    seam = SurfaceSeam("join", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")))
    adjacency = SurfaceAdjacencyRecord(
        source=SurfaceBoundaryRef(0, "right"),
        target=SurfaceBoundaryRef(1, "left"),
        seam_id="join",
    )
    shell = make_surface_shell(patches, seams=(seam,), adjacency=(adjacency,))

    payload = encode_surface_shell_payload(shell)
    decoded = decode_surface_shell_payload(payload, patches=patches)

    assert payload["adjacency"] == [encode_surface_adjacency_payload(adjacency)]
    assert decoded.adjacency == (adjacency,)


def test_decode_surface_adjacency_payload_accepts_open_boundary_without_target() -> None:
    record = SurfaceAdjacencyRecord(source=SurfaceBoundaryRef(0, "left"), target=None)

    assert decode_surface_adjacency_payload(encode_surface_adjacency_payload(record), patch_count=1) == record


def test_decode_surface_seam_payload_rejects_missing_boundary_reference() -> None:
    payload = encode_surface_seam_payload(
        SurfaceSeam("bad-ref", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(2, "left")))
    )

    with pytest.raises(ImpressFormatError, match="outside the loaded shell"):
        decode_surface_seam_payload(payload, patch_count=2)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"patch_index": -1, "boundary_id": "left"},
        {"patch_index": True, "boundary_id": "left"},
        {"patch_index": 0, "boundary_id": "   "},
        {"patch_index": 0, "boundary_id": "left", "side": "outer"},
    ],
)
def test_decode_surface_boundary_ref_payload_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        decode_surface_boundary_ref_payload(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"seam_id": "   ", "boundaries": [{"patch_index": 0, "boundary_id": "left"}]},
        {"seam_id": "open", "boundaries": []},
        {
            "seam_id": "too-many",
            "boundaries": [
                {"patch_index": 0, "boundary_id": "left"},
                {"patch_index": 1, "boundary_id": "right"},
                {"patch_index": 2, "boundary_id": "top"},
            ],
        },
        {
            "seam_id": "dup",
            "boundaries": [{"patch_index": 0, "boundary_id": "left"}, {"patch_index": 0, "boundary_id": "left"}],
        },
        {"seam_id": "bad-continuity", "boundaries": [{"patch_index": 0, "boundary_id": "left"}], "continuity": " "},
        {"seam_id": "bad-metadata", "boundaries": [{"patch_index": 0, "boundary_id": "left"}], "metadata": []},
    ],
)
def test_decode_surface_seam_payload_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        decode_surface_seam_payload(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"source": {"patch_index": 2, "boundary_id": "right"}, "target": None},
        {"source": {"patch_index": 0, "boundary_id": "right"}, "target": {"patch_index": 2, "boundary_id": "left"}},
        {"source": {"patch_index": 0, "boundary_id": "right"}, "target": None, "seam_id": "missing"},
        {"source": {"patch_index": 0, "boundary_id": "right"}, "target": None, "seam_id": " "},
        {"source": {"patch_index": 0, "boundary_id": "right"}, "target": None, "continuity": " "},
        {"source": {"patch_index": 0, "boundary_id": "right"}, "target": None, "kind": "extra"},
    ],
)
def test_decode_surface_adjacency_payload_rejects_invalid_references(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        decode_surface_adjacency_payload(payload, patch_count=2, seam_ids=("join",))


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


def test_encode_decode_trim_loop_payload_round_trips_normalized_orientation() -> None:
    outer = TrimLoop(points_uv=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], category="outer")
    inner = TrimLoop(points_uv=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], category="inner")

    outer_payload = encode_trim_loop_payload(outer)
    inner_payload = encode_trim_loop_payload(inner)

    assert outer_payload["category"] == "outer"
    assert outer_payload["clockwise"] is False
    assert inner_payload["category"] == "inner"
    assert inner_payload["clockwise"] is True
    assert decode_trim_loop_payload(outer_payload).canonical_payload()["category"] == "outer"
    assert decode_trim_loop_payload(inner_payload).canonical_payload()["category"] == "inner"


def test_surface_patch_payload_codec_round_trips_trim_loops() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        trim_loops=(
            TrimLoop(points_uv=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], category="outer"),
            TrimLoop(points_uv=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], category="inner"),
        ),
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert len(payload["trim_loops"]) == 2
    assert decoded.stable_identity == patch.stable_identity
    assert [trim.category for trim in decoded.trim_loops] == ["outer", "inner"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"category": "mystery", "points_uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]},
        {"category": "outer", "points_uv": [[0.0, 0.0], [1.0, 0.0]]},
        {"category": "outer", "points_uv": [[float("inf"), 0.0], [1.0, 0.0], [0.0, 1.0]]},
        {"category": "outer", "points_uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "clockwise": "no"},
        {"category": "outer", "points_uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "clockwise": True},
    ],
)
def test_decode_trim_loop_payload_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ImpressFormatError):
        decode_trim_loop_payload(payload)


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
