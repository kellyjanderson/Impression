from __future__ import annotations

import json

import pytest

import impression.io.impress as impress_io
from impression.io import (
    CURRENT_IMPRESS_SCHEMA_VERSION,
    DEFAULT_IMPRESS_LENGTH_UNIT,
    IMPRESS_FORMAT,
    ImpressBodyEntry,
    ImpressFormatError,
    ImpressSaveOptions,
    ImpressUnits,
    ImpressWholeStoreFixtureCoverageReport,
    DisplacementCSGImpressPayloadRecord,
    DisplacementCSGImpressRoundTripDiagnostic,
    HeightmapCSGImpressPayloadRecord,
    HeightmapCSGImpressRoundTripDiagnostic,
    ImplicitCSGImpressPayloadRecord,
    ImplicitCSGImpressRoundTripDiagnostic,
    SampledImplicitPromotionImpressPayloadRecord,
    SampledImplicitPromotionImpressRoundTripDiagnostic,
    InvalidSurfaceWrapperDiagnostic,
    IMPRESS_DIAGNOSTIC_METADATA_FIELDS,
    SurfaceBodyStore,
    SurfacePatchBasePayload,
    UnsupportedImpressSchemaVersion,
    atomic_write_text,
    assert_impress_patch_codec_coverage_for_available_families,
    assert_impress_whole_store_fixture_coverage,
    decode_surface_adjacency_payload,
    decode_surface_boundary_ref_payload,
    decode_surface_body_payload,
    decode_surface_patch_base_payload,
    decode_surface_patch_payload,
    decode_surface_seam_payload,
    decode_surface_shell_payload,
    decode_trim_loop_payload,
    decode_impress_document_payload,
    dumps_impress_json,
    encode_displacement_csg_impress_payload,
    encode_heightmap_csg_impress_payload,
    encode_implicit_csg_impress_payload,
    encode_sampled_implicit_promotion_impress_payload,
    encode_surface_adjacency_payload,
    encode_surface_boundary_ref_payload,
    encode_surface_body_payload,
    encode_surface_patch_base_payload,
    encode_surface_patch_payload,
    encode_surface_seam_payload,
    encode_surface_shell_payload,
    encode_trim_loop_payload,
    make_impress_document_payload,
    make_impress_document_root,
    make_surface_body_store,
    inspect_impress_patch_family_dispatch,
    inspect_impress_patch_codec_coverage,
    inspect_impress_whole_store_fixture_coverage,
    displacement_csg_impress_payload_record,
    heightmap_csg_impress_payload_record,
    implicit_csg_impress_payload_record,
    sampled_implicit_promotion_impress_payload_record,
    load_impress,
    loads_impress_json,
    validate_impress_units,
    validate_impress_document_root,
    validate_surface_patch_serialization_guard,
    validate_surface_body_store,
    verify_displacement_csg_impress_round_trip,
    verify_heightmap_csg_impress_round_trip,
    verify_implicit_csg_impress_round_trip,
    verify_sampled_implicit_promotion_impress_round_trip,
    save_impress,
    write_impress_json,
)
import numpy as np
from impression.modeling.path3d import Path3D
from impression.modeling.surface import (
    BSplineSurfacePatch,
    DisplacementSurfacePatch,
    HeightmapSurfacePatch,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SubdivisionCrease,
    SubdivisionSurfacePatch,
    SweepSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBoundaryRef,
    SurfaceSeam,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
    make_implicit_field_node,
)
from impression.modeling import (
    adapt_surface_patch_to_implicit_field,
    build_sampled_implicit_promotion_matrix,
    build_sampled_implicit_promotion_provenance_record,
    compose_displacement_csg_result,
    compose_heightmap_csg_result,
    compose_implicit_field_csg_result,
)
from tests.reference_images import (
    ExpectedDiagnosticKeyRecord,
    NegativeDiagnosticFixtureRecord,
    evaluate_negative_diagnostic_fixture_matrix,
    normalize_diagnostic_snapshot,
)


def _impress_negative_fixture(
    fixture_id: str,
    operation,
    *,
    expected_code: str = "ImpressFormatError",
) -> NegativeDiagnosticFixtureRecord:
    try:
        operation()
    except Exception as exc:  # noqa: BLE001 - negative fixture runner records refusal shape.
        snapshot = normalize_diagnostic_snapshot(
            {
                "code": type(exc).__name__,
                "message": str(exc),
            },
            fixture_id=fixture_id,
        )
    else:
        raise AssertionError(f".impress negative fixture {fixture_id!r} did not refuse.")
    return NegativeDiagnosticFixtureRecord(
        fixture_id=fixture_id,
        domain=".impress",
        expected_keys=(
            ExpectedDiagnosticKeyRecord(("code",), expected_code),
            ExpectedDiagnosticKeyRecord(("message",)),
        ),
        expected_snapshot=snapshot,
    )


def test_make_impress_document_root_declares_format_and_schema() -> None:
    root = make_impress_document_root(metadata={"author": "test"})

    assert root.to_json_object() == {
        "format": IMPRESS_FORMAT,
        "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION,
        "units": {"length": DEFAULT_IMPRESS_LENGTH_UNIT},
        "metadata": {"author": "test"},
    }


def test_impress_patch_codec_coverage_inventory_covers_available_families() -> None:
    records = assert_impress_patch_codec_coverage_for_available_families()

    by_family = {record.family: record for record in records}
    assert by_family["planar"].patch_kind == "PlanarSurfacePatch"
    assert by_family["ruled"].patch_kind == "RuledSurfacePatch"
    assert by_family["revolution"].patch_kind == "RevolutionSurfacePatch"
    assert by_family["planar"].required_for_available is True
    assert by_family["heightmap"].required_for_available is True
    assert by_family["heightmap"].covered is True
    assert by_family["displacement"].covered is True


def test_impress_patch_codec_coverage_inventory_reports_supported_family_list() -> None:
    records = inspect_impress_patch_codec_coverage()

    assert {record.family for record in records} == {
        "planar",
        "ruled",
        "revolution",
        "bspline",
        "nurbs",
        "sweep",
        "subdivision",
        "implicit",
        "heightmap",
        "displacement",
    }


def test_impress_patch_family_dispatch_inventory_is_allow_listed() -> None:
    records = inspect_impress_patch_family_dispatch()

    assert {record.kind for record in records} == {
        "PlanarSurfacePatch",
        "RuledSurfacePatch",
        "RevolutionSurfacePatch",
        "BSplineSurfacePatch",
        "NURBSSurfacePatch",
        "SweepSurfacePatch",
        "SubdivisionSurfacePatch",
        "ImplicitSurfacePatch",
        "HeightmapSurfacePatch",
        "DisplacementSurfacePatch",
    }
    assert {record.family for record in records} == {
        "planar",
        "ruled",
        "revolution",
        "bspline",
        "nurbs",
        "sweep",
        "subdivision",
        "implicit",
        "heightmap",
        "displacement",
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


def _round_trip_fixture_body() -> object:
    planar = PlanarSurfacePatch(
        family="planar",
        metadata={"kernel": {"face": "planar"}},
        trim_loops=(TrimLoop(points_uv=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], category="outer"),),
    )
    ruled = RuledSurfacePatch(
        family="ruled",
        metadata={"consumer": {"label": "guide"}},
        start_curve=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        end_curve=[(1.0, 0.0, 0.25), (1.0, 1.0, 0.25)],
    )
    seam = SurfaceSeam(
        "planar-ruled",
        (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
        metadata={"kernel": {"continuity_source": "authored"}},
    )
    adjacency = SurfaceAdjacencyRecord(
        source=SurfaceBoundaryRef(0, "right"),
        target=SurfaceBoundaryRef(1, "left"),
        seam_id="planar-ruled",
    )
    return make_surface_body(
        [make_surface_shell([planar, ruled], seams=(seam,), adjacency=(adjacency,), metadata={"shell": "primary"})],
        metadata={"body": "fixture"},
    )


def _codec_covered_patch_fixtures() -> tuple[object, ...]:
    control_net = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 1.0, 0.4], [1.0, 2.0, 0.1]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.2], [2.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    field = make_implicit_field_node("sphere", parameters={"center": (0.0, 0.0, 0.0), "radius": 0.75})
    return (
        PlanarSurfacePatch(
            family="planar",
            metadata={"kernel": {"fixture": "planar"}},
            trim_loops=(TrimLoop(points_uv=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], category="outer"),),
        ),
        RuledSurfacePatch(
            family="ruled",
            start_curve=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            end_curve=[(1.0, 0.0, 0.25), (1.0, 1.0, 0.25)],
        ),
        RevolutionSurfacePatch(
            family="revolution",
            profile_curve=[(1.0, 0.0, 0.0), (1.5, 0.0, 2.0)],
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            sweep_angle_deg=180.0,
        ),
        BSplineSurfacePatch(
            family="bspline",
            degree_u=2,
            degree_v=2,
            knots_u=knots,
            knots_v=knots,
            control_net=control_net,
        ),
        NURBSSurfacePatch(
            family="nurbs",
            degree_u=2,
            degree_v=2,
            knots_u=knots,
            knots_v=knots,
            control_net=control_net,
            weights=np.ones((3, 3), dtype=float),
        ),
        SweepSurfacePatch(
            family="sweep",
            profile_points_uv=[(0.0, 0.0), (0.5, 0.25), (1.0, 0.0)],
            path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.5, 1.0), (0.0, 0.0, 2.0)]),
            frame_policy="fixed",
        ),
        HeightmapSurfacePatch(
            family="heightmap",
            height_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
            alpha_mask=np.array([[True, True], [False, True]], dtype=bool),
            alpha_mode="mask",
            xy_scale=(0.5, 0.25),
            center=(1.0, 2.0, 3.0),
            height_scale=2.0,
        ),
        DisplacementSurfacePatch(
            family="displacement",
            source_patch=PlanarSurfacePatch(family="planar", metadata={"kernel": {"fixture": "source"}}),
            displacement_samples=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            alpha_mask=np.array([[True, True], [False, True]], dtype=bool),
            alpha_mode="ignore",
            height_scale=0.5,
            direction="z",
            projection="planar",
            plane="xy",
            projection_bounds=(-1.0, 1.0, -1.0, 1.0),
        ),
        SubdivisionSurfacePatch(
            family="subdivision",
            control_points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.1), (0.0, 1.0, 0.0)],
            faces=((0, 1, 2, 3),),
            subdivision_level=1,
        ),
        ImplicitSurfacePatch(
            family="implicit",
            field=field,
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        ),
    )


def test_encode_surface_patch_base_payload_preserves_identity_and_metadata_without_geometry() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        metadata={"kernel": {"face": "base"}, "consumer": {"name": "panel"}},
        capability_flags=frozenset({"analytic"}),
        trim_loops=(TrimLoop(points_uv=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], category="outer"),),
    )

    payload = encode_surface_patch_base_payload(patch)

    assert payload["stable_identity"] == patch.stable_identity
    assert payload["kind"] == "PlanarSurfacePatch"
    assert payload["family"] == "planar"
    assert payload["metadata"] == patch.metadata
    assert payload["capability_flags"] == ["analytic"]
    assert "geometry" not in payload


def test_decode_surface_patch_base_payload_returns_typed_bundle_before_family_dispatch() -> None:
    patch = PlanarSurfacePatch(family="planar", metadata={"kernel": {"face": "base"}})
    payload = encode_surface_patch_payload(patch)

    base = decode_surface_patch_base_payload(payload)

    assert isinstance(base, SurfacePatchBasePayload)
    assert base.kind == "PlanarSurfacePatch"
    assert base.family == "planar"
    assert base.stable_identity == patch.stable_identity
    assert base.metadata == patch.metadata
    assert base.geometry == payload["geometry"]
    assert base.constructor_kwargs()["metadata"] == patch.metadata


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("kind"), "requires a non-empty kind"),
        (lambda payload: payload.update({"stable_identity": ""}), "stable_identity must be a non-empty string"),
        (lambda payload: payload.update({"metadata": []}), "metadata must be an object"),
        (lambda payload: payload.update({"capability_flags": ["ok", ""]}), "capability_flags must contain"),
        (lambda payload: payload.update({"transform_matrix": [[1.0]]}), "transform_matrix must be a 4x4"),
    ],
)
def test_decode_surface_patch_base_payload_refuses_malformed_base_fields(mutate, message) -> None:  # noqa: ANN001
    payload = encode_surface_patch_payload(PlanarSurfacePatch(family="planar"))
    mutate(payload)

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_base_payload(payload)


def test_decode_surface_patch_payload_refuses_stable_identity_mismatch() -> None:
    payload = encode_surface_patch_payload(PlanarSurfacePatch(family="planar"))
    payload["metadata"] = {"kernel": {"changed": True}}

    with pytest.raises(ImpressFormatError, match="stable_identity does not match"):
        decode_surface_patch_payload(payload)


def test_decode_surface_patch_payload_refuses_unknown_family_before_geometry_interpretation() -> None:
    payload = encode_surface_patch_payload(PlanarSurfacePatch(family="planar"))
    payload["kind"] = "MysterySurfacePatch"
    payload["geometry"] = {"unexpected": "family-specific"}

    with pytest.raises(ImpressFormatError, match="Unsupported SurfacePatch kind"):
        decode_surface_patch_payload(payload)


def _all_codec_covered_family_body() -> object:
    return make_surface_body(
        [
            make_surface_shell(
                _codec_covered_patch_fixtures(),
                seams=(
                    SurfaceSeam(
                        "fixture-planar-ruled",
                        (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
                        metadata={"kernel": {"continuity_source": "authored"}},
                    ),
                ),
                adjacency=(
                    SurfaceAdjacencyRecord(
                        source=SurfaceBoundaryRef(0, "right"),
                        target=SurfaceBoundaryRef(1, "left"),
                        seam_id="fixture-planar-ruled",
                    ),
                ),
                metadata={"fixture": "all-codec-covered"},
            )
        ],
        metadata={"body": "all-codec-covered"},
    )


def _assert_loaded_body_preserves_identity_and_metadata(loaded_body: object, expected_body: object) -> None:
    assert loaded_body.stable_identity == expected_body.stable_identity
    assert loaded_body.metadata == expected_body.metadata
    assert loaded_body.shells[0].metadata == expected_body.shells[0].metadata
    assert [patch.metadata for patch in loaded_body.shells[0].patches] == [
        patch.metadata for patch in expected_body.shells[0].patches
    ]
    assert loaded_body.shells[0].seams[0].metadata == expected_body.shells[0].seams[0].metadata


def test_impress_all_codec_covered_families_round_trip_preserves_identity_and_families() -> None:
    body = _all_codec_covered_family_body()

    loaded = loads_impress_json(dumps_impress_json(make_impress_document_payload([body])))

    loaded_body = loaded.bodies[0]
    expected_patches = body.shells[0].patches
    loaded_patches = loaded_body.shells[0].patches
    assert loaded_body.stable_identity == body.stable_identity
    assert [patch.family for patch in loaded_patches] == [patch.family for patch in expected_patches]
    assert [patch.stable_identity for patch in loaded_patches] == [patch.stable_identity for patch in expected_patches]
    assert {patch.family for patch in loaded_patches} == {
        record.family for record in inspect_impress_patch_codec_coverage() if record.covered
    }


def test_impress_whole_store_fixture_coverage_gate_accepts_complete_surface_truth() -> None:
    body = _all_codec_covered_family_body()
    payload = make_impress_document_payload(
        [body],
        metadata={
            "topology_rails": [{"id": "rail.outer", "anchors": ["station0.p0", "station1.p0"]}],
            "lifecycle_records": [{"entity": "point.notch", "event": "birth"}],
            "operation_provenance": [{"operation": "fixture-loft", "producer": "surface"}],
        },
    )

    report = assert_impress_whole_store_fixture_coverage(payload)
    loaded = loads_impress_json(dumps_impress_json(payload))

    assert isinstance(report, ImpressWholeStoreFixtureCoverageReport)
    assert report.covered is True
    assert set(report.covered_families) == set(report.required_families)
    assert report.covered_store_areas == ("bodies", "body_store", "patches")
    assert loaded.root.metadata["operation_provenance"][0]["producer"] == "surface"
    assert loaded.bodies[0].shells[0].seams[0].metadata == {"kernel": {"continuity_source": "authored"}}
    assert loaded.bodies[0].shells[0].adjacency[0].seam_id == "fixture-planar-ruled"


def test_impress_whole_store_fixture_coverage_gate_reports_missing_payloads() -> None:
    payload = make_impress_document_payload([_round_trip_fixture_body()])
    patches = dict(payload["patches"])  # type: ignore[arg-type]
    patches.pop(next(iter(patches)))
    payload["patches"] = patches
    payload["mesh"] = {"vertices": [], "faces": []}

    report = inspect_impress_whole_store_fixture_coverage(payload)

    assert report.covered is False
    assert "metadata.topology_rails" in report.missing_payloads
    assert "patches.family[implicit]" in report.missing_payloads
    assert report.mesh_truth_present is True
    with pytest.raises(ImpressFormatError, match="mesh truth payload"):
        assert_impress_whole_store_fixture_coverage(payload)


def _invalid_impress_file(path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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


def test_impress_refuses_heightmap_triangle_wrapper_surface_truth() -> None:
    wrapper_patch = PlanarSurfacePatch(
        family="planar",
        metadata={"kernel": {"producer": "heightmap", "triangle_face_index": 0}},
    )
    body = make_surface_body((make_surface_shell((wrapper_patch,), connected=False),))
    diagnostic = validate_surface_patch_serialization_guard(wrapper_patch)

    assert isinstance(diagnostic, InvalidSurfaceWrapperDiagnostic)
    assert diagnostic.to_json_object()["producer"] == "heightmap"
    with pytest.raises(ImpressFormatError, match="mesh-derived surface wrapper"):
        encode_surface_patch_payload(wrapper_patch)
    with pytest.raises(ImpressFormatError, match="heightmap triangle wrappers"):
        make_impress_document_payload([body])


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


def test_atomic_write_text_replaces_existing_file_without_temp_litter(tmp_path) -> None:
    path = tmp_path / "model.impress"
    path.write_text("old", encoding="utf-8")

    assert atomic_write_text(path, "new") == path

    assert path.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob(".model.impress.*.tmp")) == []


def test_atomic_write_text_cleans_up_temp_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.impress"
    path.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):  # noqa: ANN001
        raise OSError("replace failed")

    monkeypatch.setattr(impress_io.os, "replace", fail_replace)

    with pytest.raises(ImpressFormatError, match="Unable to write"):
        atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".model.impress.*.tmp")) == []


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


def test_impress_acceptance_round_trip_preserves_identity_and_metadata(tmp_path) -> None:
    body = _round_trip_fixture_body()
    path = tmp_path / "acceptance.impress"

    save_impress([body], path, metadata={"document": "acceptance"})
    loaded = load_impress(path)

    assert loaded.root.metadata == {"document": "acceptance"}
    assert len(loaded.bodies) == 1
    _assert_loaded_body_preserves_identity_and_metadata(loaded.bodies[0], body)
    assert path.read_text(encoding="utf-8") == dumps_impress_json(make_impress_document_payload([body], metadata={"document": "acceptance"}))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"mesh": {"vertices": [], "faces": []}}), "Unsupported `.impress` document fields"),
        (
            lambda payload: payload["patches"].update(  # type: ignore[index,union-attr]
                {
                    "implicit-patch": {
                        "kind": "ImplicitSurfacePatch",
                        "family": "implicit",
                        "domain": {"u_range": [0.0, 1.0], "v_range": [0.0, 1.0], "normalized": True},
                        "capability_flags": [],
                        "transform_matrix": np.eye(4).tolist(),
                        "metadata": {"code": "lambda x: x"},
                        "trim_loops": [],
                        "geometry": {"expression": "exec('unsafe')"},
                    }
                }
            ),
            "Unsupported ImplicitSurfacePatch geometry fields",
        ),
        (
            lambda payload: payload["body_store"]["bodies"][0].update({"stable_identity": "not-the-body"}),  # type: ignore[index,union-attr]
            "stable_identity does not match",
        ),
    ],
)
def test_impress_acceptance_invalid_files_refuse_explicitly(tmp_path, mutate, message) -> None:
    payload = make_impress_document_payload([_round_trip_fixture_body()])
    mutate(payload)
    path = tmp_path / "invalid.impress"
    _invalid_impress_file(path, payload)

    with pytest.raises(ImpressFormatError, match=message):
        load_impress(path)


def test_impress_invalid_payload_diagnostics_are_deterministic_and_surface_native(tmp_path) -> None:
    payload = make_impress_document_payload([_round_trip_fixture_body()])
    payload["metadata"] = {"diagnostics": [{"code": "invalid", "message": "broken", "unknown": "field"}]}
    path = tmp_path / "invalid-diagnostic.impress"
    _invalid_impress_file(path, payload)

    messages = []
    for _ in range(2):
        with pytest.raises(ImpressFormatError) as exc_info:
            load_impress(path)
        messages.append(str(exc_info.value))

    assert IMPRESS_DIAGNOSTIC_METADATA_FIELDS == frozenset({"code", "message", "path", "severity"})
    assert messages == [
        "Unsupported `.impress` diagnostic metadata fields at diagnostics[0]: unknown.",
        "Unsupported `.impress` diagnostic metadata fields at diagnostics[0]: unknown.",
    ]
    assert "mesh" not in messages[0]


def test_impress_unsafe_payload_refusal_does_not_recover_as_mesh_or_mutate_payload() -> None:
    payload = make_impress_document_payload([_all_codec_covered_family_body()])
    patch_payload = next(
        patch for patch in payload["patches"].values()  # type: ignore[union-attr]
        if isinstance(patch, dict) and patch.get("kind") == "ImplicitSurfacePatch"
    )
    geometry = dict(patch_payload["geometry"])
    field = dict(geometry["field"])
    field["parameters"] = {"script": "lambda x: x"}
    geometry["field"] = field
    patch_payload["geometry"] = geometry
    before = dumps_impress_json(payload)

    with pytest.raises(ImpressFormatError, match="Unsafe implicit field payload"):
        loads_impress_json(before)

    assert dumps_impress_json(payload) == before
    assert "mesh" not in payload


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
    assert payload["geometry"]["payload_version"] == 1
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

    ruled_payload = encode_surface_patch_payload(ruled)
    revolution_payload = encode_surface_patch_payload(revolution)

    assert ruled_payload["geometry"]["payload_version"] == 1
    assert revolution_payload["geometry"]["payload_version"] == 1
    assert decode_surface_patch_payload(ruled_payload).stable_identity == ruled.stable_identity
    assert decode_surface_patch_payload(revolution_payload).stable_identity == revolution.stable_identity


def test_encode_decode_spline_surface_patch_payloads_round_trip() -> None:
    control_net = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 1.0, 0.4], [1.0, 2.0, 0.1]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.2], [2.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    domain = ParameterDomain(u_range=(0.0, 1.0), v_range=(0.0, 1.0))
    bspline = BSplineSurfacePatch(
        family="bspline",
        domain=domain,
        degree_u=2,
        degree_v=2,
        knots_u=knots,
        knots_v=knots,
        control_net=control_net,
    )
    nurbs = NURBSSurfacePatch(
        family="nurbs",
        domain=domain,
        degree_u=2,
        degree_v=2,
        knots_u=knots,
        knots_v=knots,
        control_net=control_net,
        weights=np.array([[1.0, 1.25, 1.0], [1.0, 2.0, 1.0], [1.0, 1.25, 1.0]], dtype=float),
    )

    bspline_payload = encode_surface_patch_payload(bspline)
    nurbs_payload = encode_surface_patch_payload(nurbs)

    assert bspline_payload["geometry"] == {
        "payload_version": 1,
        "degree_u": 2,
        "degree_v": 2,
        "knots_u": list(knots),
        "knots_v": list(knots),
        "control_net": control_net.tolist(),
    }
    assert nurbs_payload["geometry"]["payload_version"] == 1
    assert nurbs_payload["geometry"]["weights"] == nurbs.weights.tolist()
    assert decode_surface_patch_payload(bspline_payload).stable_identity == bspline.stable_identity
    assert decode_surface_patch_payload(nurbs_payload).stable_identity == nurbs.stable_identity


def test_encode_decode_sweep_surface_patch_payload_round_trips_profile_path_and_frame() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.5, 1.0), (0.0, 0.0, 2.0)])
    patch = SweepSurfacePatch(
        family="sweep",
        profile_points_uv=[(0.0, 0.0), (0.5, 0.2), (1.0, 0.0)],
        path=path,
        frame_policy="fixed",
        profile_reference="profile:outer",
        path_reference="path:centerline",
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["geometry"] == {
        "payload_version": 1,
        "profile_points_uv": patch.profile_points_uv.tolist(),
        "path_points": path.sample().tolist(),
        "frame_policy": "fixed",
        "profile_reference": "profile:outer",
        "path_reference": "path:centerline",
    }
    assert decoded.stable_identity == patch.stable_identity
    assert isinstance(decoded, SweepSurfacePatch)
    assert decoded.frame_policy == "fixed"
    assert decoded.profile_reference == "profile:outer"
    assert decoded.path_reference == "path:centerline"


def test_encode_decode_subdivision_surface_patch_payload_round_trips_cage_creases_and_level() -> None:
    patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.2),
            (0.0, 1.0, 0.0),
        ],
        faces=((0, 1, 2, 3),),
        creases=(SubdivisionCrease((1, 0), sharpness=2.5),),
        subdivision_level=2,
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["geometry"] == {
        "payload_version": 1,
        "scheme": "catmull_clark",
        "subdivision_level": 2,
        "control_points": patch.control_points.tolist(),
        "faces": [[0, 1, 2, 3]],
        "creases": [{"edge": [0, 1], "sharpness": 2.5}],
    }
    assert decoded.stable_identity == patch.stable_identity
    assert isinstance(decoded, SubdivisionSurfacePatch)
    assert decoded.faces == ((0, 1, 2, 3),)
    assert decoded.creases == (SubdivisionCrease((0, 1), sharpness=2.5),)


def test_encode_decode_implicit_surface_patch_payload_round_trips_safe_field_tree() -> None:
    field = make_implicit_field_node(
        "union",
        children=(
            make_implicit_field_node("sphere", parameters={"center": (0.0, 0.0, 0.0), "radius": 0.75}),
            make_implicit_field_node("box", parameters={"center": (0.5, 0.0, 0.0), "half_extents": (0.25, 0.5, 0.5)}),
        ),
    )
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=field,
        bounds=(-1.0, 1.0, -1.5, 1.5, -2.0, 2.0),
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["geometry"] == {
        "payload_version": 1,
        "field": field.canonical_payload(),
        "bounds": [-1.0, 1.0, -1.5, 1.5, -2.0, 2.0],
    }
    assert decoded.stable_identity == patch.stable_identity
    assert isinstance(decoded, ImplicitSurfacePatch)
    assert decoded.bounds == (-1.0, 1.0, -1.5, 1.5, -2.0, 2.0)


def test_decode_implicit_surface_patch_payload_preserves_metadata_domain_and_identity() -> None:
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=make_implicit_field_node("sphere", parameters={"radius": 0.75}),
        domain=ParameterDomain(u_range=(-1.0, 1.0), v_range=(-2.0, 2.0), normalized=False),
        metadata={"kernel": {"fixture": "implicit-safe"}, "consumer": {"label": "field"}},
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["stable_identity"] == patch.stable_identity
    assert decoded.stable_identity == patch.stable_identity
    assert decoded.domain == patch.domain
    assert decoded.metadata == patch.metadata


def test_implicit_csg_impress_payload_round_trips_composed_field_and_provenance() -> None:
    left = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
    )
    right = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.5)),
        bounds=(-1.0, 1.0, -1.0, 1.0, 0.4, 0.6),
    )
    result = compose_implicit_field_csg_result("union", (left, right), samples=(3, 3, 3), max_sample_count=27)

    payload = encode_implicit_csg_impress_payload(result)
    loaded = decode_impress_document_payload(payload)
    diagnostic = verify_implicit_csg_impress_round_trip(result)

    assert len(loaded.bodies) == 1
    assert isinstance(diagnostic, ImplicitCSGImpressRoundTripDiagnostic)
    assert diagnostic.supported is True
    assert diagnostic.before == diagnostic.after
    assert diagnostic.before is not None
    assert isinstance(diagnostic.before, ImplicitCSGImpressPayloadRecord)
    assert diagnostic.before.operation == "implicit-csg-union"
    assert diagnostic.before.field_root_kind == "union"
    assert diagnostic.before.no_mesh_fallback is True
    assert payload["metadata"]["surface_csg_payload"]["kind"] == "implicit-csg"


def test_implicit_csg_impress_payload_refuses_malformed_or_mesh_truth_metadata() -> None:
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=make_implicit_field_node("sphere"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        metadata={
            "kernel": {
                "operation": "implicit-csg-union",
                "source_operand_ids": ("a",),
                "no_mesh_fallback": False,
            }
        },
    )
    body = make_surface_body((make_surface_shell((patch,), connected=True),))

    diagnostic = verify_implicit_csg_impress_round_trip(body)

    assert diagnostic.supported is False
    assert diagnostic.code == "implicit-csg-impress-roundtrip-failed"
    assert "no_mesh_fallback=true" in diagnostic.message
    with pytest.raises(ImpressFormatError, match="no_mesh_fallback=true"):
        implicit_csg_impress_payload_record(body)


def test_heightmap_csg_impress_payload_round_trips_composition_metadata() -> None:
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[1.0, 0.5], [1.5, 4.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )
    result = compose_heightmap_csg_result("union", left, right)

    payload = encode_heightmap_csg_impress_payload(result)
    loaded = decode_impress_document_payload(payload)
    diagnostic = verify_heightmap_csg_impress_round_trip(result)
    record = heightmap_csg_impress_payload_record(result.body)

    assert len(loaded.bodies) == 1
    assert isinstance(record, HeightmapCSGImpressPayloadRecord)
    assert record.operation == "union"
    assert record.sample_shape == (2, 2)
    assert record.resample_kernel == "none"
    assert record.lossiness == "lossless"
    assert record.no_mesh_fallback is True
    assert isinstance(diagnostic, HeightmapCSGImpressRoundTripDiagnostic)
    assert diagnostic.supported is True
    assert diagnostic.before == diagnostic.after
    assert payload["metadata"]["surface_csg_payload"]["kind"] == "heightmap-csg"


def test_heightmap_csg_impress_payload_refuses_malformed_or_mesh_truth_metadata() -> None:
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        metadata={
            "kernel": {
                "heightmap_csg_composition": {
                    "operation": "union",
                    "operand_ids": ("a", "b"),
                    "sample_shape": (2, 2),
                    "resample_kernel": "none",
                    "lossiness": "lossless",
                    "alignment": {"clipping": {}},
                    "projection_frame": {"plane": "xy"},
                    "no_mesh_fallback": False,
                }
            }
        },
    )
    body = make_surface_body((make_surface_shell((patch,), connected=False),))

    diagnostic = verify_heightmap_csg_impress_round_trip(body)

    assert diagnostic.supported is False
    assert diagnostic.code == "heightmap-csg-impress-roundtrip-failed"
    assert "no_mesh_fallback=true" in diagnostic.message
    with pytest.raises(ImpressFormatError, match="no_mesh_fallback=true"):
        heightmap_csg_impress_payload_record(body)


def test_displacement_csg_impress_payload_round_trips_composition_metadata() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"fixture_source": "shared"})
    left = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    right = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.5, 0.1], [0.25, 1.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    result = compose_displacement_csg_result("union", left, right)

    payload = encode_displacement_csg_impress_payload(result)
    loaded = decode_impress_document_payload(payload)
    diagnostic = verify_displacement_csg_impress_round_trip(result)
    record = displacement_csg_impress_payload_record(result.body)

    assert len(loaded.bodies) == 1
    assert isinstance(record, DisplacementCSGImpressPayloadRecord)
    assert record.operation == "union"
    assert record.sample_shape == (2, 2)
    assert record.resample_kernel == "none"
    assert record.lossiness == "lossless"
    assert record.source_patch_id == source.stable_identity
    assert record.no_mesh_fallback is True
    assert isinstance(diagnostic, DisplacementCSGImpressRoundTripDiagnostic)
    assert diagnostic.supported is True
    assert diagnostic.before == diagnostic.after
    assert payload["metadata"]["surface_csg_payload"]["kind"] == "displacement-csg"


def test_displacement_csg_impress_payload_refuses_malformed_or_mesh_truth_metadata() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"fixture_source": "shared"})
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.ones((2, 2), dtype=float),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
        metadata={
            "kernel": {
                "displacement_csg_composition": {
                    "operation": "union",
                    "operand_ids": ("a", "b"),
                    "source_patch_id": source.stable_identity,
                    "sample_shape": (2, 2),
                    "resampling": {"resample_kernel": "none", "lossiness": "lossless"},
                    "projection_bounds": (-1.0, 1.0, -1.0, 1.0),
                    "lossiness": "lossless",
                    "no_mesh_fallback": False,
                }
            }
        },
    )
    body = make_surface_body((make_surface_shell((patch,), connected=False),))

    diagnostic = verify_displacement_csg_impress_round_trip(body)

    assert diagnostic.supported is False
    assert diagnostic.code == "displacement-csg-impress-roundtrip-failed"
    assert "no_mesh_fallback=true" in diagnostic.message
    with pytest.raises(ImpressFormatError, match="no_mesh_fallback=true"):
        displacement_csg_impress_payload_record(body)


def _promoted_sampled_implicit_body(target_family: str):
    row = next(
        row
        for row in build_sampled_implicit_promotion_matrix(operations=("union",)).rows
        if row.target_family == target_family
    )
    provenance = build_sampled_implicit_promotion_provenance_record(row, operand_ids=("left-source", "right-source"))
    metadata = {"kernel": {"sampled_implicit_promotion": provenance.canonical_payload()}}
    if target_family == "implicit":
        patch = ImplicitSurfacePatch(
            family="implicit",
            field=make_implicit_field_node("sphere"),
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        )
    elif target_family == "subdivision":
        patch = SubdivisionSurfacePatch(family="subdivision")
    elif target_family == "nurbs":
        patch = NURBSSurfacePatch(family="nurbs")
    elif target_family == "bspline":
        patch = BSplineSurfacePatch(family="bspline")
    else:
        raise AssertionError(f"Unsupported promotion test target {target_family!r}.")
    return make_surface_body((make_surface_shell((patch,), connected=False),), metadata=metadata)


@pytest.mark.parametrize("target_family", ["implicit", "subdivision", "bspline", "nurbs"])
def test_sampled_implicit_promotion_impress_payload_round_trips_target_family_metadata(target_family: str) -> None:
    body = _promoted_sampled_implicit_body(target_family)

    payload = encode_sampled_implicit_promotion_impress_payload(body)
    loaded = decode_impress_document_payload(payload)
    diagnostic = verify_sampled_implicit_promotion_impress_round_trip(body)
    record = sampled_implicit_promotion_impress_payload_record(body)

    assert len(loaded.bodies) == 1
    assert isinstance(record, SampledImplicitPromotionImpressPayloadRecord)
    assert record.operation == "union"
    assert record.target_family == target_family
    assert record.route_status == "promotion-route"
    assert record.source_operand_ids == ("left-source", "right-source")
    assert record.no_mesh_fallback is True
    assert isinstance(diagnostic, SampledImplicitPromotionImpressRoundTripDiagnostic)
    assert diagnostic.supported is True
    assert diagnostic.before == diagnostic.after
    assert payload["metadata"]["surface_csg_payload"]["kind"] == "sampled-implicit-promotion"


def test_sampled_implicit_promotion_impress_payload_refuses_malformed_metadata() -> None:
    body = _promoted_sampled_implicit_body("implicit")
    metadata = dict(body.metadata)
    kernel = dict(metadata["kernel"])
    promotion = dict(kernel["sampled_implicit_promotion"])
    lossiness = dict(promotion["lossiness"])
    lossiness["no_mesh_fallback"] = False
    promotion["lossiness"] = lossiness
    kernel["sampled_implicit_promotion"] = promotion
    malformed = make_surface_body(body.shells, metadata={"kernel": kernel})

    diagnostic = verify_sampled_implicit_promotion_impress_round_trip(malformed)

    assert diagnostic.supported is False
    assert diagnostic.code == "sampled-implicit-promotion-impress-roundtrip-failed"
    assert "no_mesh_fallback=true" in diagnostic.message
    with pytest.raises(ImpressFormatError, match="no_mesh_fallback=true"):
        sampled_implicit_promotion_impress_payload_record(malformed)


def test_decode_implicit_surface_patch_payload_reports_path_specific_unknown_node() -> None:
    patch = ImplicitSurfacePatch(family="implicit")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    geometry["field"] = {
        "kind": "union",
        "parameters": {},
        "children": [{"kind": "__import__", "parameters": {}, "children": []}],
    }
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=r"field\.children\[0\].*allowed nodes"):
        decode_surface_patch_payload(payload)


def test_decode_implicit_surface_patch_payload_refuses_over_budget_tree_with_path() -> None:
    patch = ImplicitSurfacePatch(family="implicit")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    geometry["field"] = {
        "kind": "union",
        "parameters": {},
        "children": [
            {"kind": "sphere", "parameters": {"radius": 1.0}, "children": []}
            for _index in range(9)
        ],
    }
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=r"field\.children.*max_children_per_node"):
        decode_surface_patch_payload(payload)


def test_decode_implicit_surface_patch_payload_refuses_executable_parameter_with_path() -> None:
    patch = ImplicitSurfacePatch(family="implicit")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    geometry["field"] = {
        "kind": "sphere",
        "parameters": {"label": "__import__('os').system('boom')"},
        "children": [],
    }
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=r"field\.parameters\.label.*executable payloads"):
        decode_surface_patch_payload(payload)


def test_encode_decode_heightmap_surface_patch_payload_round_trips_sampled_grid() -> None:
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.array([[True, False], [True, True]], dtype=bool),
        alpha_mode="mask",
        xy_scale=(0.5, 0.25),
        center=(1.0, 2.0, 3.0),
        height_scale=2.0,
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["kind"] == "HeightmapSurfacePatch"
    assert payload["family"] == "heightmap"
    assert payload["geometry"] == {
        "payload_version": 1,
        "height_samples": patch.height_samples.tolist(),
        "alpha_mask": patch.alpha_mask.tolist(),
        "alpha_mode": "mask",
        "xy_scale": [0.5, 0.25],
        "center": [1.0, 2.0, 3.0],
        "height_scale": 2.0,
    }
    assert decoded.stable_identity == patch.stable_identity
    assert isinstance(decoded, HeightmapSurfacePatch)
    assert decoded.alpha_mode == "mask"
    assert np.array_equal(decoded.alpha_mask, patch.alpha_mask)


def test_encode_decode_displacement_surface_patch_payload_round_trips_source_and_samples() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"kernel": {"source": "base-face"}})
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.array([[True, False], [True, True]], dtype=bool),
        alpha_mode="ignore",
        height_scale=2.0,
        direction=(0.0, 0.0, 1.0),
        projection="planar",
        plane="xy",
        projection_bounds=(-1.0, 1.0, -2.0, 2.0),
    )

    payload = encode_surface_patch_payload(patch)
    decoded = decode_surface_patch_payload(payload)

    assert payload["kind"] == "DisplacementSurfacePatch"
    assert payload["family"] == "displacement"
    assert payload["geometry"]["payload_version"] == 1
    assert payload["geometry"]["source_patch"]["kind"] == "PlanarSurfacePatch"
    assert payload["geometry"]["displacement_samples"] == patch.displacement_samples.tolist()
    assert payload["geometry"]["alpha_mask"] == patch.alpha_mask.tolist()
    assert payload["geometry"]["direction"] == [0.0, 0.0, 1.0]
    assert payload["geometry"]["projection_bounds"] == [-1.0, 1.0, -2.0, 2.0]
    assert decoded.stable_identity == patch.stable_identity
    assert isinstance(decoded, DisplacementSurfacePatch)
    assert decoded.source_patch.stable_identity == source.stable_identity


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


@pytest.mark.parametrize("mutation, message", [
    (lambda geometry: geometry.pop("payload_version"), "payload_version"),
    (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
    (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported PlanarSurfacePatch geometry fields"),
])
def test_decode_analytic_surface_patch_payload_rejects_unversioned_or_unknown_geometry(
    mutation,
    message: str,
) -> None:
    patch = PlanarSurfacePatch(family="planar")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported BSplineSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"degree_u": 0}), "positive integer"),
        (lambda geometry: geometry.update({"control_net": [[[0.0, 0.0, float("nan")]]]}), "control_net"),
    ],
)
def test_decode_bspline_surface_patch_payload_rejects_invalid_geometry(mutation, message: str) -> None:
    patch = BSplineSurfacePatch(family="bspline")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


def test_decode_nurbs_surface_patch_payload_rejects_invalid_weights() -> None:
    patch = NURBSSurfacePatch(family="nurbs")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    geometry["weights"] = [[1.0, 0.0], [1.0, 1.0]]
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match="weights"):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported SweepSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"profile_points_uv": [[0.0, 0.0]]}), "profile_points_uv"),
        (lambda geometry: geometry.update({"path_points": [[0.0, 0.0, 0.0]]}), "path_points"),
        (lambda geometry: geometry.update({"frame_policy": "magic"}), "frame_policy"),
        (lambda geometry: geometry.update({"profile_reference": ""}), "profile_reference"),
    ],
)
def test_decode_sweep_surface_patch_payload_rejects_invalid_geometry(mutation, message: str) -> None:
    patch = SweepSurfacePatch(family="sweep")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported SubdivisionSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"control_points": [[0.0, 0.0, 0.0]]}), "control_points"),
        (lambda geometry: geometry.update({"faces": [[0, 1]]}), "at least three vertices"),
        (lambda geometry: geometry.update({"faces": [[0, 1, True]]}), "indices must be integers"),
        (lambda geometry: geometry.update({"creases": [{"edge": [0, 2], "sharpness": 1.0}]}), "existing cage edges"),
        (lambda geometry: geometry.update({"creases": [{"edge": [0, 1], "sharpness": -1.0}]}), "sharpness"),
        (lambda geometry: geometry.update({"subdivision_level": -1}), "non-negative integer"),
        (lambda geometry: geometry.update({"scheme": "loop"}), "scheme"),
    ],
)
def test_decode_subdivision_surface_patch_payload_rejects_invalid_geometry(mutation, message: str) -> None:
    patch = SubdivisionSurfacePatch(family="subdivision")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported ImplicitSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"bounds": [0.0, 0.0, -1.0, 1.0, -1.0, 1.0]}), "positive span"),
        (lambda geometry: geometry.update({"field": {"kind": "python_eval", "parameters": {}, "children": []}}), "Unsupported implicit field node kind"),
        (lambda geometry: geometry.update({"field": {"kind": "sphere", "parameters": {"eval": "boom"}, "children": []}}), "Unsafe implicit field payload"),
        (
            lambda geometry: geometry.update(
                {
                    "field": {
                        "kind": "sphere",
                        "parameters": {"label": "__import__('os')"},
                        "children": [],
                    }
                }
            ),
            "Unsafe implicit field payload",
        ),
        (lambda geometry: geometry.update({"field": {"kind": "sphere", "code": "x"}}), "Unsupported implicit field node fields"),
    ],
)
def test_decode_implicit_surface_patch_payload_rejects_unsafe_or_invalid_geometry(mutation, message: str) -> None:
    patch = ImplicitSurfacePatch(family="implicit")
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


def test_impress_unsafe_payload_negative_fixtures_feed_diagnostic_matrix() -> None:
    planar_payload = encode_surface_patch_payload(PlanarSurfacePatch(family="planar"))
    unsupported_family_payload = dict(planar_payload)
    unsupported_family_payload["kind"] = "UnknownSurfacePatch"
    unsupported_family_payload["family"] = "unknown"
    unsafe_implicit_payload = encode_surface_patch_payload(ImplicitSurfacePatch(family="implicit"))
    unsafe_implicit_geometry = dict(unsafe_implicit_payload["geometry"])  # type: ignore[arg-type]
    unsafe_implicit_geometry["field"] = {
        "kind": "sphere",
        "parameters": {"eval": "boom"},
        "children": [],
    }
    unsafe_implicit_payload["geometry"] = unsafe_implicit_geometry
    wrapper_patch = PlanarSurfacePatch(
        family="planar",
        metadata={"kernel": {"producer": "heightmap", "triangle_face_index": 0}},
    )

    fixtures = (
        _impress_negative_fixture("impress/malformed-json", lambda: loads_impress_json("{")),
        _impress_negative_fixture(
            "impress/unsupported-document-field",
            lambda: decode_impress_document_payload(
                {
                    "format": IMPRESS_FORMAT,
                    "schema_version": CURRENT_IMPRESS_SCHEMA_VERSION,
                    "units": {"length": DEFAULT_IMPRESS_LENGTH_UNIT},
                    "metadata": {},
                    "body_store": {"bodies": []},
                    "bodies": {},
                    "patches": {},
                    "mesh": {},
                }
            ),
        ),
        _impress_negative_fixture(
            "impress/unsupported-family",
            lambda: decode_surface_patch_payload(unsupported_family_payload),
        ),
        _impress_negative_fixture(
            "impress/unsafe-implicit-field",
            lambda: decode_surface_patch_payload(unsafe_implicit_payload),
        ),
        _impress_negative_fixture(
            "impress/mesh-wrapper-serialization",
            lambda: encode_surface_patch_payload(wrapper_patch),
        ),
    )

    report = evaluate_negative_diagnostic_fixture_matrix(fixtures, required_domains=(".impress",))

    assert report.passed is True
    assert report.domain_coverage[0].fixture_count == 5
    assert all(fixture.expected_snapshot is not None for fixture in fixtures)
    assert any(
        fixture.fixture_id == "impress/unsafe-implicit-field"
        and "Unsafe implicit field payload" in fixture.expected_snapshot.payload["message"]  # type: ignore[index, union-attr]
        for fixture in fixtures
    )
    assert any(
        fixture.fixture_id == "impress/mesh-wrapper-serialization"
        and "mesh-derived surface wrapper" in fixture.expected_snapshot.payload["message"]  # type: ignore[index, union-attr]
        for fixture in fixtures
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported HeightmapSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"height_samples": [[0.0]]}), "height_samples"),
        (lambda geometry: geometry.update({"height_samples": [[0.0, float("nan")], [1.0, 2.0]]}), "height_samples"),
        (lambda geometry: geometry.update({"alpha_mask": [[True, "yes"], [False, True]]}), "alpha_mask"),
        (lambda geometry: geometry.update({"xy_scale": [0.0, 1.0]}), "xy_scale"),
        (lambda geometry: geometry.update({"alpha_mode": "magic"}), "alpha_mode"),
    ],
)
def test_decode_heightmap_surface_patch_payload_rejects_invalid_geometry(mutation, message: str) -> None:
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.array([[True, False], [True, True]], dtype=bool),
    )
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
        decode_surface_patch_payload(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda geometry: geometry.pop("payload_version"), "payload_version"),
        (lambda geometry: geometry.update({"payload_version": 2}), "payload_version"),
        (lambda geometry: geometry.update({"mesh": {"vertices": []}}), "Unsupported DisplacementSurfacePatch geometry fields"),
        (lambda geometry: geometry.update({"source_patch": {"kind": "UnknownSurfacePatch"}}), "SurfacePatch payload requires"),
        (lambda geometry: geometry.update({"displacement_samples": [[0.0]]}), "displacement_samples"),
        (lambda geometry: geometry.update({"alpha_mask": [[True, "yes"], [False, True]]}), "alpha_mask"),
        (lambda geometry: geometry.update({"direction": [0.0, 0.0, 0.0]}), "direction vector must be non-zero"),
        (lambda geometry: geometry.update({"plane": "abc"}), "plane"),
        (lambda geometry: geometry.update({"projection_bounds": [0.0, 0.0, -1.0, 1.0]}), "projection_bounds"),
    ],
)
def test_decode_displacement_surface_patch_payload_rejects_invalid_geometry(mutation, message: str) -> None:
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=PlanarSurfacePatch(family="planar"),
        displacement_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.array([[True, False], [True, True]], dtype=bool),
        direction="z",
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    payload = encode_surface_patch_payload(patch)
    geometry = dict(payload["geometry"])  # type: ignore[arg-type]
    mutation(geometry)
    payload["geometry"] = geometry

    with pytest.raises(ImpressFormatError, match=message):
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
