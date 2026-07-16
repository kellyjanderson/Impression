from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from impression.io import decode_surface_patch_payload, encode_surface_patch_payload, load_impress, save_impress
from impression.devtools.reference_review import (
    ReferenceReviewStatus,
    ReviewSourceModelRecord,
    approve_reference_artifacts,
    load_source_records_from_file,
    update_fixture_review_status_in_file,
)
from impression.devtools.reference_review.ui.queue_context import FixtureQueueViewModel
from impression.modeling import boolean_difference, loft, make_box
from impression.modeling.drawing2d import make_circle
from impression.modeling.loft import RailSource, resolve_authored_rails
from impression.modeling.path3d import Path3D
from impression.modeling.surface import (
    BSplineSurfacePatch,
    DisplacementSurfacePatch,
    HeightmapSurfacePatch,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SubdivisionSurfacePatch,
    SurfaceBoundaryRef,
    SurfaceSeam,
    SweepSurfacePatch,
    TrimLoop,
    make_implicit_field_node,
    make_surface_body,
    make_surface_shell,
)
from impression.modeling.topology import TopologyPath
from tests.reference_review_fixtures.stl_review_sources import (
    build_patch_family_csg_reference_matrix,
    build_reference_csg_fixture_readiness_report,
    build_unsupported_family_refusal_fixture_matrix,
    collect_reference_csg_gap_records,
    probe_reference_csg_runtime_support,
)
from tests.reference_images import (
    CvArtifactBundle,
    CvArtifactBundleContract,
    classify_reference_fixture_pair_promotion_gate,
    clean_reference_path,
    clean_reference_stl_path,
    ensure_complete_cv_artifact_bundle,
    image_signal_stats,
    planar_loop_bounds,
    reference_artifact_state,
    render_planar_section_bitmap,
    render_planar_section_diff_image,
    write_reference_artifact_contract_version,
)


def _rail_triangle(*, correspond: tuple[str, str, str]) -> TopologyPath:
    return (
        TopologyPath.closed()
        .point("a", (0.0, 0.0), correspond=correspond[0])
        .point("b", (1.0, 0.0), correspond=correspond[1])
        .point("c", (0.0, 1.0), correspond=correspond[2])
        .build()
    )


def test_reference_expansion_authored_rails_are_distinct_from_generated_rails() -> None:
    explicit_source = _rail_triangle(correspond=("shared", "source-b", "source-c"))
    explicit_target = (
        TopologyPath.closed()
        .point("renamed", (0.0, 0.0), correspond="shared")
        .point("b", (1.0, 0.0), correspond="target-b")
        .point("c", (0.0, 1.0), correspond="target-c")
        .build()
    )
    generated_source = TopologyPath.named_rect(2.0, 1.0)
    generated_target = TopologyPath.named_rect(3.0, 1.5)

    explicit_result = resolve_authored_rails(explicit_source, explicit_target)
    generated_result = resolve_authored_rails(generated_source, generated_target)

    assert ("a", "renamed") in explicit_result.matches
    assert explicit_result.source_by_match[("a", "renamed")] == RailSource.EXPLICIT_ID
    assert ("bottom-left", "bottom-left") in generated_result.matches
    assert generated_result.source_by_match[("bottom-left", "bottom-left")] == RailSource.GENERATED_RAIL


def test_reference_expansion_loft_section_bundle_carries_expected_actual_and_diff(tmp_path: Path) -> None:
    expected_loops = [np.asarray([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], dtype=float)]
    actual_loops = [np.asarray([(-0.45, -0.5), (0.55, -0.45), (0.48, 0.55), (-0.52, 0.45)], dtype=float)]
    bounds = planar_loop_bounds(expected_loops, actual_loops)
    expected_path = tmp_path / "loft-section-expected.png"
    actual_path = tmp_path / "loft-section-actual.png"
    diff_path = tmp_path / "loft-section-diff.png"

    render_planar_section_bitmap(expected_loops, expected_path, bounds=bounds)
    render_planar_section_bitmap(actual_loops, actual_path, bounds=bounds)
    render_planar_section_diff_image(expected_loops, actual_loops, diff_path, bounds=bounds)
    bundle = ensure_complete_cv_artifact_bundle(
        CvArtifactBundle(
            contract=CvArtifactBundleContract(
                fixture_id="loft_sections/reference_expansion_contract",
                lane="loft-section-comparison",
                required_keys=("expected", "actual", "diff"),
                authoritative_keys=("expected", "actual", "diff"),
                review_keys=("expected", "actual", "diff"),
            ),
            stage="review",
            artifacts={"expected": expected_path, "actual": actual_path, "diff": diff_path},
        )
    )

    assert set(bundle.artifacts) == {"expected", "actual", "diff"}
    assert image_signal_stats(expected_path, background_threshold=5)["occupancy"] > 0.002
    assert image_signal_stats(actual_path, background_threshold=5)["occupancy"] > 0.002
    assert image_signal_stats(diff_path, background_threshold=5)["occupancy"] > 0.002


def test_reference_expansion_lofted_body_csg_refuses_without_fallback() -> None:
    body = loft(
        [make_circle(radius=0.4), make_circle(radius=0.52), make_circle(radius=0.34)],
        path=[(0.0, 0.0, 0.0), (0.08, 0.02, 0.75), (0.12, -0.02, 1.5)],
        cap_ends=True,
        samples=32,
    )
    cutter = make_box(size=(0.5, 0.5, 0.8), center=(0.0, 0.0, 0.75))

    result = boolean_difference(body, [cutter])

    assert result.status == "unsupported"
    assert result.body is None
    assert "Loft CSG eligibility" in str(result.failure_reason)
    assert "no mesh fallback" in str(result.failure_reason)


def test_reference_expansion_approval_moves_dirty_stl_to_gold_and_persists_status(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    dirty = tmp_path / "reference-stl" / "dirty" / "demo" / "fixture.stl"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("solid demo\nendsolid demo\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/approve",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "artifact_paths": [dirty.relative_to(tmp_path).as_posix()],
                    }
                ]
            }
        )
    )
    record = load_source_records_from_file(fixture_file).valid_items[0].record

    promotion = approve_reference_artifacts(record)
    update = update_fixture_review_status_in_file(
        fixture_file,
        fixture_id=record.fixture_id,
        status=ReferenceReviewStatus.APPROVED,
        artifact_paths=promotion.artifact_paths,
    )
    reloaded = load_source_records_from_file(fixture_file).valid_items[0].record

    assert promotion.updated
    assert update.updated
    assert not dirty.exists()
    assert reloaded.review_status is ReferenceReviewStatus.APPROVED
    assert reloaded.artifact_paths[0].parts[-3:] == ("gold", "demo", "fixture.stl")
    assert reloaded.artifact_paths[0].read_text() == "solid demo\nendsolid demo\n"


def test_reference_expansion_decline_leaves_dirty_stl_and_persists_status(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    dirty = tmp_path / "reference-stl" / "dirty" / "demo" / "fixture.stl"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("solid demo\nendsolid demo\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/decline",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "artifact_paths": [dirty.relative_to(tmp_path).as_posix()],
                    }
                ]
            }
        )
    )

    update = update_fixture_review_status_in_file(
        fixture_file,
        fixture_id="demo/decline",
        status=ReferenceReviewStatus.DECLINED,
    )
    reloaded = load_source_records_from_file(fixture_file).valid_items[0].record

    assert update.updated
    assert dirty.exists()
    assert reloaded.review_status is ReferenceReviewStatus.DECLINED
    assert reloaded.artifact_paths[0].parts[-3:] == ("dirty", "demo", "fixture.stl")


def test_reference_expansion_queue_prefers_unreviewed_fixture_when_approved_is_present(tmp_path: Path) -> None:
    sources = []
    for name in ("approved", "declined", "unreviewed"):
        source = tmp_path / f"{name}.py"
        source.write_text("def build():\n    return None\n")
        sources.append(source)
    records = (
        ReviewSourceModelRecord(
            fixture_id="demo/approved",
            feature_name="demo",
            source_path=sources[0],
            review_status=ReferenceReviewStatus.APPROVED,
        ),
        ReviewSourceModelRecord(
            fixture_id="demo/declined",
            feature_name="demo",
            source_path=sources[1],
            review_status=ReferenceReviewStatus.DECLINED,
        ),
        ReviewSourceModelRecord(
            fixture_id="demo/unreviewed",
            feature_name="demo",
            source_path=sources[2],
            review_status=ReferenceReviewStatus.UNREVIEWED,
        ),
    )

    queue = FixtureQueueViewModel(records)

    assert queue.selected_record is records[2]
    assert [item.status for item in queue.items] == ["approved", "declined", "unreviewed"]
    assert [item.fixture_id for item in queue.items if item.status != "approved"] == [
        "demo/declined",
        "demo/unreviewed",
    ]


def test_reference_csg_gap_audit_maps_progression_to_fixture_readiness(tmp_path: Path) -> None:
    plan = tmp_path / "reference-test-expansion-plan.md"
    plan.write_text(
        "\n".join(
            [
                "- [x] `RT-CSG-001` cube union sphere",
                "- [ ] `RT-CSG-009` coincident-face box union and difference",
            ]
        )
    )
    artifact = tmp_path / "reference-stl" / "dirty" / "surfacebody" / "csg" / "rt_csg_001.stl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("solid rt_csg_001\nendsolid rt_csg_001\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "surfacebody/csg/rt_csg_001_cube_union_sphere",
                        "source_path": "stl_review_sources.py",
                        "entrypoint": "build_surfacebody_csg_rt_csg_001_cube_union_sphere",
                        "purpose": "Validate RT-CSG-001: primitive analytic cube union sphere.",
                        "artifact_paths": [artifact.relative_to(tmp_path).as_posix()],
                    }
                ]
            }
        )
    )

    records = collect_reference_csg_gap_records(plan_path=plan, fixture_file=fixture_file)
    support = probe_reference_csg_runtime_support(records)
    readiness = build_reference_csg_fixture_readiness_report(plan_path=plan, fixture_file=fixture_file)

    assert [record.reference_id for record in records] == ["RT-CSG-001", "RT-CSG-009"]
    assert records[0].fixture_id == "surfacebody/csg/rt_csg_001_cube_union_sphere"
    assert records[0].source_entrypoint == "build_surfacebody_csg_rt_csg_001_cube_union_sphere"
    assert records[1].fixture_id is None
    assert {result.reference_id: result.supported for result in support} == {
        "RT-CSG-001": True,
        "RT-CSG-009": False,
    }
    assert readiness[0].ready_for_fixture
    assert readiness[0].artifact_paths == (artifact.resolve(),)
    assert readiness[1].unsupported_implementation_gap
    assert readiness[1].diagnostics == ("progression-unchecked", "missing-fixture-record")


def test_patch_family_csg_reference_matrix_formats_artifact_policy() -> None:
    rows = build_patch_family_csg_reference_matrix(families=("planar", "implicit"))

    assert rows == tuple(sorted(rows, key=lambda row: (row.operation, row.left_family, row.right_family)))
    assert {row.artifact_policy for row in rows} == {"dirty-stl", "diagnostic-refusal"}
    assert all(row.no_hidden_mesh_fallback_required for row in rows)
    assert all(row.fixture_ready is (row.artifact_policy == "dirty-stl") for row in rows)
    assert all(row.diagnostic for row in rows if row.artifact_policy == "diagnostic-refusal")


def test_unsupported_family_refusal_fixtures_are_diagnostic_only() -> None:
    rows = build_unsupported_family_refusal_fixture_matrix()

    assert {row.route_kind for row in rows} == {"refusal", "unsafe", "malformed"}
    assert all(row.fixture_id.startswith("surfacebody/csg/rt_patch_csg_013_") for row in rows)
    assert {row.artifact_policy for row in rows} == {"diagnostic-refusal"}
    assert {row.expected_output for row in rows} == {"diagnostic evidence"}
    assert all(row.fixture_ready for row in rows)
    assert all(row.no_mesh_fallback for row in rows)
    assert all(row.diagnostic for row in rows)
    assert all("stl" not in row.expected_output.lower() for row in rows)


def test_unsupported_family_refusal_review_records_load_without_artifacts() -> None:
    summary = load_source_records_from_file(Path("tests/reference_review_fixtures/dirty-stl-fixtures.json"))
    records = {
        item.record.fixture_id: item.record
        for item in summary.valid_items
        if item.record.fixture_id.startswith("surfacebody/csg/rt_patch_csg_013_")
    }

    assert set(records) == {row.fixture_id for row in build_unsupported_family_refusal_fixture_matrix()}
    assert {record.expected_output for record in records.values()} == {"diagnostic evidence"}
    assert all(record.artifact_paths == () for record in records.values())


def test_loft_self_intersection_diagnostic_review_record_loads_without_artifact() -> None:
    summary = load_source_records_from_file(Path("tests/reference_review_fixtures/dirty-stl-fixtures.json"))
    record = next(
        item.record
        for item in summary.valid_items
        if item.record.fixture_id == "loft/rt_loft_037_self_intersection_diagnostic"
    )

    assert record.expected_output == "diagnostic evidence"
    assert record.entrypoint == "build_loft_rt_loft_037_self_intersection_reference"
    assert record.artifact_paths == ()
    assert "RT-LOFT-037" in (record.purpose or "")


def test_underconstrained_branching_loft_csg_review_record_loads_without_artifact() -> None:
    summary = load_source_records_from_file(Path("tests/reference_review_fixtures/dirty-stl-fixtures.json"))
    record = next(
        item.record
        for item in summary.valid_items
        if item.record.fixture_id == "loft/csg/rt_loft_csg_014_underconstrained_branch_refusal"
    )

    assert record.expected_output == "diagnostic evidence"
    assert record.entrypoint == "build_loft_rt_loft_csg_014_underconstrained_branch_reference"
    assert record.artifact_paths == ()
    assert "RT-LOFT-CSG-014" in (record.purpose or "")


def test_reference_expansion_fixture_generator_refuses_stale_contract_versions(tmp_path: Path) -> None:
    reference_image_root = tmp_path / "reference-images"
    reference_stl_root = tmp_path / "reference-stl"
    fixture_id = "reference_expansion/stale_contract"
    clean_image = clean_reference_path(reference_image_root, fixture_id)
    clean_stl = clean_reference_stl_path(reference_stl_root, fixture_id)
    clean_image.parent.mkdir(parents=True)
    clean_stl.parent.mkdir(parents=True)
    clean_image.write_bytes(b"stale clean image placeholder")
    clean_stl.write_text("solid stale\nendsolid stale\n", encoding="utf-8")
    stale_contract = classify_reference_fixture_pair_promotion_gate(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=fixture_id,
        contract_version="v1",
    ).contract
    write_reference_artifact_contract_version(
        reference_artifact_state(reference_image_root, fixture_id, kind="image"),
        stale_contract,
    )
    write_reference_artifact_contract_version(
        reference_artifact_state(reference_stl_root, fixture_id, kind="stl"),
        stale_contract,
    )

    report = classify_reference_fixture_pair_promotion_gate(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=fixture_id,
        contract_version="v2",
    )

    assert report.promoted is False
    assert {diagnostic.code for diagnostic in report.diagnostics} == {"invalidated-contract"}
    assert {diagnostic.artifact_kind for diagnostic in report.diagnostics} == {"image", "stl"}


def test_reference_expansion_impress_round_trip_preserves_patch_family_payloads(tmp_path: Path) -> None:
    control_net = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 1.0, 0.4], [1.0, 2.0, 0.1]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.2], [2.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    patches = (
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
        SubdivisionSurfacePatch(
            family="subdivision",
            control_points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.1), (0.0, 1.0, 0.0)],
            faces=((0, 1, 2, 3),),
            subdivision_level=1,
        ),
        ImplicitSurfacePatch(
            family="implicit",
            field=make_implicit_field_node("sphere", parameters={"center": (0.0, 0.0, 0.0), "radius": 0.75}),
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
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
    )
    seam = SurfaceSeam(
        "planar-ruled",
        (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
        metadata={"kernel": {"continuity_source": "reference-expansion"}},
    )
    body = make_surface_body([make_surface_shell(patches, seams=(seam,), metadata={"fixture": "patch-family-round-trip"})])
    path = tmp_path / "patch-family-round-trip.impress"
    before_payloads = tuple(encode_surface_patch_payload(patch) for patch in patches)

    save_impress([body], path)
    loaded = load_impress(path)
    loaded_patches = tuple(loaded.bodies[0].iter_patches(world=True))
    after_payloads = tuple(encode_surface_patch_payload(patch) for patch in loaded_patches)

    assert path.exists()
    assert {payload["family"] for payload in after_payloads} == {payload["family"] for payload in before_payloads}
    assert after_payloads == before_payloads
    assert all(decode_surface_patch_payload(payload).stable_identity for payload in after_payloads)
