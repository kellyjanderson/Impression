from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from impression.modeling import Loft, make_box
from impression.mesh import section_mesh_with_plane
from impression.modeling.drafting import make_arrow
from impression.modeling.text import make_text
from impression.modeling.heightmap import heightmap
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_branching_manifold_profiles,
    build_cylinder_correspondence_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)
from tests.text_font_fixtures import require_glyph_capable_font
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
    surface_body_section_loops,
)
from tests.reference_images import (
    dirty_reference_path,
    dirty_reference_stl_path,
    planar_loop_bounds,
    render_planar_section_bitmap,
    render_planar_section_diff_image,
    render_surface_body_image,
    render_surface_body_triptych_image,
    write_surface_body_stl,
)

REFERENCE_ROOT = PROJECT_ROOT / "project" / "reference-images"
REFERENCE_STL_ROOT = PROJECT_ROOT / "project" / "reference-stl"


def _load_docs_example_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load docs example module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_sections(profiles: list[object]):
    from impression.modeling import as_section

    return [as_section(profile) for profile in profiles]


def _expected_section_loops(profile: object, station_origin: np.ndarray) -> list[np.ndarray]:
    from impression.modeling import as_section

    section = as_section(profile)
    offset = np.asarray(station_origin, dtype=float)[:2]
    loops: list[np.ndarray] = []
    for region in section.regions:
        loops.append(np.asarray(region.outer.points, dtype=float) + offset)
        for hole in region.holes:
            loops.append(np.asarray(hole.points, dtype=float) + offset)
    return loops


def _actual_section_loops(body, station_origin: np.ndarray) -> list[np.ndarray]:
    from impression.modeling import tessellate_surface_body

    mesh = tessellate_surface_body(body).mesh
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(station_origin[2])),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    return [polyline.points[:, :2] for polyline in result.polylines if polyline.closed]


def _write_loft_section_artifacts(
    *,
    fixture_name: str,
    profiles: list[object],
    path: np.ndarray,
    station_index: int,
) -> None:
    sections = _as_sections(profiles)
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(sections)),
        stations=path,
        topology=sections,
        cap_ends=True,
        split_merge_mode="resolve",
    )
    expected_loops = _expected_section_loops(profiles[station_index], path[station_index])
    actual_loops = _actual_section_loops(body, path[station_index])
    shared_bounds = planar_loop_bounds(expected_loops, actual_loops)
    render_planar_section_bitmap(
        expected_loops,
        dirty_reference_path(REFERENCE_ROOT, f"loft_sections/{fixture_name}_expected"),
        bounds=shared_bounds,
    )
    render_planar_section_bitmap(
        actual_loops,
        dirty_reference_path(REFERENCE_ROOT, f"loft_sections/{fixture_name}_actual"),
        bounds=shared_bounds,
    )
    render_planar_section_diff_image(
        expected_loops,
        actual_loops,
        dirty_reference_path(REFERENCE_ROOT, f"loft_sections/{fixture_name}_diff"),
        bounds=shared_bounds,
    )


def _write_surface_csg_slice_artifacts(
    *,
    slice_basename: str,
    expected_loops: list[np.ndarray],
    actual_loops: list[np.ndarray],
) -> None:
    shared_bounds = planar_loop_bounds(expected_loops, actual_loops)
    render_planar_section_bitmap(
        expected_loops,
        dirty_reference_path(REFERENCE_ROOT, f"surfacebody_sections/{slice_basename}_expected"),
        bounds=shared_bounds,
    )
    render_planar_section_bitmap(
        actual_loops,
        dirty_reference_path(REFERENCE_ROOT, f"surfacebody_sections/{slice_basename}_actual"),
        bounds=shared_bounds,
    )
    render_planar_section_diff_image(
        expected_loops,
        actual_loops,
        dirty_reference_path(REFERENCE_ROOT, f"surfacebody_sections/{slice_basename}_diff"),
        bounds=shared_bounds,
    )


def _write_surface_csg_artifacts() -> None:
    union_fixture = build_csg_union_box_post_fixture()
    union_base = union_fixture["left_operand"]
    union_body = union_fixture["result_body"]
    union_post = union_fixture["right_operand"]
    render_surface_body_triptych_image(
        union_base,
        union_body,
        union_post,
        dirty_reference_path(REFERENCE_ROOT, "surfacebody/csg_union_box_post"),
    )
    write_surface_body_stl(
        union_body,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/csg_union_box_post"),
    )
    _write_surface_csg_slice_artifacts(
        slice_basename="csg_union_box_post",
        expected_loops=union_fixture["expected_slice_loops"],
        actual_loops=surface_body_section_loops(union_body, float(union_fixture["slice_z"])),
    )

    difference_fixture = build_csg_difference_slot_fixture()
    difference_base = difference_fixture["left_operand"]
    difference_body = difference_fixture["result_body"]
    difference_cutter = difference_fixture["right_operand"]
    render_surface_body_triptych_image(
        difference_base,
        difference_body,
        difference_cutter,
        dirty_reference_path(REFERENCE_ROOT, "surfacebody/csg_difference_slot"),
    )
    write_surface_body_stl(
        difference_body,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/csg_difference_slot"),
    )
    _write_surface_csg_slice_artifacts(
        slice_basename="csg_difference_slot",
        expected_loops=difference_fixture["expected_slice_loops"],
        actual_loops=surface_body_section_loops(difference_body, float(difference_fixture["slice_z"])),
    )


def main() -> None:
    text_font_path = require_glyph_capable_font("SURFACE")
    box = make_box(size=(2.0, 3.0, 1.5), backend="surface")
    render_surface_body_image(box, dirty_reference_path(REFERENCE_ROOT, "surfacebody/box"))
    write_surface_body_stl(box, dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/box"))

    arrow = make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703", backend="surface")
    render_surface_body_image(arrow, dirty_reference_path(REFERENCE_ROOT, "surfacebody/drafting_arrow"))
    write_surface_body_stl(arrow, dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/drafting_arrow"))

    text_body = make_text(
        "SURFACE",
        depth=0.08,
        font_size=0.3,
        font_path=str(text_font_path),
        color="#5b84b1",
        backend="surface",
    )
    render_surface_body_image(text_body, dirty_reference_path(REFERENCE_ROOT, "surfacebody/text_surface"))
    write_surface_body_stl(text_body, dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/text_surface"))

    terrain = heightmap(
        np.asarray(
            [
                [0.0, 0.2, 0.6, 0.9],
                [0.1, 0.5, 0.8, 0.7],
                [0.2, 0.7, 1.0, 0.4],
                [0.0, 0.3, 0.5, 0.2],
            ],
            dtype=float,
        ),
        height=0.6,
        xy_scale=0.3,
        alpha_mode="ignore",
        backend="surface",
    )
    render_surface_body_image(terrain, dirty_reference_path(REFERENCE_ROOT, "surfacebody/heightmap_surface"))
    write_surface_body_stl(terrain, dirty_reference_stl_path(REFERENCE_STL_ROOT, "surfacebody/heightmap_surface"))

    _write_surface_csg_artifacts()

    profiles, path = build_branching_manifold_profiles()
    branching = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )
    render_surface_body_image(
        branching,
        dirty_reference_path(REFERENCE_ROOT, "loft/branching_manifold"),
    )
    write_surface_body_stl(
        branching,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "loft/branching_manifold"),
    )

    hourglass_module = _load_docs_example_module(
        "loft_hourglass_vessel_example",
        "docs/examples/loft/real_world/loft_hourglass_vessel_example.py",
    )
    hourglass_body = hourglass_module.build_surface_body(
        hourglass_module.TEST_PARAMETERS,
        hourglass_module.TEST_QUALITY,
    )
    render_surface_body_image(
        hourglass_body,
        dirty_reference_path(REFERENCE_ROOT, "loft/hourglass_vessel"),
    )
    write_surface_body_stl(
        hourglass_body,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "loft/hourglass_vessel"),
    )

    square_profiles, square_path = build_square_correspondence_profiles()
    square_body = Loft(
        progression=np.linspace(0.0, 1.0, len(square_profiles)),
        stations=square_path,
        topology=square_profiles,
        cap_ends=True,
    )
    render_surface_body_image(
        square_body,
        dirty_reference_path(REFERENCE_ROOT, "loft/square_correspondence"),
    )
    write_surface_body_stl(
        square_body,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "loft/square_correspondence"),
    )

    cylinder_profiles, cylinder_path = build_cylinder_correspondence_profiles()
    cylinder_body = Loft(
        progression=np.linspace(0.0, 1.0, len(cylinder_profiles)),
        stations=cylinder_path,
        topology=cylinder_profiles,
        cap_ends=True,
    )
    render_surface_body_image(
        cylinder_body,
        dirty_reference_path(REFERENCE_ROOT, "loft/cylinder_correspondence"),
    )
    write_surface_body_stl(
        cylinder_body,
        dirty_reference_stl_path(REFERENCE_STL_ROOT, "loft/cylinder_correspondence"),
    )

    section_fixtures = [
        ("square_station_1", build_square_correspondence_profiles, 1),
        ("cylinder_station_1", build_cylinder_correspondence_profiles, 1),
        ("anchor_shift_rectangle_station_2", build_anchor_shift_rectangle_profiles, 2),
        ("phase_shift_cylinder_station_2", build_phase_shift_cylinder_profiles, 2),
        ("dual_cylinder_station_1", build_dual_cylinder_correspondence_profiles, 1),
        ("perforated_cylinder_station_1", build_perforated_cylinder_correspondence_profiles, 1),
    ]
    for fixture_name, builder, station_index in section_fixtures:
        profiles, path = builder()
        _write_loft_section_artifacts(
            fixture_name=fixture_name,
            profiles=profiles,
            path=path,
            station_index=station_index,
        )

if __name__ == "__main__":
    main()
