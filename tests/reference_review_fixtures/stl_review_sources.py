"""Loadable source entrypoints for dirty STL reference review fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from impression.modeling import (
    Loft,
    handoff_hinge_surface,
    make_bistable_hinge,
    make_box,
    make_living_hinge,
    make_traditional_hinge_pair,
)
from impression.modeling.drafting import make_arrow
from impression.modeling.heightmap import heightmap
from impression.modeling.text import make_text
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
)
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_branching_manifold_profiles,
    build_cylinder_correspondence_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)
from tests.text_font_fixtures import require_glyph_capable_font

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _loft_from_profiles(builder):
    profiles, path = builder()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )


def build_loft_anchor_shift_rectangle():
    return _loft_from_profiles(build_anchor_shift_rectangle_profiles)


def build_loft_branching_manifold():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )


def build_loft_cylinder_correspondence():
    return _loft_from_profiles(build_cylinder_correspondence_profiles)


def build_loft_hourglass_vessel():
    module_path = _PROJECT_ROOT / "docs/examples/loft/real_world/loft_hourglass_vessel_example.py"
    spec = importlib.util.spec_from_file_location("loft_hourglass_vessel_example", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_surface_body(module.TEST_PARAMETERS, module.TEST_QUALITY)


def build_loft_phase_shift_cylinder():
    return _loft_from_profiles(build_phase_shift_cylinder_profiles)


def build_loft_square_correspondence():
    return _loft_from_profiles(build_square_correspondence_profiles)


def build_surfacebody_box():
    return make_box(size=(2.0, 3.0, 1.5), center=(0.0, 0.0, 0.0), backend="surface")


def build_surfacebody_csg_difference_slot():
    return build_csg_difference_slot_fixture()["result_body"]


def build_surfacebody_csg_union_box_post():
    return build_csg_union_box_post_fixture()["result_body"]


def build_surfacebody_drafting_arrow():
    return make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703", backend="surface")


def build_surfacebody_heightmap_surface():
    return heightmap(
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


def build_surfacebody_hinge_bistable_blank():
    return handoff_hinge_surface(make_bistable_hinge(width=40.0, preload_offset=2.0, backend="surface"))


def build_surfacebody_hinge_living_panel():
    return handoff_hinge_surface(
        make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8, backend="surface")
    )


def build_surfacebody_hinge_traditional_pair():
    return handoff_hinge_surface(
        make_traditional_hinge_pair(width=24.0, knuckle_count=5, opened_angle_deg=32.0, backend="surface")
    )


def build_surfacebody_text_surface():
    return make_text(
        "SURFACE",
        depth=0.08,
        font_size=0.3,
        font_path=str(require_glyph_capable_font()),
        color="#5b84b1",
        backend="surface",
    )
