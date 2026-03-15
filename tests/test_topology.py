from __future__ import annotations

import numpy as np

from impression.modeling.drawing2d import Path2D, PlanarShape2D, make_circle, make_rect
from impression.modeling.topology import (
    Loop,
    anchor_loop,
    as_section,
    classify_loops,
    ensure_winding,
    loops_resampled,
    regions_from_paths,
    signed_area,
    triangulate_loops,
)


def test_signed_area_and_winding():
    square = Path2D.from_points([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True).sample()
    assert signed_area(square) > 0
    cw = ensure_winding(square, clockwise=True)
    assert signed_area(cw) < 0


def test_section_from_shape_with_hole():
    outer = make_rect(size=(4.0, 3.0)).outer
    hole = make_circle(radius=0.5).outer
    shape = PlanarShape2D(outer=outer, holes=[hole])
    section = as_section(shape)
    assert len(section.regions) == 1
    assert len(section.regions[0].holes) == 1


def test_as_section_from_planarshape():
    outer = make_rect(size=(5.0, 2.0)).outer
    hole = make_circle(radius=0.4).outer
    shape = PlanarShape2D(outer=outer, holes=[hole])
    section = as_section(shape)
    assert len(section.regions) == 1
    assert len(section.regions[0].holes) == 1


def test_classify_loops_outer_and_holes():
    outer = Loop(Path2D.from_points([(0, 0), (4, 0), (4, 4), (0, 4)], closed=True).sample()).points
    hole = Loop(Path2D.from_points([(1, 1), (1, 2), (2, 2), (2, 1)], closed=True).sample()).points
    resolved_outer, resolved_holes = classify_loops([hole, outer], expected_holes=1)
    assert signed_area(resolved_outer) > 0
    assert len(resolved_holes) == 1
    assert signed_area(resolved_holes[0]) < 0


def test_regions_from_paths_groups_hole_under_outer():
    outer = Path2D.from_points([(0, 0), (6, 0), (6, 4), (0, 4)], closed=True)
    hole = Path2D.from_points([(2, 1), (2, 3), (4, 3), (4, 1)], closed=True)
    regions = regions_from_paths([outer, hole])
    assert len(regions) == 1
    assert len(regions[0].holes) == 1


def test_anchor_loop_prefers_smallest_angle_then_largest_radius():
    loop = Loop(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, 0.0],
                [0.0, -2.0],
            ],
            dtype=float,
        )
    ).points
    anchored = anchor_loop(loop)
    assert np.allclose(anchored[0], np.array([2.0, 0.0]))


def test_triangulate_loops_covers_all_boundary_edges_for_multi_hole_loops():
    outer = make_rect(size=(2.0, 1.4)).outer
    hole_a = make_rect(size=(0.3, 0.3), center=(-0.4, 0.0)).outer
    hole_b = make_rect(size=(0.3, 0.3), center=(0.4, 0.0)).outer
    shape = PlanarShape2D(outer=outer, holes=[hole_a, hole_b])

    loops = loops_resampled(
        as_section(shape),
        count=32,
        segments_per_circle=64,
        bezier_samples=32,
        enforce_winding=True,
    )
    _, faces = triangulate_loops(loops)

    edge_set: set[tuple[int, int]] = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((a, b) if a < b else (b, a))
        edge_set.add((b, c) if b < c else (c, b))
        edge_set.add((c, a) if c < a else (a, c))

    cursor = 0
    for loop in loops:
        n = int(loop.shape[0])
        for i in range(n):
            a = cursor + i
            b = cursor + ((i + 1) % n)
            key = (a, b) if a < b else (b, a)
            assert key in edge_set
        cursor += n
