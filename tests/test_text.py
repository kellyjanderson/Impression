from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from impression.modeling import Section, TessellationRequest, make_text, tessellate_surface_body, text_profiles
from tests.text_font_fixtures import require_glyph_capable_font


def _require_text_font() -> Path:
    return require_glyph_capable_font("SURFACEABOI")


def _section_bounds(sections: list[Section]) -> tuple[float, float, float, float]:
    points = np.vstack(
        [
            loop.points
            for section in sections
            for region in section.regions
            for loop in (region.outer, *region.holes)
        ]
    )
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def test_text_profiles_returns_sections_with_expected_alignment() -> None:
    font_path = _require_text_font()
    left = text_profiles("II", font_size=1.0, font_path=str(font_path), justify="left")
    center = text_profiles("II", font_size=1.0, font_path=str(font_path), justify="center")
    right = text_profiles("II", font_size=1.0, font_path=str(font_path), justify="right")

    assert left
    assert all(isinstance(section, Section) for section in center)

    left_bounds = _section_bounds(left)
    center_bounds = _section_bounds(center)
    right_bounds = _section_bounds(right)

    assert left_bounds[0] > 0.0
    assert np.isclose(center_bounds[0], -center_bounds[1], atol=5e-3)
    assert right_bounds[1] < 0.0
    assert np.isclose(left_bounds[1] - left_bounds[0], center_bounds[1] - center_bounds[0], atol=1e-9)


def test_text_profiles_multiline_line_height_changes_vertical_span() -> None:
    font_path = _require_text_font()
    compact = text_profiles("I\nI", font_size=1.0, font_path=str(font_path), line_height=1.0)
    loose = text_profiles("I\nI", font_size=1.0, font_path=str(font_path), line_height=2.0)

    compact_bounds = _section_bounds(compact)
    loose_bounds = _section_bounds(loose)

    compact_span = compact_bounds[3] - compact_bounds[2]
    loose_span = loose_bounds[3] - loose_bounds[2]

    assert loose_span > compact_span + 0.5


def test_make_text_mesh_respects_depth_and_direction_axis() -> None:
    font_path = _require_text_font()
    mesh = make_text("OO", depth=0.2, font_size=1.0, font_path=str(font_path))
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    assert np.isclose(zmin, 0.0, atol=1e-9)
    assert np.isclose(zmax - zmin, 0.2, atol=1e-9)
    assert xmax - xmin > ymax - ymin

    oriented = make_text(
        "OO",
        depth=0.2,
        center=(1.0, 2.0, 3.0),
        direction=(1.0, 0.0, 0.0),
        font_size=1.0,
        font_path=str(font_path),
    )
    xmin, xmax, ymin, ymax, zmin, zmax = oriented.bounds

    assert np.isclose(xmin, 1.0, atol=1e-9)
    assert np.isclose(xmax - xmin, 0.2, atol=1e-9)
    assert ymax > ymin
    assert zmax > zmin


def test_surface_text_tessellates_to_expected_depth_and_patch_count() -> None:
    font_path = _require_text_font()
    body = make_text(
        "OO",
        depth=0.2,
        font_size=1.0,
        font_path=str(font_path),
        backend="surface",
    )
    mesh = tessellate_surface_body(body, TessellationRequest(intent="preview")).mesh
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    assert body.patch_count > 100
    assert np.isclose(zmin, 0.0, atol=1e-9)
    assert np.isclose(zmax - zmin, 0.2, atol=1e-9)
    assert xmax > xmin
    assert ymax > ymin
    assert mesh.n_faces > 0


def test_text_profiles_use_distinct_glyph_geometry_for_distinct_letters() -> None:
    font_path = _require_text_font()
    a_sections = text_profiles("A", font_size=1.0, font_path=str(font_path))
    b_sections = text_profiles("B", font_size=1.0, font_path=str(font_path))

    def _signature(sections: list[Section]) -> tuple[tuple[tuple[float, float], ...], ...]:
        signature: list[tuple[tuple[float, float], ...]] = []
        for section in sections:
            for region in section.regions:
                signature.append(tuple(tuple(np.round(point, 6)) for point in region.outer.points))
                for hole in region.holes:
                    signature.append(tuple(tuple(np.round(point, 6)) for point in hole.points))
        return tuple(signature)

    assert _signature(a_sections) != _signature(b_sections)


def test_make_text_invalid_depth() -> None:
    font_path = _require_text_font()
    with pytest.raises(ValueError):
        make_text("A", depth=0.0, font_path=str(font_path))


def test_text_missing_font() -> None:
    with pytest.raises(FileNotFoundError):
        text_profiles("A", font_path="does-not-exist.ttf")
