from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from impression.modeling import (
    SurfaceBody,
    SurfaceConsumerCollection,
    TessellationRequest,
    displace_heightmap,
    heightmap,
    make_arrow,
    make_dimension,
    make_line,
    make_plane,
    make_text,
    tessellate_surface_body,
    text,
)

FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "NotoSansSymbols2-Regular.ttf"
)
TEXT_MODULE = importlib.import_module("impression.modeling.text")


def test_surface_plane_and_line_tessellate_non_empty() -> None:
    plane = make_plane(size=(2.0, 1.0), center=(0.0, 0.0, 0.0), backend="surface")
    line = make_line((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), thickness=0.1, backend="surface")
    assert isinstance(plane, SurfaceBody)
    assert isinstance(line, SurfaceBody)
    assert tessellate_surface_body(plane, TessellationRequest(intent="preview")).mesh.n_faces > 0
    assert tessellate_surface_body(line, TessellationRequest(intent="preview")).mesh.n_faces > 0


def test_surface_arrow_returns_surface_body() -> None:
    arrow = make_arrow((0.0, 0.0, 0.0), (1.0, 0.5, 0.0), backend="surface")
    assert isinstance(arrow, SurfaceBody)
    mesh = tessellate_surface_body(arrow, TessellationRequest(intent="preview")).mesh
    assert mesh.n_faces > 0


def test_surface_dimension_returns_consumer_collection() -> None:
    result = make_dimension(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        offset=0.2,
        text="2.00",
        font_path=str(FONT_PATH),
        backend="surface",
    )
    assert isinstance(result, SurfaceConsumerCollection)
    assert len(result.items) == 2
    assert all(item.body.patch_count > 0 for item in result.items)


def test_surface_text_returns_surface_body() -> None:
    body = make_text(
        "IMP",
        depth=0.1,
        font_size=0.4,
        font_path=str(FONT_PATH),
        backend="surface",
    )
    assert isinstance(body, SurfaceBody)
    mesh = tessellate_surface_body(body, TessellationRequest(intent="preview")).mesh
    assert body.patch_count > 0
    assert np.isclose(mesh.bounds[5] - mesh.bounds[4], 0.1, atol=1e-9)
    assert mesh.n_faces > 0


def test_text_alias_surface_backend_matches_make_text() -> None:
    body = text(
        "OK",
        depth=0.08,
        font_size=0.35,
        font_path=str(FONT_PATH),
        backend="surface",
    )
    direct = make_text(
        "OK",
        depth=0.08,
        font_size=0.35,
        font_path=str(FONT_PATH),
        backend="surface",
    )
    assert isinstance(body, SurfaceBody)
    assert body.patch_count == direct.patch_count
    mesh = tessellate_surface_body(body, TessellationRequest(intent="preview")).mesh
    direct_mesh = tessellate_surface_body(direct, TessellationRequest(intent="preview")).mesh
    assert np.allclose(mesh.bounds, direct_mesh.bounds, atol=1e-9)
    assert mesh.n_faces == direct_mesh.n_faces


def test_text_module_avoids_public_extrude_dependency() -> None:
    source = Path(TEXT_MODULE.__file__).read_text()
    assert "from .extrude import linear_extrude" not in source


def test_surface_heightmap_and_displacement_return_surface_bodies() -> None:
    terrain = heightmap(
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        height=0.5,
        alpha_mode="ignore",
        backend="surface",
    )
    assert isinstance(terrain, SurfaceBody)
    assert tessellate_surface_body(terrain, TessellationRequest(intent="preview")).mesh.n_faces > 0

    plane = make_plane(size=(1.0, 1.0), backend="surface")
    displaced = displace_heightmap(
        plane,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        height=0.25,
        plane="xy",
        direction="z",
        backend="surface",
    )
    assert isinstance(displaced, SurfaceBody)
    assert tessellate_surface_body(displaced, TessellationRequest(intent="preview")).mesh.n_faces > 0
