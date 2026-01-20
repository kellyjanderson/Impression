"""Modeling utilities: primitives, CSG helpers, and path abstractions."""

from __future__ import annotations

from .transform import rotate, translate
from .primitives import (
    make_box,
    make_cone,
    make_cylinder,
    make_nhedron,
    make_ngon,
    make_polyhedron,
    make_prism,
    make_sphere,
    make_torus,
)
from .csg import boolean_union, boolean_difference, boolean_intersection, union_meshes
from .paths import Path
from .group import MeshGroup, group

__all__ = [
    "make_box",
    "make_cylinder",
    "make_cone",
    "make_ngon",
    "make_nhedron",
    "make_polyhedron",
    "make_prism",
    "make_sphere",
    "make_torus",
    "make_text",
    "boolean_union",
    "boolean_difference",
    "boolean_intersection",
    "union_meshes",
    "Path",
    "MeshGroup",
    "group",
    "rotate",
    "translate",
]


def make_text(*args, **kwargs):
    """Lazily import text support to avoid build123d/OCCT overhead during preview."""

    from .text import make_text as _make_text

    return _make_text(*args, **kwargs)
