"""Modeling utilities: primitives, CSG helpers, and path abstractions."""

from __future__ import annotations

from .transform import rotate, translate, scale, resize, mirror, multmatrix, rotate_euler
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
from .drawing2d import (
    Arc2D,
    Bezier2D,
    Line2D,
    Path2D,
    Profile2D,
    make_circle,
    make_ngon as make_ngon_2d,
    make_polygon,
    make_polyline,
    make_rect,
)
from .extrude import linear_extrude, rotate_extrude
from .loft import loft, loft_profiles
from .morph import morph, morph_profiles
from .path3d import Arc3D, Bezier3D, Line3D, Path3D
from .csg import boolean_union, boolean_difference, boolean_intersection, union_meshes
from .paths import Path
from .group import MeshGroup, group
from .ops import offset, hull, minkowski

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
    "Arc2D",
    "Bezier2D",
    "Line2D",
    "Path2D",
    "Profile2D",
    "make_rect",
    "make_circle",
    "make_ngon_2d",
    "make_polygon",
    "make_polyline",
    "Line3D",
    "Arc3D",
    "Bezier3D",
    "Path3D",
    "linear_extrude",
    "rotate_extrude",
    "morph",
    "morph_profiles",
    "loft",
    "loft_profiles",
    "boolean_union",
    "boolean_difference",
    "boolean_intersection",
    "union_meshes",
    "Path",
    "MeshGroup",
    "group",
    "rotate",
    "translate",
    "scale",
    "resize",
    "mirror",
    "multmatrix",
    "rotate_euler",
    "offset",
    "hull",
    "minkowski",
]
# build123d-backed text support temporarily disabled.
