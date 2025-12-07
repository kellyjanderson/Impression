"""Modeling utilities: primitives, CSG helpers, and path abstractions."""

from __future__ import annotations

from .transform import rotate, translate
from .primitives import (
    make_box,
    make_cone,
    make_cylinder,
    make_prism,
    make_sphere,
    make_torus,
)
from .text import make_text
from .csg import boolean_union, boolean_difference, boolean_intersection
from .paths import Path
from .group import MeshGroup, group

__all__ = [
    "make_box",
    "make_cylinder",
    "make_cone",
    "make_prism",
    "make_sphere",
    "make_torus",
    "make_text",
    "boolean_union",
    "boolean_difference",
    "boolean_intersection",
    "Path",
    "MeshGroup",
    "group",
    "rotate",
    "translate",
]
