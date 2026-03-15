from __future__ import annotations

from typing import Iterable

import numpy as np

from impression.modeling.drawing2d import Path2D
from impression.modeling.topology import (
    Loop,
    Region,
    Section,
    ensure_winding,
    largest_loop,
    point_in_polygon,
    as_section,
    signed_area,
)


def offset_planar(
    shape: Section | Region | Path2D | object,
    r: float | None = None,
    delta: float | None = None,
    chamfer: bool = False,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[Section]:
    """Offset a planar shape and return topology-native Sections."""

    if r is None and delta is None:
        raise ValueError("offset requires r or delta.")
    if r is not None and delta is not None:
        raise ValueError("Provide only one of r or delta.")
    dist = float(r if r is not None else delta)

    CrossSection, JoinType = load_cross_section()
    sections = _shape_to_sections(shape, segments_per_circle, bezier_samples)
    join_type = JoinType.Miter if chamfer else JoinType.Round
    out: list[Section] = []
    for section in sections:
        contours = _section_to_contours(section)
        if not contours:
            continue
        result = CrossSection(contours).offset(dist, join_type=join_type)
        out.extend(cross_section_to_sections(result))
    return out


def hull_planar(shapes: Iterable[Section | Region | Path2D | object]) -> list[Section]:
    """Compute convex hull for planar inputs and return topology-native Sections."""

    items = list(shapes)
    if not items:
        raise ValueError("hull requires at least one shape.")

    CrossSection, _ = load_cross_section()
    cross_sections = []
    for item in items:
        sections = _shape_to_sections(item, 64, 32)
        for section in sections:
            contours = _section_to_contours(section)
            if contours:
                cross_sections.append(CrossSection(contours))

    if not cross_sections:
        return []
    result = CrossSection.batch_hull(cross_sections)
    return cross_section_to_sections(result)


def is_planar_shape(value: object) -> bool:
    if isinstance(value, (Section, Region, Path2D)):
        return True
    return hasattr(value, "outer") and hasattr(value, "holes")


def _shape_to_sections(
    shape: Section | Region | Path2D | object,
    segments_per_circle: int,
    bezier_samples: int,
) -> list[Section]:
    section = as_section(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )
    if not section.regions:
        return []
    return [section]


def _section_to_contours(section: Section) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    normalized = section.normalized()
    for region in normalized.regions:
        contours.append(np.asarray(region.outer.points, dtype=float))
        for hole in region.holes:
            contours.append(np.asarray(hole.points, dtype=float))
    return contours


def cross_section_to_sections(cross_section) -> list[Section]:
    sections: list[Section] = []
    for part in cross_section.decompose():
        contours = [np.asarray(poly, dtype=float) for poly in part.to_polygons()]
        if not contours:
            continue

        outers = [c for c in contours if signed_area(c) >= 0.0]
        holes = [c for c in contours if signed_area(c) < 0.0]
        if not outers:
            outer = largest_loop(contours)
            outers = [outer]
            holes = [c for c in contours if c is not outer]

        outer_records = [{"pts": outer, "area": abs(signed_area(outer)), "holes": []} for outer in outers]
        for hole in holes:
            p = np.asarray(hole, dtype=float)
            if p.size == 0:
                continue
            probe = p.mean(axis=0)
            containers = [record for record in outer_records if point_in_polygon(probe, record["pts"])]
            if not containers:
                continue
            owner = min(containers, key=lambda rec: rec["area"])
            owner["holes"].append(hole)

        regions: list[Region] = []
        for record in outer_records:
            outer_loop = Loop(ensure_winding(np.asarray(record["pts"], dtype=float), clockwise=False))
            hole_loops = tuple(
                Loop(ensure_winding(np.asarray(hole, dtype=float), clockwise=True))
                for hole in record["holes"]
            )
            regions.append(Region(outer=outer_loop, holes=hole_loops))

        if regions:
            sections.append(Section(tuple(regions)).normalized())
    return sections


def load_cross_section():
    try:
        from manifold3d import CrossSection, JoinType
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("manifold3d is required for 2D offset/hull.") from exc
    return CrossSection, JoinType
