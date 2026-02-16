from __future__ import annotations

from typing import Sequence

import numpy as np

from impression.mesh import Mesh, combine_meshes

from ._color import set_mesh_color
from .csg import boolean_difference, boolean_union
from .group import MeshGroup, group
from .primitives import make_box, make_cylinder
from .transform import rotate


def _safe_union(parts: Sequence[Mesh]) -> Mesh:
    if not parts:
        raise ValueError("Expected at least one mesh to union.")
    if len(parts) == 1:
        return parts[0].copy()
    try:
        return boolean_union(parts)
    except Exception:
        return combine_meshes(parts)


def _safe_difference(base: Mesh, cutters: Sequence[Mesh]) -> Mesh:
    if not cutters:
        return base.copy()
    try:
        return boolean_difference(base, cutters)
    except Exception:
        return base.copy()


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def _make_strut(
    p0: Sequence[float],
    p1: Sequence[float],
    width: float,
    thickness: float,
) -> Mesh:
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    delta = p1 - p0
    length = float(np.linalg.norm(delta[:2]))
    if length <= 1e-9:
        raise ValueError("Strut endpoints are coincident.")
    center = (p0 + p1) / 2.0
    angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))
    strut = make_box(size=(length, width, thickness), center=center)
    rotate(strut, axis=(0.0, 0.0, 1.0), angle_deg=angle_deg, origin=center)
    return strut


def make_traditional_hinge_leaf(
    *,
    width: float = 30.0,
    leaf_depth: float = 12.0,
    leaf_thickness: float = 2.0,
    barrel_diameter: float = 6.0,
    pin_diameter: float = 3.0,
    knuckle_count: int = 5,
    knuckle_clearance: float = 0.35,
    barrel_gap: float = 0.4,
    attachment_overlap: float = 0.2,
    connector_width_scale: float = 0.75,
    leaf_index: int = 0,
    resolution: int = 64,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Create one leaf of a traditional barrel hinge.

    The hinge axis is the X-axis at Y=0, Z=0.
    """

    _validate_positive("width", width)
    _validate_positive("leaf_depth", leaf_depth)
    _validate_positive("leaf_thickness", leaf_thickness)
    _validate_positive("barrel_diameter", barrel_diameter)
    _validate_positive("pin_diameter", pin_diameter)
    if knuckle_count < 2:
        raise ValueError("knuckle_count must be >= 2.")
    if knuckle_clearance < 0:
        raise ValueError("knuckle_clearance must be >= 0.")
    if barrel_gap < 0:
        raise ValueError("barrel_gap must be >= 0.")
    if attachment_overlap < 0:
        raise ValueError("attachment_overlap must be >= 0.")
    if not (0.1 <= connector_width_scale <= 1.0):
        raise ValueError("connector_width_scale must be in [0.1, 1.0].")
    if leaf_index not in {0, 1}:
        raise ValueError("leaf_index must be 0 or 1.")

    barrel_radius = barrel_diameter / 2.0
    pin_hole_radius = (pin_diameter / 2.0) + (knuckle_clearance / 2.0)
    segment_width = width / float(knuckle_count)
    # Use clearance on both sides of each knuckle segment so opposing leaves do not fuse.
    knuckle_width = max(segment_width - (2.0 * knuckle_clearance), segment_width * 0.45)

    side = 1.0 if leaf_index == 0 else -1.0
    # Plate is offset away from the barrel; connectors bridge this gap.
    plate_center_y = side * (barrel_radius + barrel_gap + leaf_depth / 2.0)
    plate = make_box(size=(width, leaf_depth, leaf_thickness), center=(0.0, plate_center_y, 0.0))

    knuckles: list[Mesh] = []
    knuckle_centers: list[float] = []
    for idx in range(knuckle_count):
        if idx % 2 != leaf_index:
            continue
        x_center = -width / 2.0 + (idx + 0.5) * segment_width
        knuckle_centers.append(x_center)
        knuckle = make_cylinder(
            radius=barrel_radius,
            height=knuckle_width,
            center=(x_center, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            resolution=max(24, resolution),
        )
        knuckles.append(knuckle)

    # Explicit connector blocks keep each leaf visibly and structurally attached to its own knuckles.
    connectors: list[Mesh] = []
    barrel_edge_y = side * barrel_radius
    plate_inner_y = side * (barrel_radius + barrel_gap)
    connector_span = max(barrel_gap + (2.0 * attachment_overlap), leaf_thickness * 0.75)
    connector_center_y = side * (barrel_radius + barrel_gap / 2.0)
    connector_width = max(knuckle_width * connector_width_scale, knuckle_width * 0.35)
    for x_center in knuckle_centers:
        connector = make_box(
            size=(connector_width, connector_span, leaf_thickness),
            center=(x_center, connector_center_y, 0.0),
        )
        connectors.append(connector)

    leaf = _safe_union([plate, *knuckles, *connectors])
    pin_bore = make_cylinder(
        radius=pin_hole_radius,
        height=width + max(2.0, knuckle_clearance * 4.0),
        center=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        resolution=max(24, resolution),
    )
    leaf = _safe_difference(leaf, [pin_bore])

    if color is not None:
        set_mesh_color(leaf, color)
    return leaf


def make_traditional_hinge_pair(
    *,
    width: float = 30.0,
    leaf_depth: float = 12.0,
    leaf_thickness: float = 2.0,
    barrel_diameter: float = 6.0,
    pin_diameter: float = 3.0,
    knuckle_count: int = 5,
    knuckle_clearance: float = 0.35,
    barrel_gap: float = 0.4,
    attachment_overlap: float = 0.2,
    connector_width_scale: float = 0.75,
    opened_angle_deg: float = 0.0,
    include_pin: bool = True,
    pin_extension: float = 0.8,
    resolution: int = 64,
    leaf_a_color: Sequence[float] | str | None = "#7f8fa6",
    leaf_b_color: Sequence[float] | str | None = "#8f7f6a",
    pin_color: Sequence[float] | str | None = "#b0b0b0",
) -> MeshGroup:
    """Create a traditional two-leaf hinge assembly.

    Returns a MeshGroup so leaves and pin stay separate for assembly previews.
    """

    leaf_a = make_traditional_hinge_leaf(
        width=width,
        leaf_depth=leaf_depth,
        leaf_thickness=leaf_thickness,
        barrel_diameter=barrel_diameter,
        pin_diameter=pin_diameter,
        knuckle_count=knuckle_count,
        knuckle_clearance=knuckle_clearance,
        barrel_gap=barrel_gap,
        attachment_overlap=attachment_overlap,
        connector_width_scale=connector_width_scale,
        leaf_index=0,
        resolution=resolution,
        color=leaf_a_color,
    )
    leaf_b = make_traditional_hinge_leaf(
        width=width,
        leaf_depth=leaf_depth,
        leaf_thickness=leaf_thickness,
        barrel_diameter=barrel_diameter,
        pin_diameter=pin_diameter,
        knuckle_count=knuckle_count,
        knuckle_clearance=knuckle_clearance,
        barrel_gap=barrel_gap,
        attachment_overlap=attachment_overlap,
        connector_width_scale=connector_width_scale,
        leaf_index=1,
        resolution=resolution,
        color=leaf_b_color,
    )

    if abs(opened_angle_deg) > 1e-9:
        rotate(leaf_b, axis=(1.0, 0.0, 0.0), angle_deg=opened_angle_deg, origin=(0.0, 0.0, 0.0))

    parts: list[Mesh] = [leaf_a, leaf_b]
    if include_pin:
        pin = make_cylinder(
            radius=pin_diameter / 2.0,
            height=width + (2.0 * max(pin_extension, 0.0)),
            center=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            resolution=max(24, resolution),
            color=pin_color,
        )
        parts.append(pin)

    return group(parts)


def make_living_hinge(
    *,
    width: float = 50.0,
    height: float = 24.0,
    thickness: float = 1.8,
    hinge_band_width: float = 14.0,
    slit_width: float = 0.5,
    slit_pitch: float = 1.6,
    edge_margin: float = 1.5,
    bridge: float = 1.2,
    color: Sequence[float] | str | None = "#7d8f7a",
) -> Mesh:
    """Create a printable living hinge panel with alternating slit cuts."""

    _validate_positive("width", width)
    _validate_positive("height", height)
    _validate_positive("thickness", thickness)
    _validate_positive("hinge_band_width", hinge_band_width)
    _validate_positive("slit_width", slit_width)
    _validate_positive("slit_pitch", slit_pitch)
    if edge_margin < 0:
        raise ValueError("edge_margin must be >= 0.")
    if bridge <= 0:
        raise ValueError("bridge must be > 0.")
    if hinge_band_width > width:
        raise ValueError("hinge_band_width must be <= width.")

    slot_length = height - (2.0 * edge_margin) - bridge
    if slot_length <= 0:
        raise ValueError("height/edge_margin/bridge do not leave room for hinge slits.")

    panel = make_box(size=(width, height, thickness), center=(0.0, 0.0, 0.0))
    slot_count = max(1, int(np.floor((hinge_band_width - slit_width) / slit_pitch)) + 1)

    x_start = -hinge_band_width / 2.0 + slit_width / 2.0
    z_depth = thickness + 0.5
    cutters: list[Mesh] = []
    for idx in range(slot_count):
        x = x_start + idx * slit_pitch
        if x < -hinge_band_width / 2.0 - 1e-9 or x > hinge_band_width / 2.0 + 1e-9:
            continue

        if idx % 2 == 0:
            y_min = -height / 2.0 + edge_margin
            y_max = (height / 2.0) - edge_margin - bridge
        else:
            y_min = (-height / 2.0) + edge_margin + bridge
            y_max = height / 2.0 - edge_margin

        span = y_max - y_min
        if span <= 0:
            continue
        y_center = (y_min + y_max) / 2.0
        cutter = make_box(size=(slit_width, span, z_depth), center=(x, y_center, 0.0))
        cutters.append(cutter)

    living = _safe_difference(panel, cutters)
    if color is not None:
        set_mesh_color(living, color)
    return living


def make_bistable_hinge(
    *,
    width: float = 42.0,
    height: float = 16.0,
    thickness: float = 2.0,
    anchor_width: float = 10.0,
    shuttle_width: float = 9.0,
    shuttle_height: float = 6.0,
    ligament_width: float = 1.4,
    preload_offset: float = 2.5,
    color: Sequence[float] | str | None = "#8e7a9c",
) -> Mesh:
    """Create a bistable flexure hinge blank (double-ligament over-center style)."""

    _validate_positive("width", width)
    _validate_positive("height", height)
    _validate_positive("thickness", thickness)
    _validate_positive("anchor_width", anchor_width)
    _validate_positive("shuttle_width", shuttle_width)
    _validate_positive("shuttle_height", shuttle_height)
    _validate_positive("ligament_width", ligament_width)

    if (2.0 * anchor_width + shuttle_width) >= width:
        raise ValueError("anchor_width + shuttle_width leaves no room for flexure span.")
    if shuttle_height >= height:
        raise ValueError("shuttle_height must be < height.")

    left_anchor = make_box(
        size=(anchor_width, height, thickness),
        center=(-width / 2.0 + anchor_width / 2.0, 0.0, 0.0),
    )
    right_anchor = make_box(
        size=(anchor_width, height, thickness),
        center=(width / 2.0 - anchor_width / 2.0, 0.0, 0.0),
    )
    shuttle = make_box(
        size=(shuttle_width, shuttle_height, thickness),
        center=(0.0, preload_offset, 0.0),
    )

    left_inner_x = -width / 2.0 + anchor_width
    right_inner_x = width / 2.0 - anchor_width
    shuttle_left_x = -shuttle_width / 2.0
    shuttle_right_x = shuttle_width / 2.0

    top_anchor_y = height / 2.0 - ligament_width / 2.0
    bot_anchor_y = -height / 2.0 + ligament_width / 2.0
    top_shuttle_y = preload_offset + shuttle_height / 2.0 - ligament_width / 2.0
    bot_shuttle_y = preload_offset - shuttle_height / 2.0 + ligament_width / 2.0

    struts = [
        _make_strut((left_inner_x, top_anchor_y, 0.0), (shuttle_left_x, top_shuttle_y, 0.0), ligament_width, thickness),
        _make_strut((left_inner_x, bot_anchor_y, 0.0), (shuttle_left_x, bot_shuttle_y, 0.0), ligament_width, thickness),
        _make_strut((shuttle_right_x, top_shuttle_y, 0.0), (right_inner_x, top_anchor_y, 0.0), ligament_width, thickness),
        _make_strut((shuttle_right_x, bot_shuttle_y, 0.0), (right_inner_x, bot_anchor_y, 0.0), ligament_width, thickness),
    ]

    hinge = _safe_union([left_anchor, right_anchor, shuttle, *struts])
    if color is not None:
        set_mesh_color(hinge, color)
    return hinge


__all__ = [
    "make_traditional_hinge_leaf",
    "make_traditional_hinge_pair",
    "make_living_hinge",
    "make_bistable_hinge",
]
