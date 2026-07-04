"""Thread mating test plate with one male/female pair per supported thread profile.

Print this model to physically validate thread engagement across thread families.
Each pair contains:
- Left: external threaded rod
- Right: round nut/coupler with matching internal thread
"""

import numpy as np

from impression.modeling import (
    MeshQuality,
    ThreadSpec,
    apply_fit,
    boolean_union,
    boolean_difference,
    group,
    lookup_standard_thread,
    make_cylinder,
    make_ngon,
    make_round_nut,
    make_threaded_rod,
    make_text,
    translate,
)


QUALITY = MeshQuality(segments_per_turn=24, circumferential_segments=84)
PAIR_X = 32.0
CELL_X = 70.0
CELL_Y = 40.0


def _add_key_marker(
    center: tuple[float, float, float],
    outer_radius: float,
    sides: int,
    height: float,
    radius: float,
    z_offset: float,
):
    marker_center = (
        center[0] + outer_radius + radius * 0.7,
        center[1],
        center[2] + z_offset + height * 0.5,
    )
    return make_ngon(
        sides=sides,
        radius=radius,
        height=height,
        center=marker_center,
        direction=(0.0, 0.0, 1.0),
    )


def _add_rod_knob(rod, diameter: float, length: float, key_sides: int):
    """Add a grippy disk knob on the rod end (external threads only)."""

    b = rod.bounds
    knob_radius = max(8.0, diameter * 1.3)
    knob_thickness = 2.0
    center = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, b[5] + knob_thickness / 2.0)
    knob = make_cylinder(
        radius=knob_radius,
        height=knob_thickness,
        center=center,
        direction=(0.0, 0.0, 1.0),
        resolution=96,
    )

    # Cut radial scallops for grip using the key shape.
    cutouts = []
    grip_count = 12
    grip_radius = knob_radius * 0.2
    ring_radius = knob_radius - grip_radius * 0.6
    for i in range(grip_count):
        angle = 2.0 * np.pi * i / grip_count
        cx = center[0] + ring_radius * np.cos(angle)
        cy = center[1] + ring_radius * np.sin(angle)
        cutouts.append(
            make_ngon(
                sides=key_sides,
                radius=grip_radius,
                height=knob_thickness * 1.4,
                center=(cx, cy, center[2]),
                direction=(0.0, 0.0, 1.0),
            )
        )
    knob = boolean_difference(knob, cutouts)

    marker = _add_key_marker(
        center=center,
        outer_radius=knob_radius,
        sides=key_sides,
        height=knob_thickness * 1.4,
        radius=max(1.6, knob_radius * 0.12),
        z_offset=-knob_thickness * 0.2,
    )
    knob = boolean_difference(knob, [marker])
    return boolean_union([rod, knob])


def _engrave_label(nut, label: str, font_size: float, depth: float):
    b = nut.bounds
    center = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, b[5] - depth * 0.6)
    text_mesh = make_text(
        label,
        depth=depth,
        center=center,
        direction=(0.0, 0.0, 1.0),
        font_size=font_size,
        justify="center",
        valign="middle",
    )
    return boolean_difference(nut, [text_mesh])


def _make_pair(base_spec: ThreadSpec, label: str, key_sides: int):
    male = apply_fit(base_spec, "fdm_default", kind="external")
    female = apply_fit(base_spec, "fdm_default", kind="internal")

    rod = make_threaded_rod(male, quality=QUALITY)
    rod = _add_rod_knob(rod, male.major_diameter, male.length, key_sides)
    # Flip so the knob sits on the build plate.
    rod = rod.rotate_vector((1.0, 0.0, 0.0), 180.0, point=(0.0, 0.0, 0.0), inplace=False)

    nut_outer_diameter = max(female.major_diameter * 2.2, female.major_diameter + 8.0)
    nut = make_round_nut(
        female,
        thickness=max(8.0, min(14.0, female.length * 0.85)),
        outer_diameter=nut_outer_diameter,
        quality=QUALITY,
    )
    nut_center = ((nut.bounds[0] + nut.bounds[1]) * 0.5, (nut.bounds[2] + nut.bounds[3]) * 0.5, nut.bounds[4])
    nut_marker = _add_key_marker(
        center=nut_center,
        outer_radius=nut_outer_diameter / 2.0,
        sides=key_sides,
        height=3.0,
        radius=max(1.8, nut_outer_diameter * 0.12),
        z_offset=0.0,
    )
    nut = boolean_union([nut, nut_marker])
    nut = _engrave_label(nut, label, font_size=4.0, depth=0.8)

    # Put both on build plate (z>=0) and side-by-side.
    rod_z_shift = -rod.bounds[4]
    nut_z_shift = -nut.bounds[4]
    translate(rod, (0.0, 0.0, rod_z_shift))
    translate(nut, (PAIR_X, 0.0, nut_z_shift))
    return [rod, nut]


def build():
    specs = [
        ("ISO M8x1.25", lookup_standard_thread("metric", "M8x1.25", length=14.0), 3),
        ("UN 1/4-20", lookup_standard_thread("unified", "1/4-20", length=14.0), 4),
        ("ACME 10x2", lookup_standard_thread("acme", major_diameter=10.0, pitch=2.0, length=14.0), 5),
        ("TR 10x2", lookup_standard_thread("trapezoidal", major_diameter=10.0, pitch=2.0, length=14.0), 6),
        ("Pipe 1/16", lookup_standard_thread("pipe", major_diameter=12.0, tpi=16.0, length=14.0), 7),
        ("Whitworth", ThreadSpec(major_diameter=10.0, pitch=1.5, length=14.0, profile="whitworth"), 8),
        ("Square", ThreadSpec(major_diameter=10.0, pitch=1.5, length=14.0, profile="square"), 9),
        ("Buttress", ThreadSpec(major_diameter=10.0, pitch=1.5, length=14.0, profile="buttress"), 10),
        ("Rounded", ThreadSpec(major_diameter=10.0, pitch=1.5, length=14.0, profile="rounded"), 11),
    ]

    meshes = []
    for idx, (label, spec, sides) in enumerate(specs):
        row = idx // 3
        col = idx % 3
        pair_meshes = _make_pair(spec, label, sides)
        x = col * CELL_X
        y = row * CELL_Y
        for m in pair_meshes:
            translate(m, (x, y, 0.0))
            meshes.append(m)

    return group(meshes)
