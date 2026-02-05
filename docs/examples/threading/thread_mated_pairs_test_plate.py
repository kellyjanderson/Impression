"""Thread mating test plate with one male/female pair per supported thread profile.

Print this model to physically validate thread engagement across thread families.
Each pair contains:
- Left: external threaded rod
- Right: round nut/coupler with matching internal thread
"""

from impression.modeling import (
    MeshQuality,
    ThreadSpec,
    apply_fit,
    group,
    lookup_standard_thread,
    make_round_nut,
    make_threaded_rod,
    translate,
)


QUALITY = MeshQuality(segments_per_turn=22, circumferential_segments=72)
PAIR_X = 18.0
CELL_X = 44.0
CELL_Y = 24.0


def _make_pair(base_spec: ThreadSpec):
    male = apply_fit(base_spec, "fdm_default", kind="external")
    female = apply_fit(base_spec, "fdm_default", kind="internal")

    rod = make_threaded_rod(male, quality=QUALITY)
    nut = make_round_nut(
        female,
        thickness=max(6.0, min(10.0, female.length * 0.75)),
        outer_diameter=max(female.major_diameter * 2.0, female.major_diameter + 6.0),
        quality=QUALITY,
    )

    # Put both on build plate (z>=0) and side-by-side.
    rod_z_shift = -rod.bounds[4]
    nut_z_shift = -nut.bounds[4]
    translate(rod, (0.0, 0.0, rod_z_shift))
    translate(nut, (PAIR_X, 0.0, nut_z_shift))
    return [rod, nut]


def build():
    specs = [
        lookup_standard_thread("metric", "M6x1", length=10.0),
        lookup_standard_thread("unified", "1/4-20", length=10.0),
        lookup_standard_thread("acme", major_diameter=8.0, pitch=2.0, length=10.0),
        lookup_standard_thread("trapezoidal", major_diameter=8.0, pitch=2.0, length=10.0),
        lookup_standard_thread("pipe", major_diameter=10.0, tpi=16.0, length=10.0),
        ThreadSpec(major_diameter=8.0, pitch=1.25, length=10.0, profile="whitworth"),
        ThreadSpec(major_diameter=8.0, pitch=1.50, length=10.0, profile="square"),
        ThreadSpec(major_diameter=8.0, pitch=1.50, length=10.0, profile="buttress"),
        ThreadSpec(major_diameter=8.0, pitch=1.50, length=10.0, profile="rounded"),
    ]

    meshes = []
    for idx, spec in enumerate(specs):
        row = idx // 3
        col = idx % 3
        pair_meshes = _make_pair(spec)
        x = col * CELL_X
        y = row * CELL_Y
        for m in pair_meshes:
            translate(m, (x, y, 0.0))
            meshes.append(m)

    return group(meshes)
