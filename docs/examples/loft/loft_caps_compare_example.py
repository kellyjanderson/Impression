"""Side-by-side comparison of mesh vs SDF endcaps."""

from __future__ import annotations

from impression.modeling import loft, make_text
from impression.modeling.drawing2d import make_rect


def _make_profiles():
    # Keep cross-sections identical so side walls are a clean baseline;
    # this isolates cap behavior in the comparison.
    return [make_rect(size=(1.0, 0.6)), make_rect(size=(1.0, 0.6))]


def build():
    profiles = _make_profiles()

    mesh_caps = loft(
        profiles,
        start_cap="dome",
        end_cap="dome",
        start_cap_length=0.6,
        end_cap_length=0.6,
        cap_scale_dims="both",
    )
    mesh_caps.translate((-1.2, 0.8, 0.0), inplace=True)

    sdf_caps = loft(
        profiles,
        cap_type="sdf",
        cap_radius=0.2,
        cap_grid_spacing=0.1,
    )
    sdf_caps.translate((1.2, 0.8, 0.0), inplace=True)

    flat_caps = loft(profiles, cap_ends=True)
    flat_caps.translate((-1.2, -0.8, 0.0), inplace=True)

    no_caps = loft(profiles)
    no_caps.translate((1.2, -0.8, 0.0), inplace=True)

    label_mesh = make_text(
        "mesh",
        depth=0.05,
        font_size=0.22,
        center=(-1.2, 0.2, 0.0),
        color="#7a6b5e",
    )
    label_sdf = make_text(
        "sdf",
        depth=0.05,
        font_size=0.22,
        center=(1.2, 0.2, 0.0),
        color="#6b778a",
    )
    label_flat = make_text(
        "flat",
        depth=0.05,
        font_size=0.22,
        center=(-1.2, -1.4, 0.0),
        color="#7a8b6b",
    )
    label_none = make_text(
        "none",
        depth=0.05,
        font_size=0.22,
        center=(1.2, -1.4, 0.0),
        color="#8b6b7a",
    )

    return [mesh_caps, sdf_caps, flat_caps, no_caps, label_mesh, label_sdf, label_flat, label_none]


if __name__ == "__main__":
    build()
