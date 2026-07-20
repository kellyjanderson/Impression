"""SDF loft with rounded endcaps on text profiles."""

from __future__ import annotations

from impression.modeling import loft_sdf, text_profiles


def build():
    profiles = text_profiles(
        "LOFT",
        font_size=5.0,
        font="Helvetica",
        color="#8b7a6a",
    )
    meshes = []
    for profile in profiles:
        meshes.append(
            loft_sdf(
                [profile, profile],
                positions=[0.0, 3.0],
                cap_radius=0.6,
                grid_spacing=0.2,
            )
        )
    return meshes


if __name__ == "__main__":
    build()
