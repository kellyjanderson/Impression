from __future__ import annotations

from impression.modeling import (
    Path2D,
    Profile2D,
    loft_endcaps,
    make_circle,
    make_polygon,
    make_rect,
    translate,
)


def _profile_from_points(points, color=None):
    outer = Path2D.from_points(points, closed=True)
    profile = Profile2D(outer=outer)
    if color is not None:
        profile.with_color(color)
    return profile


def _with_hole():
    outer = make_rect(size=(12.0, 8.0), color="#8b7a6a").outer
    hole = make_circle(radius=2.0).outer
    return Profile2D(outer=outer, holes=[hole]).with_color("#8b7a6a")


def _letter_l():
    points = [
        (0.0, 0.0),
        (8.0, 0.0),
        (8.0, 2.0),
        (2.0, 2.0),
        (2.0, 10.0),
        (0.0, 10.0),
    ]
    return _profile_from_points(points, color="#7f8fa6")


def _acute():
    return make_polygon(
        points=[(0.0, 0.0), (10.0, 1.5), (9.0, 8.0), (2.0, 10.0)],
        color="#7a8f7c",
    )


def _curved():
    return make_circle(radius=5.0, color="#7e6f86")


def build():
    profiles = [
        ("hole", _with_hole()),
        ("l_shape", _letter_l()),
        ("acute", _acute()),
        ("curved", _curved()),
    ]

    amounts = {
        "hole": {"ROUND": 1.0, "CHAMFER": 0.6, "COVE": 0.6},
        "l_shape": {"ROUND": 1.2, "CHAMFER": 0.6, "COVE": 1.0},
        "acute": {"ROUND": 1.0, "CHAMFER": 0.6, "COVE": 0.6},
        "curved": {"ROUND": 2.0, "CHAMFER": 1.5, "COVE": 1.5},
    }

    cap_modes = ["FLAT", "CHAMFER", "ROUND", "COVE"]
    meshes = []
    spacing_x = 18.0
    spacing_y = 18.0

    for row, (name, profile) in enumerate(profiles):
        for col, mode in enumerate(cap_modes):
            if mode == "FLAT":
                amount = 0.0
            else:
                amount = amounts[name][mode]
            lofted = loft_endcaps(
                [profile, profile],
                endcap_mode=mode,
                endcap_amount=amount,
                endcap_steps=24,
                endcap_placement="BOTH",
            )
            translate(lofted, (col * spacing_x, -row * spacing_y, 0.0))
            meshes.append(lofted)

    return meshes
