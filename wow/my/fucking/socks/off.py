from __future__ import annotations

import math
import time

import numpy as np

from impression.modeling import (
    Path2D,
    Path3D,
    Section,
    Station,
    loft,
    loft_endcaps,
    loft_sections,
    make_box,
    make_circle,
    make_polygon,
    make_sphere,
    make_text,
    make_torus,
    as_section,
    rotate,
    text_profiles,
    translate,
)
from impression.modeling.drawing2d import PlanarShape2D


# Preview will re-run build() on this interval without touching files.
ANIMATE_INTERVAL_SECONDS = 1.0
PREVIEW_SHOW_BOUNDS = False
PREVIEW_SHOW_AXES = True
ENABLE_BACKGROUND_PRIMITIVES = False
_START_TIME: float | None = None
_TEXT_FORWARD = (0.0, -1.0, 0.0)


def _elapsed() -> float:
    global _START_TIME
    now = time.monotonic()
    if _START_TIME is None:
        _START_TIME = now
    return now - _START_TIME


def _frame(yaw_deg: float, pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    n = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ],
        dtype=float,
    )
    n /= np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(up, n))) > 0.92:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(up, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v, n


def _hex_rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {value!r}")
    r = int(text[0:2], 16) / 255.0
    g = int(text[2:4], 16) / 255.0
    b = int(text[4:6], 16) / 255.0
    return (r, g, b, float(alpha))


def _hud(elapsed: float, title: str, subtitle: str) -> list:
    t = make_text(
        title,
        depth=0.28,
        direction=_TEXT_FORWARD,
        font_size=1.05,
        justify="left",
        color="#ddd6c8",
    )
    s = make_text(
        subtitle,
        depth=0.16,
        direction=_TEXT_FORWARD,
        font_size=0.56,
        justify="left",
        color="#99a6b2",
    )
    timer = make_text(
        f"T+{int(elapsed):03d}s",
        depth=0.14,
        direction=_TEXT_FORWARD,
        font_size=0.52,
        justify="left",
        color="#87c0a5",
    )
    translate(t, (-6.2, 4.9, 0.4))
    translate(s, (-6.2, 3.8, 0.4))
    translate(timer, (4.7, 4.9, 0.4))
    return [t, s, timer]


def _distance_field(elapsed: float) -> list:
    objs = []
    phase = elapsed * 0.22
    for i in range(14):
        ang = phase + i * 0.35
        rad = 8.0 + (i % 3) * 0.9
        x = math.cos(ang) * rad
        y = math.sin(ang * 1.3) * 3.4
        z = 3.0 + math.sin(ang * 0.7) * 1.2
        s = make_sphere(radius=0.12 + (i % 3) * 0.03, center=(x, y, z), color="#5f6f8a")
        objs.append(s)
    for j in range(4):
        ang = -phase * 0.6 + j * 1.2
        ring = make_torus(
            major_radius=0.42 + 0.07 * j,
            minor_radius=0.03,
            center=(math.cos(ang) * 6.2, 3.7 + j * 0.6, 4.3 + math.sin(ang) * 0.5),
            color="#4e5e73",
            n_theta=56,
            n_phi=18,
        )
        objs.append(ring)
    return objs


def _act_path_ballet(elapsed: float) -> list:
    p = elapsed * 0.08
    profiles = [
        make_circle(radius=1.3 + 0.12 * math.sin(p), color="#8ab3cd"),
        make_polygon(
            points=[
                (-1.4, -1.0),
                (1.2, -1.2),
                (1.7, 0.0),
                (0.9, 1.2),
                (-1.0, 1.0),
                (-1.6, 0.0),
            ],
            color="#9cb89f",
        ),
        make_box(size=(2.4, 1.7, 0.01), center=(0, 0, 0), color="#b59fc9"),
        make_polygon(
            points=[
                (-1.1, -0.8),
                (1.3, -1.1),
                (1.1, 0.9),
                (0.2, 1.4),
                (-1.4, 0.9),
            ],
            color="#c3ab8f",
        ),
        make_circle(radius=1.0 + 0.08 * math.cos(p * 1.4), color="#8ab3cd"),
    ]
    # Use only XY loops from box footprint.
    profiles[2] = make_polygon(
        points=[(-1.2, -0.85), (1.2, -0.85), (1.2, 0.85), (-1.2, 0.85)],
        color="#b59fc9",
    )
    path = Path3D.from_points(
        [
            (-5.2, -2.8 + math.sin(p) * 0.3, 0.0),
            (-2.6, 1.6, 2.0 + math.sin(p * 0.8) * 0.5),
            (0.0, -1.4, 4.0),
            (2.9, 2.0, 6.0 + math.cos(p) * 0.4),
            (5.4, -0.6, 8.2),
        ]
    )
    body = loft(
        profiles,
        path=path,
        samples=88,
        start_cap="dome",
        end_cap="slope",
        start_cap_length=1.3,
        end_cap_length=1.9,
        cap_steps=9,
        cap_scale_dims="both",
    )
    body.color = _hex_rgba("#83aeca")
    return [body]


def _act_endcap_forge(elapsed: float) -> list:
    phase = elapsed * 0.1
    shape_a = make_polygon(
        points=[(-2.1, -1.5), (2.2, -1.2), (1.4, 2.2), (-1.9, 1.1)],
        color="#9db79f",
    )
    shape_b = PlanarShape2D(
        outer=Path2D.from_points([(0.0, 0.0), (2.9, 0.0), (2.9, 0.7), (0.8, 0.7), (0.8, 3.4), (0.0, 3.4)], closed=True),
        holes=[],
    ).with_color("#a6b5c7")
    shape_c = PlanarShape2D(
        outer=make_polygon(points=[(-2.8, -1.9), (2.8, -1.9), (2.8, 1.9), (-2.8, 1.9)]).outer,
        holes=[make_circle(radius=0.9).outer],
    ).with_color("#c2b09a")

    modes = [("CHAMFER", 0.36), ("ROUND", 0.30), ("COVE", 0.26)]
    shapes = [shape_a, shape_b, shape_c]
    scene = []
    for idx, (mode, amount) in enumerate(modes):
        for row, profile in enumerate(shapes):
            mesh = loft_endcaps(
                [profile, profile],
                endcap_mode=mode,
                endcap_amount=amount,
                endcap_steps=26,
                endcap_placement="BOTH",
            )
            x = -4.8 + idx * 4.8
            y = -4.4 + row * 4.1
            z = math.sin(phase + idx * 0.7 + row * 0.5) * 0.24
            translate(mesh, (x, y, z))
            scene.append(mesh)
    return scene


def _act_topology_orbit(elapsed: float) -> list:
    p = elapsed * 0.12
    main0 = make_polygon(points=[(-1.9, -1.4), (1.9, -1.4), (1.9, 1.4), (-1.9, 1.4)], color="#88a8c2")
    main1 = PlanarShape2D(outer=main0.outer, holes=[make_circle(radius=0.55).outer]).with_color("#88a8c2")
    sat = make_circle(radius=0.72, center=(2.7, 0.0), color="#c6a98f")

    section0 = Section((as_section(main0).regions[0],))
    section1 = Section((as_section(main1).regions[0],))
    section2 = Section((as_section(main1).regions[0], as_section(sat).regions[0]))
    section3 = Section((as_section(main0).regions[0],))

    defs = [
        (0.0, section0, np.array([-2.0, -0.5, 0.0]), 0.0, 0.0),
        (1.0, section1, np.array([-0.7, 0.7, 1.9]), 8.0 + math.sin(p) * 9.0, 2.5),
        (2.0, section2, np.array([0.9, -0.8, 3.9]), -15.0 + math.cos(p) * 8.0, -4.0),
        (3.0, section3, np.array([2.3, 0.7, 5.9]), 9.0, 1.5),
    ]
    stations = []
    for t, section, origin, yaw, pitch in defs:
        u, v, n = _frame(yaw, pitch)
        stations.append(Station(t=t, section=section, origin=origin, u=u, v=v, n=n))
    mesh = loft_sections(stations, samples=74, cap_ends=True)
    mesh.color = _hex_rgba("#7ea6c3")
    return [mesh]


def _act_text_takeover(elapsed: float) -> list:
    spin = elapsed * 0.06
    profiles = text_profiles(
        "SOCKS OFF",
        font_size=2.6,
        justify="center",
        letter_spacing=0.08,
        color="#c7b59d",
    )
    scene = []
    for i, profile in enumerate(profiles):
        h = 0.42 + 0.09 * math.sin(spin + i * 0.35)
        part = loft(
            [profile, profile],
            samples=52,
            start_cap="dome",
            end_cap="dome",
            start_cap_length=h,
            end_cap_length=h,
            cap_steps=7,
        )
        # Stand text upright and make it face forward instead of down.
        rotate(part, axis=(1.0, 0.0, 0.0), angle_deg=-90.0)
        translate(part, (0.0, 0.0, 0.22 * math.sin(spin * 1.5 + i * 0.2)))
        part.color = _hex_rgba("#c7b59d")
        scene.append(part)
    plinth = make_box(size=(16.2, 4.6, 0.35), center=(0.0, -0.2, -0.45), color="#4c5969")
    scene.append(plinth)
    return scene


def _finale() -> list:
    msg = make_text(
        "Here are your sox bak. Have a good rest of your day.",
        depth=0.24,
        direction=_TEXT_FORWARD,
        font_size=0.86,
        justify="center",
        color="#d8d0c4",
    )
    translate(msg, (0.0, 0.0, 0.15))
    base = make_box(size=(16.0, 4.8, 0.42), center=(0.0, 0.0, -0.38), color="#4b5a6b")
    ring_l = make_torus(major_radius=1.3, minor_radius=0.17, center=(-6.1, 0.0, 0.22), color="#829fb9")
    ring_r = make_torus(major_radius=1.3, minor_radius=0.17, center=(6.1, 0.0, 0.22), color="#829fb9")
    return [base, ring_l, ring_r, msg]


def build_at(elapsed: float):
    total = 180.0
    t = min(elapsed, total)

    if t < 45.0:
        title = "Act I: Path Ballet"
        subtitle = "Twist-minimized frames and cap choreography"
        scene = _act_path_ballet(t)
    elif t < 90.0:
        title = "Act II: Endcap Forge"
        subtitle = "CHAMFER / ROUND / COVE on sharp and holed profiles"
        scene = _act_endcap_forge(t - 45.0)
    elif t < 135.0:
        title = "Act III: Topology Orbit"
        subtitle = "Section-native region/hole birth and death"
        scene = _act_topology_orbit(t - 90.0)
    elif t < 170.0:
        title = "Act IV: Text Takeover"
        subtitle = "Native text profiles sculpted with loft"
        scene = _act_text_takeover(t - 135.0)
    else:
        title = "Finale"
        subtitle = "Socks successfully returned"
        scene = _finale()

    if t < 170.0 and ENABLE_BACKGROUND_PRIMITIVES:
        scene.extend(_distance_field(t))
    scene.extend(_hud(t, title, subtitle))
    return scene


def build():
    return build_at(_elapsed())
