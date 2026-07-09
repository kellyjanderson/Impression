"""Loft showcase director.

Run once:
    impression preview docs/examples/loft/loft_showcase.py

Then switch demos:
    python scripts/dev/loft_demo.py list
    python scripts/dev/loft_demo.py set caps-lab
    python scripts/dev/loft_demo.py play --interval 6
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

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
    make_rect,
    make_text,
    as_section,
    text_profiles,
    translate,
)
from impression.modeling.drawing2d import PlanarShape2D

SHOWCASE_SELECTOR = Path(__file__).with_name(".loft-showcase")
DEMO_ORDER = (
    "caps-lab",
    "path-choreo",
    "topology-transitions",
    "text-sculpt",
)


def _read_selected_demo() -> str:
    env_name = os.environ.get("IMPRESSION_LOFT_DEMO")
    if env_name:
        return env_name.strip().lower()
    if SHOWCASE_SELECTOR.exists():
        value = SHOWCASE_SELECTOR.read_text().strip().lower()
        if value:
            return value
    return DEMO_ORDER[0]


def _title_mesh(title: str, subtitle: str) -> list:
    title_mesh = make_text(
        title,
        depth=0.18,
        font_size=0.6,
        justify="left",
        color="#d7d3cb",
    )
    subtitle_mesh = make_text(
        subtitle,
        depth=0.1,
        font_size=0.34,
        justify="left",
        color="#8f99a3",
    )
    translate(title_mesh, (-7.2, 6.2, 0.2))
    translate(subtitle_mesh, (-7.2, 5.2, 0.2))
    return [title_mesh, subtitle_mesh]


def _rect_with_hole(size: tuple[float, float], hole_radius: float, color: str) -> PlanarShape2D:
    outer = make_rect(size=size).outer
    hole = make_circle(radius=hole_radius).outer
    return PlanarShape2D(outer=outer, holes=[hole]).with_color(color)


def _l_profile(scale: float, color: str) -> PlanarShape2D:
    points = [
        (0.0, 0.0),
        (2.3 * scale, 0.0),
        (2.3 * scale, 0.55 * scale),
        (0.65 * scale, 0.55 * scale),
        (0.65 * scale, 3.0 * scale),
        (0.0, 3.0 * scale),
    ]
    return PlanarShape2D(outer=Path2D.from_points(points, closed=True)).with_color(color)


def _station_frame(yaw_deg: float, pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if abs(float(np.dot(up, n))) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(up, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v, n


def _demo_caps_lab() -> list:
    cap_modes = ["FLAT", "CHAMFER", "ROUND", "COVE"]
    shapes: list[tuple[str, PlanarShape2D]] = [
        ("Circle", make_circle(radius=2.1, color="#b7a4cd")),
        ("Acute", make_polygon(points=[(-2.2, -1.8), (2.1, -1.3), (1.5, 2.2), (-1.8, 1.4)], color="#9bb59e")),
        ("L", _l_profile(scale=1.35, color="#a5b6c8")),
        ("Hole", _rect_with_hole((5.1, 3.5), hole_radius=0.9, color="#c1b09a")),
    ]
    amounts = {
        "Circle": {"FLAT": 0.0, "CHAMFER": 0.55, "ROUND": 0.45, "COVE": 0.45},
        "Acute": {"FLAT": 0.0, "CHAMFER": 0.35, "ROUND": 0.25, "COVE": 0.25},
        "L": {"FLAT": 0.0, "CHAMFER": 0.16, "ROUND": 0.12, "COVE": 0.10},
        "Hole": {"FLAT": 0.0, "CHAMFER": 0.30, "ROUND": 0.22, "COVE": 0.20},
    }

    scene = []
    spacing_x = 9.4
    spacing_y = 8.8
    origin_x = -14.2
    origin_y = 2.2

    for row, (shape_name, profile) in enumerate(shapes):
        for col, mode in enumerate(cap_modes):
            mesh = loft_endcaps(
                [profile, profile],
                endcap_mode=mode,
                endcap_amount=amounts[shape_name][mode],
                endcap_steps=24,
                endcap_placement="BOTH",
            )
            translate(mesh, (origin_x + col * spacing_x, origin_y - row * spacing_y, 0.0))
            scene.append(mesh)

    for col, mode in enumerate(cap_modes):
        label = make_text(mode, depth=0.12, font_size=0.36, justify="center", color="#d8d2c8")
        translate(label, (origin_x + col * spacing_x, 4.8, 0.1))
        scene.append(label)

    for row, (name, _) in enumerate(shapes):
        label = make_text(name, depth=0.1, font_size=0.3, justify="left", color="#aeb7bf")
        translate(label, (-19.0, origin_y - row * spacing_y, 0.1))
        scene.append(label)

    scene.extend(_title_mesh("Loft Endcap Lab", "FLAT / CHAMFER / ROUND / COVE on mixed profile topology"))
    return scene


def _demo_path_choreo() -> list:
    profiles = [
        make_circle(radius=1.2, color="#8eb6d3"),
        make_polygon(points=[(-1.2, -1.0), (1.2, -1.0), (1.6, 0.0), (0.9, 1.0), (-0.9, 1.0), (-1.6, 0.0)], color="#9bb69f"),
        make_rect(size=(2.4, 1.6), color="#b4a2c6"),
        make_polygon(points=[(-1.3, -0.6), (1.1, -1.2), (1.5, 0.4), (0.4, 1.3), (-1.2, 1.1)], color="#c5af95"),
        make_circle(radius=1.0, color="#8eb6d3"),
    ]
    path = Path3D.from_points(
        [
            (-4.0, -2.5, 0.0),
            (-2.2, 1.0, 1.8),
            (0.0, -1.3, 3.6),
            (2.4, 1.8, 5.2),
            (4.4, -0.5, 7.0),
        ]
    )
    body = loft(
        profiles,
        path=path,
        samples=84,
        start_cap="dome",
        end_cap="slope",
        start_cap_length=1.1,
        end_cap_length=1.8,
        cap_steps=8,
        cap_scale_dims="both",
    )
    body.color = "#7fabc8"
    scene = [body]
    scene.extend(_title_mesh("Path Choreography", "Parallel-transport orientation + asymmetric endcaps"))
    return scene


def _demo_topology_transitions() -> list:
    main_no_hole = make_rect(size=(3.6, 2.8), color="#89a9c2")
    main_with_hole = _rect_with_hole((3.6, 2.8), hole_radius=0.52, color="#89a9c2")
    satellite = make_circle(radius=0.74, center=(2.8, 0.0), color="#c5a88e")

    section0 = Section((as_section(main_no_hole).regions[0],))
    section1 = Section((as_section(main_with_hole).regions[0],))
    section2 = Section((as_section(main_with_hole).regions[0], as_section(satellite).regions[0]))
    section3 = Section((as_section(main_no_hole).regions[0],))

    stations = []
    defs = [
        (0.0, section0, np.array([-1.8, -0.8, 0.0]), 0.0, 0.0),
        (1.0, section1, np.array([-0.8, 0.5, 1.7]), 8.0, 3.0),
        (2.0, section2, np.array([0.9, -0.4, 3.6]), -12.0, -4.0),
        (3.0, section3, np.array([2.1, 0.7, 5.4]), 6.0, 2.0),
    ]
    for t, section, origin, yaw, pitch in defs:
        u, v, n = _station_frame(yaw, pitch)
        stations.append(Station(t=t, section=section, origin=origin, u=u, v=v, n=n))

    mesh = loft_sections(stations, samples=72, cap_ends=True)
    mesh.color = "#7da2bf"
    scene = [mesh]
    scene.extend(_title_mesh("Topology Transitions", "Region/hole birth + death across explicit station frames"))
    return scene


def _demo_text_sculpt() -> list:
    profiles = text_profiles(
        "LOFT",
        font_size=2.2,
        justify="left",
        font="Arial",
        color="#c6b39a",
        letter_spacing=0.1,
    )
    letters = []
    for profile in profiles:
        part = loft(
            [profile, profile],
            samples=56,
            start_cap="dome",
            end_cap="dome",
            start_cap_length=0.55,
            end_cap_length=0.55,
            cap_steps=8,
        )
        part.color = "#c6b39a"
        letters.append(part)

    plinth = make_box(size=(12.0, 3.2, 0.45), center=(4.3, -0.3, -0.45), color="#495666")
    scene = [plinth, *letters]
    scene.extend(_title_mesh("Text as Loft Source", "text_profiles -> loft() with dome caps"))
    return scene


def _build_demo(name: str) -> list:
    key = name.strip().lower()
    if key == "auto":
        idx = int(time.time() // 6) % len(DEMO_ORDER)
        key = DEMO_ORDER[idx]
    demos = {
        "caps-lab": _demo_caps_lab,
        "path-choreo": _demo_path_choreo,
        "topology-transitions": _demo_topology_transitions,
        "text-sculpt": _demo_text_sculpt,
    }
    builder = demos.get(key)
    if builder is None:
        fallback = make_text(
            "Unknown demo: " + name,
            depth=0.14,
            font_size=0.45,
            justify="left",
            color="#d8c7c7",
        )
        translate(fallback, (-6.2, 0.0, 0.0))
        hint = make_text(
            "Valid: " + ", ".join(DEMO_ORDER) + ", auto",
            depth=0.08,
            font_size=0.28,
            justify="left",
            color="#9faab5",
        )
        translate(hint, (-6.2, -0.8, 0.0))
        return [fallback, hint]
    return builder()


def build():
    return _build_demo(_read_selected_demo())
