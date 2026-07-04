from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from impression.modeling import Section, Station, as_section, loft_sections, make_circle

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _section_from_circles(centers: list[tuple[float, float]], radius: float):
    regions = [as_section(make_circle(radius=radius, center=center)).regions[0] for center in centers]
    return Section(tuple(regions), color=(0.62, 0.73, 0.86, 1.0))


def _frame(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    n = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=float)
    n /= np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, n))) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    u = up - np.dot(up, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v, n


def build():
    # Real-world framing: a compact air splitter manifold where branches
    # reconfigure through the run while preserving printable watertight solids.
    s0 = _section_from_circles([(-6.0, 0.0), (6.0, 0.0)], radius=2.7)
    s1 = _section_from_circles([(-7.2, 0.0), (0.0, 0.0), (7.2, 0.0)], radius=2.35)
    s2 = _section_from_circles([(-6.5, 0.0), (6.5, 0.0)], radius=2.7)

    defs = [
        (0.0, s0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0),
        (0.5, s1, np.array([1.1, 0.2, 14.0]), 2.8, 1.7),
        (1.0, s2, np.array([2.2, -0.1, 28.0]), 5.2, 2.9),
    ]
    stations = []
    for t, section, origin, yaw, pitch in defs:
        u, v, n = _frame(yaw, pitch)
        stations.append(Station(t=t, section=section, origin=origin, u=u, v=v, n=n))

    manifold = loft_sections(
        stations,
        samples=72,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    label = make_label("SPLITTER MANIFOLD (2->3->2)", center=(1.0, 18.0, 30.0), font_size=8.0)
    return [manifold, label]
