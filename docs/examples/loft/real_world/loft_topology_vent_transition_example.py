from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from impression.modeling import Section, Station, as_section, loft_sections, make_rect
from impression.modeling.drawing2d import PlanarShape2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _ring_rect(size: tuple[float, float], wall: float, color: str) -> PlanarShape2D:
    outer = make_rect(size=size, color=color).outer
    inner = make_rect(size=(size[0] - 2.0 * wall, size[1] - 2.0 * wall)).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


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
    s0 = _ring_rect(size=(13.0, 8.0), wall=2.1, color="#8ea4c6")
    s1 = _ring_rect(size=(14.6, 8.8), wall=2.2, color="#8ea4c6")
    s2 = _ring_rect(size=(16.8, 9.8), wall=2.4, color="#8ea4c6")
    s3 = _ring_rect(size=(19.0, 11.0), wall=2.5, color="#8ea4c6")

    sections = [Section((as_section(s).regions[0],)) for s in (s0, s1, s2, s3)]

    defs = [
        (0.0, sections[0], np.array([0.0, 0.0, 0.0]), 0.0, 0.0),
        (1.0, sections[1], np.array([2.0, 0.7, 7.0]), 8.0, 4.0),
        (2.0, sections[2], np.array([4.5, -0.9, 14.0]), -9.0, -3.0),
        (3.0, sections[3], np.array([6.8, 0.5, 21.0]), 5.0, 2.0),
    ]
    stations = []
    for t, section, origin, yaw, pitch in defs:
        u, v, n = _frame(yaw, pitch)
        stations.append(Station(t=t, section=section, origin=origin, u=u, v=v, n=n))

    vent = loft_sections(stations, samples=88, cap_ends=True)
    label = make_label("VENT TRANSITION (TILTED STATIONS)", center=(3.0, 22.0, 22.0), font_size=9.0)
    return [vent, label]
