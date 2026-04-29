from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from impression.modeling import Section, Station, as_section, loft_sections, make_circle

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


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


def _base_stations() -> list[Station]:
    c0_l = as_section(make_circle(radius=2.7, center=(-6.0, 0.0))).regions[0]
    c0_r = as_section(make_circle(radius=2.7, center=(6.0, 0.0))).regions[0]
    c1_l = as_section(make_circle(radius=2.35, center=(-7.2, 0.0))).regions[0]
    c1_m = as_section(make_circle(radius=2.1, center=(0.0, 0.0))).regions[0]
    c1_r = as_section(make_circle(radius=2.35, center=(7.2, 0.0))).regions[0]
    c2_l = as_section(make_circle(radius=2.7, center=(-6.5, 0.0))).regions[0]
    c2_r = as_section(make_circle(radius=2.7, center=(6.5, 0.0))).regions[0]

    s0 = Section((c0_l, c0_r), color=(0.70, 0.78, 0.88, 1.0))
    s1 = Section((c1_l, c1_m, c1_r), color=(0.70, 0.78, 0.88, 1.0))
    s2 = Section((c2_l, c2_r), color=(0.70, 0.78, 0.88, 1.0))

    defs = [
        (0.0, s0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0),
        (0.5, s1, np.array([1.2, 0.4, 14.0]), 2.5, 1.2),
        (1.0, s2, np.array([2.4, -0.2, 28.0]), 5.0, 2.5),
    ]
    stations: list[Station] = []
    for t, section, origin, yaw, pitch in defs:
        u, v, n = _frame(yaw, pitch)
        stations.append(Station(t=t, section=section, origin=origin, u=u, v=v, n=n))
    return stations


def _with_offset(stations: list[Station], offset: tuple[float, float, float]) -> list[Station]:
    delta = np.asarray(offset, dtype=float).reshape(3)
    shifted: list[Station] = []
    for st in stations:
        shifted.append(
            Station(
                t=st.t,
                section=st.section,
                origin=st.origin + delta,
                u=st.u,
                v=st.v,
                n=st.n,
            )
        )
    return shifted


def build():
    base = _base_stations()
    off_stations = _with_offset(base, (-20.0, 0.0, 0.0))
    global_stations = _with_offset(base, (20.0, 0.0, 0.0))

    off = loft_sections(
        off_stations,
        samples=72,
        cap_ends=True,
        split_merge_mode="resolve",
        fairness_mode="off",
    )
    fair = loft_sections(
        global_stations,
        samples=72,
        cap_ends=True,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.8,
        fairness_iterations=12,
        skeleton_mode="auto",
    )

    labels = [
        make_label("BASELINE: fairness_mode=off", center=(-20.0, 17.0, 29.0), font_size=7.2),
        make_label("GLOBAL FAIRNESS: fairness_mode=global", center=(20.0, 17.0, 29.0), font_size=7.2),
        make_label("Real-world splitter manifold branch planning", center=(0.0, 24.0, 30.0), font_size=8.0),
    ]
    return [off, fair, *labels]
