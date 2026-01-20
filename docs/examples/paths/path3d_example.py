"""Path3D example."""

from __future__ import annotations

from impression.modeling import Arc3D, Bezier3D, Line3D, Path3D


def build():
    segments = [
        Line3D(start=(-1.0, 0.0, 0.0), end=(-0.2, 0.5, 0.2)),
        Arc3D(center=(0.4, 0.0, 0.2), radius=0.7, start_angle_deg=180, end_angle_deg=20, normal=(0, 0, 1)),
        Bezier3D(p0=(0.8, 0.2, 0.4), p1=(1.2, 0.8, 0.6), p2=(1.4, -0.2, 0.8), p3=(1.8, 0.3, 1.0)),
    ]
    return Path3D(segments, closed=False).with_color("#9aa6bf")
