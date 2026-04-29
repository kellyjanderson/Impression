"""Rounded half-pipe modeled with build123d (full CAD fillets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from build123d import (
    Align,
    Axis,
    Box,
    BuildLine,
    BuildSketch,
    Mode,
    Location,
    Plane,
    Polyline,
    Vector,
    fillet,
    extrude,
    make_face,
)

from impression.cad import shape_to_polydata


@dataclass(slots=True)
class HalfPipeConfig:
    radius: float = 10.0
    wall: float = 0.75
    length: float = 28.0
    deck_length: float = 4.0
    deck_height: float = 2.0
    lip_fillet: float = 0.35
    deck_fillet: float = 0.5
    end_fillet: float = 1.0
    tessellation: float = 0.2
    arc_samples: int = 96


def _arc_points(
    radius: float,
    center_z: float,
    start_angle: float,
    end_angle: float,
    samples: int,
) -> np.ndarray:
    theta = np.linspace(start_angle, end_angle, samples)
    x = radius * np.cos(theta)
    z = center_z + radius * np.sin(theta)
    return np.column_stack((x, z))


def _ring_sketch(cfg: HalfPipeConfig):
    outer = _arc_points(cfg.radius, cfg.radius, 0.0, np.pi, cfg.arc_samples)
    inner_radius = cfg.radius - cfg.wall
    if inner_radius <= 0:
        raise ValueError("Wall thickness must be smaller than the radius.")
    inner = _arc_points(inner_radius, cfg.radius, np.pi, 0.0, cfg.arc_samples)

    with BuildSketch(Plane.XZ) as sketch:
        with BuildLine():
            Polyline(*(tuple(point) for point in outer), close=True)
        with BuildLine(mode=Mode.SUBTRACT):
            Polyline(*(tuple(point) for point in inner), close=True)
        make_face()
    return sketch.sketch


def _build_deck(cfg: HalfPipeConfig):
    deck = Box(
        length=cfg.deck_length,
        width=cfg.length,
        height=cfg.deck_height,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    )
    return deck.move(Location(Vector(cfg.radius, 0.0, cfg.radius)))


def _edges_parallel_to(edges: Iterable, axis: Axis, tol: float = 1e-4) -> list:
    axis_vec = Vector(axis.direction).normalized()
    selected: List = []
    for edge in edges:
        try:
            tangent = edge.tangent_at(0.5).normalized()
        except Exception:
            continue
        if abs(abs(tangent.dot(axis_vec)) - 1.0) < tol:
            selected.append(edge)
    return selected


def _edges_near_y(edges: Iterable, target: float, tol: float = 1e-3) -> list:
    selected: List = []
    for edge in edges:
        try:
            center = edge.center()
        except Exception:
            continue
        if abs(center.Y - target) < tol:
            selected.append(edge)
    return selected


def build(config: HalfPipeConfig = HalfPipeConfig()):
    """Build a rounded half-pipe mesh using build123d and convert to PyVista."""

    profile = _ring_sketch(config)
    shell = extrude(profile, amount=config.length, dir=Axis.Y.direction, both=True).solid()
    deck = _build_deck(config)
    solid = shell.fuse(deck)

    if config.lip_fillet > 0:
        shell_edges = [
            edge
            for edge in _edges_parallel_to(solid.edges(), Axis.Y)
            if edge.center().X <= config.radius + 1e-3
        ]
        if shell_edges:
            solid = fillet(shell_edges, config.lip_fillet)

    if config.deck_fillet > 0:
        deck_edges = [
            edge
            for edge in _edges_parallel_to(solid.edges(), Axis.Y)
            if edge.center().X > config.radius + 1e-3
        ]
        if deck_edges:
            solid = fillet(deck_edges, config.deck_fillet)

    if config.end_fillet > 0:
        end_y = config.length / 2.0
        end_edges = _edges_near_y(solid.edges(), end_y) + _edges_near_y(
            solid.edges(), -end_y
        )
        if end_edges:
            solid = fillet(end_edges, config.end_fillet)

    mesh = shape_to_polydata(solid, tolerance=config.tessellation)
    return [mesh]
