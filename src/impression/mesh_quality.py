from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

MeshLOD = Literal["preview", "final"]


@dataclass(frozen=True)
class MeshQuality:
    """Controls tessellation density and runtime cost."""

    segments_per_turn: int = 48
    circumferential_segments: int | None = None
    max_chord_deviation: float | None = 0.08
    max_triangles: int | None = 400_000
    adaptive_budget: bool = False
    boolean_epsilon: float = 0.0
    lod: MeshLOD = "final"


def apply_lod(quality: MeshQuality) -> MeshQuality:
    if quality.lod == "final":
        return quality
    if quality.lod != "preview":
        raise ValueError("lod must be 'preview' or 'final'.")
    return replace(
        quality,
        segments_per_turn=max(8, int(quality.segments_per_turn * 0.5)),
        circumferential_segments=(
            None
            if quality.circumferential_segments is None
            else max(12, int(quality.circumferential_segments * 0.5))
        ),
    )


def downshift_quality(quality: MeshQuality, predicted_faces: int) -> MeshQuality:
    if quality.max_triangles is None or predicted_faces <= 0:
        return quality
    scale = max(quality.max_triangles / predicted_faces, 0.1)
    new_segments = max(8, int(quality.segments_per_turn * scale))
    if quality.circumferential_segments is not None:
        new_circ = max(12, int(quality.circumferential_segments * scale))
    else:
        new_circ = None
    return replace(quality, segments_per_turn=new_segments, circumferential_segments=new_circ)
