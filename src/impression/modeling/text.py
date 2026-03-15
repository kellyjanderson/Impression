"""Text modeling built from font outlines and layout.

This module owns font resolution, glyph extraction, and text layout. Generic
planar topology assembly (outer/hole classification and winding policy) is
delegated to ``impression.modeling.topology``.
"""

from __future__ import annotations

from typing import Sequence
from pathlib import Path
import math
import warnings

import numpy as np
from fontTools.ttLib import TTFont, TTLibFileIsCollectionError
from fontTools.pens.recordingPen import RecordingPen

from impression.mesh import Mesh, combine_meshes

from .drawing2d import Path2D, Line2D, Bezier2D
from ._color import _normalize_color, set_mesh_color
from .extrude import linear_extrude
from .topology import Section, sections_from_paths

_WARNED_TEXT_PROFILE_COLOR = False


def make_text(
    content: str,
    depth: float = 0.2,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    font_size: float = 1.0,
    justify: str = "center",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    font: str = "Arial",
    font_path: str | None = None,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Return 3D text built by extruding text profiles."""

    depth = float(depth)
    if depth <= 0:
        raise ValueError("depth must be positive.")

    profiles = text_profiles(
        content=content,
        font_size=font_size,
        justify=justify,
        valign=valign,
        letter_spacing=letter_spacing,
        line_height=line_height,
        font=font,
        font_path=font_path,
    )
    meshes = [linear_extrude(section, height=depth) for section in profiles]
    if not meshes:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))
    mesh = combine_meshes(meshes)

    direction_vec = _normalize_vec(direction)
    base_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    if not np.allclose(direction_vec, base_vec):
        axis = np.cross(base_vec, direction_vec)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-9:
            angle = math.degrees(math.acos(float(np.clip(np.dot(base_vec, direction_vec), -1.0, 1.0))))
            mesh.rotate_vector(axis, angle, point=(0.0, 0.0, 0.0), inplace=True)
        elif np.dot(base_vec, direction_vec) < 0:
            mesh.rotate_vector((1.0, 0.0, 0.0), 180.0, point=(0.0, 0.0, 0.0), inplace=True)

    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def text(
    content: str,
    depth: float = 0.2,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    font_size: float = 1.0,
    justify: str = "center",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    font: str = "Arial",
    font_path: str | None = None,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Alias for make_text."""

    return make_text(
        content=content,
        depth=depth,
        center=center,
        direction=direction,
        font_size=font_size,
        justify=justify,
        valign=valign,
        letter_spacing=letter_spacing,
        line_height=line_height,
        font=font,
        font_path=font_path,
        color=color,
    )


def text_profiles(
    content: str,
    font_size: float = 1.0,
    justify: str = "left",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    font: str = "Arial",
    font_path: str | None = None,
    color: Sequence[float] | str | None = None,
) -> list[Section]:
    """Return topology-native Section objects for the given text string.

    Topology ownership note: glyph path nesting and profile assembly are routed
    through ``sections_from_paths`` so text does not maintain duplicate
    topology logic.
    """

    if not content:
        return []

    font_path = _resolve_font_path(font_path, font)
    try:
        font = TTFont(font_path)
    except TTLibFileIsCollectionError:
        font = TTFont(font_path, fontNumber=0)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap() or {}
    units_per_em = font["head"].unitsPerEm
    scale = font_size / units_per_em
    hmtx = font["hmtx"].metrics
    ascent = getattr(font["hhea"], "ascent", units_per_em)
    descent = getattr(font["hhea"], "descent", 0)
    ascent *= scale
    descent *= scale

    lines = content.splitlines()
    line_paths: list[list[Path2D]] = []
    line_widths: list[float] = []

    for line in lines:
        x_cursor = 0.0
        paths: list[Path2D] = []
        for ch in line:
            glyph_name = cmap.get(ord(ch), ".notdef")
            glyph = glyph_set.get(glyph_name)
            if glyph is None:
                glyph = glyph_set.get(".notdef")
            if glyph is None:
                continue
            paths.extend(_glyph_paths(glyph, scale, (x_cursor, 0.0)))
            advance, _ = hmtx.get(glyph_name, (units_per_em, 0))
            x_cursor += advance * scale
            x_cursor += letter_spacing
        line_paths.append(paths)
        line_widths.append(x_cursor)

    aligned_paths: list[Path2D] = []
    total_height = len(lines) * font_size * line_height
    baseline_offset = 0.0
    if valign == "top":
        baseline_offset = -ascent
    elif valign == "middle":
        baseline_offset = -total_height / 2.0 + ascent
    elif valign == "bottom":
        baseline_offset = -total_height + ascent

    for line_index, paths in enumerate(line_paths):
        line_width = line_widths[line_index]
        if justify == "center":
            x_offset = -line_width / 2.0
        elif justify == "right":
            x_offset = -line_width
        else:
            x_offset = 0.0
        y_offset = -line_index * font_size * line_height + baseline_offset
        for path in paths:
            aligned_paths.append(_translate_path(path, (x_offset, y_offset)))

    sections = sections_from_paths(aligned_paths)
    if color is not None:
        global _WARNED_TEXT_PROFILE_COLOR
        _normalize_color(color)  # keep validation behavior for callers
        if not _WARNED_TEXT_PROFILE_COLOR:
            _WARNED_TEXT_PROFILE_COLOR = True
            warnings.warn(
                "text_profiles(..., color=...) no longer annotates profile colors; "
                "apply color at mesh stage (e.g., make_text color=...).",
                DeprecationWarning,
                stacklevel=2,
            )
    return sections


def text_sections(
    content: str,
    font_size: float = 1.0,
    justify: str = "left",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    font: str = "Arial",
    font_path: str | None = None,
    color: Sequence[float] | str | None = None,
) -> list[Section]:
    """Alias for text_profiles with topology-native naming."""

    return text_profiles(
        content=content,
        font_size=font_size,
        justify=justify,
        valign=valign,
        letter_spacing=letter_spacing,
        line_height=line_height,
        font=font,
        font_path=font_path,
        color=color,
    )


def _resolve_font_path(font_path: str | None, font: str) -> str:
    if font_path:
        path = Path(font_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Font path not found: {font_path}")
        return str(path)
    candidates = []
    font_name = font.lower().replace(" ", "")
    search_dirs = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            candidates.extend(directory.rglob(ext))
    for path in candidates:
        if font_name in path.stem.lower().replace(" ", ""):
            return str(path)
    raise FileNotFoundError(
        f"Could not locate font '{font}'. Provide font_path explicitly."
    )


def _glyph_paths(glyph, scale: float, offset: tuple[float, float]) -> list[Path2D]:
    pen = RecordingPen()
    glyph.draw(pen)
    contours: list[list[object]] = []
    current: list[object] = []
    start_point = None
    last_point = None

    def tx(pt):
        arr = np.asarray(pt, dtype=float).reshape(2)
        return arr * scale + np.asarray(offset, dtype=float)

    for command, pts in pen.value:
        if command == "moveTo":
            if current:
                contours.append(current)
            current = []
            start_point = pts[0]
            last_point = start_point
        elif command == "lineTo":
            for pt in pts:
                p1 = tx(last_point)
                p2 = tx(pt)
                current.append(Line2D(p1, p2))
                last_point = pt
        elif command == "curveTo":
            p1, p2, p3 = pts
            current.append(Bezier2D(tx(last_point), tx(p1), tx(p2), tx(p3)))
            last_point = p3
        elif command == "qCurveTo":
            segments = _quadratic_to_cubic_segments(last_point, pts, start_point)
            for c1, c2, p3 in segments:
                current.append(Bezier2D(tx(last_point), tx(c1), tx(c2), tx(p3)))
                last_point = p3
        elif command in {"closePath", "endPath"}:
            if start_point is not None and last_point is not None and start_point is not last_point:
                if not np.allclose(start_point, last_point):
                    current.append(Line2D(tx(last_point), tx(start_point)))
            if current:
                contours.append(current)
            current = []
            start_point = None
            last_point = None

    if current:
        contours.append(current)

    return [Path2D(segments=list(segments), closed=True) for segments in contours]


def _quadratic_to_cubic_segments(p0, points, start_point) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pts = list(points)
    if pts and pts[-1] is None:
        pts[-1] = start_point
    if not pts:
        return []
    end = pts[-1]
    off = pts[:-1]
    segments = []

    def as_vec(pt):
        return np.asarray(pt, dtype=float).reshape(2)

    if not off:
        end_vec = as_vec(end)
        p0_vec = as_vec(p0)
        c1 = p0_vec + (end_vec - p0_vec) / 3.0
        c2 = p0_vec + 2.0 * (end_vec - p0_vec) / 3.0
        segments.append((c1, c2, end_vec))
        return segments

    prev = as_vec(p0)
    for idx, ctrl in enumerate(off):
        ctrl_vec = as_vec(ctrl)
        if idx < len(off) - 1:
            next_off = as_vec(off[idx + 1])
            on = (ctrl_vec + next_off) / 2.0
        else:
            on = as_vec(end)
        c1 = prev + 2.0 / 3.0 * (ctrl_vec - prev)
        c2 = on + 2.0 / 3.0 * (ctrl_vec - on)
        segments.append((c1, c2, on))
        prev = on
    return segments


def _translate_path(path: Path2D, offset: Sequence[float]) -> Path2D:
    dx, dy = np.asarray(offset, dtype=float).reshape(2)
    segments = []
    for seg in path.segments:
        if isinstance(seg, Line2D):
            segments.append(Line2D(seg.start + [dx, dy], seg.end + [dx, dy]))
        elif isinstance(seg, Bezier2D):
            segments.append(
                Bezier2D(seg.p0 + [dx, dy], seg.p1 + [dx, dy], seg.p2 + [dx, dy], seg.p3 + [dx, dy])
            )
        else:
            segments.append(seg)
    return Path2D(segments=segments, closed=path.closed, color=path.color, metadata=dict(path.metadata))


def _normalize_vec(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("direction must be non-zero.")
    return arr / norm


__all__ = ["make_text", "text", "text_profiles", "text_sections"]
