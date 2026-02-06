from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from impression.mesh import Mesh
from impression.cache import LRUCache
from impression.mesh_quality import MeshQuality, apply_lod


ArrayLike = np.ndarray

_HEIGHTMAP_CACHE = LRUCache(max_size=32)


def _as_scale(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    arr = np.asarray(value, dtype=float).reshape(2)
    return float(arr[0]), float(arr[1])


def _normalize_image_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        gray = arr.astype(float)
        if gray.max() > 1.0:
            gray = gray / 255.0
        mask = np.ones_like(gray, dtype=bool)
        return np.clip(gray, 0.0, 1.0), mask
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise ValueError("Heightmap array must be HxW, HxWx3, or HxWx4.")
    arr = arr.astype(float)
    if arr.max() > 1.0:
        arr = arr / 255.0
    rgb = arr[..., :3]
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if arr.shape[2] == 4:
        alpha = arr[..., 3]
        mask = alpha > 0.0
    else:
        mask = np.ones(gray.shape, dtype=bool)
    return np.clip(gray, 0.0, 1.0), mask


def _load_heightmap(image: str | Path | Image.Image | ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            img = opened.convert("RGBA")
        arr = np.asarray(img, dtype=float)
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gray = gray / 255.0
        mask = alpha > 0
        return np.clip(gray, 0.0, 1.0), mask
    if isinstance(image, Image.Image):
        img = image.convert("RGBA")
        arr = np.asarray(img, dtype=float)
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gray = gray / 255.0
        mask = alpha > 0
        return np.clip(gray, 0.0, 1.0), mask
    if isinstance(image, np.ndarray):
        return _normalize_image_array(image)
    raise TypeError("heightmap expects a file path, PIL image, or numpy array.")


def heightmap(
    image: str | Path | Image.Image | ArrayLike,
    height: float = 1.0,
    xy_scale: float | Sequence[float] = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    alpha_mode: str = "mask",
    quality: MeshQuality | None = None,
) -> Mesh:
    """Create a heightfield mesh from an image.

    alpha_mode:
        - "mask": skip faces that touch fully transparent pixels (holes).
        - "ignore": treat transparent pixels as zero height (no holes).
    """

    cache_key = _heightmap_cache_key(image, height, xy_scale, center, alpha_mode, quality)
    cached = _HEIGHTMAP_CACHE.get(cache_key) if cache_key is not None else None
    if cached is not None:
        return cached.copy()

    heights, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    if heights.size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))

    height = float(height)
    sx, sy = _as_scale(xy_scale)
    center_vec = np.asarray(center, dtype=float).reshape(3)

    rows, cols = heights.shape
    xs = (np.arange(cols, dtype=float) - (cols - 1) / 2.0) * sx + center_vec[0]
    ys = ((rows - 1 - np.arange(rows, dtype=float)) - (rows - 1) / 2.0) * sy + center_vec[1]
    xv, yv = np.meshgrid(xs, ys)

    zv = center_vec[2] + heights * height
    if alpha_mode == "ignore":
        zv = np.where(mask, zv, center_vec[2])
    elif alpha_mode != "mask":
        raise ValueError("alpha_mode must be 'mask' or 'ignore'.")

    vertices = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

    faces: list[list[int]] = []
    if rows >= 2 and cols >= 2:
        for r in range(rows - 1):
            for c in range(cols - 1):
                i0 = r * cols + c
                i1 = r * cols + c + 1
                i2 = (r + 1) * cols + c + 1
                i3 = (r + 1) * cols + c
                if alpha_mode == "mask":
                    if not (mask[r, c] and mask[r, c + 1] and mask[r + 1, c] and mask[r + 1, c + 1]):
                        continue
                faces.append([i0, i1, i2])
                faces.append([i0, i2, i3])

    faces_arr = np.asarray(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int)
    mesh = Mesh(vertices, faces_arr)
    if cache_key is not None:
        _HEIGHTMAP_CACHE.set(cache_key, mesh.copy())
    return mesh


def _vertex_normals(mesh: Mesh) -> np.ndarray:
    verts = mesh.vertices
    faces = mesh.faces
    normals = np.zeros_like(verts)
    if faces.size == 0 or verts.size == 0:
        return normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1)
    nonzero = norms > 0
    normals[nonzero] = normals[nonzero] / norms[nonzero][:, None]
    normals[~nonzero] = np.array([0.0, 0.0, 1.0])
    return normals


def _displace_direction(mesh: Mesh, direction: str | Sequence[float]) -> np.ndarray:
    if isinstance(direction, str):
        axis = direction.lower()
        if axis == "normal":
            return _vertex_normals(mesh)
        if axis == "x":
            return np.tile(np.array([1.0, 0.0, 0.0]), (mesh.n_vertices, 1))
        if axis == "y":
            return np.tile(np.array([0.0, 1.0, 0.0]), (mesh.n_vertices, 1))
        if axis == "z":
            return np.tile(np.array([0.0, 0.0, 1.0]), (mesh.n_vertices, 1))
        raise ValueError("direction must be 'normal', 'x', 'y', 'z', or a vector.")
    vec = np.asarray(direction, dtype=float).reshape(3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("direction vector must be non-zero.")
    vec = vec / norm
    return np.tile(vec, (mesh.n_vertices, 1))


def _sample_heightmap(
    heights: np.ndarray,
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = heights.shape
    x = np.clip(u, 0.0, 1.0) * (cols - 1)
    y = (1.0 - np.clip(v, 0.0, 1.0)) * (rows - 1)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)

    dx = x - x0
    dy = y - y0

    h00 = heights[y0, x0]
    h10 = heights[y0, x1]
    h01 = heights[y1, x0]
    h11 = heights[y1, x1]

    heights_sampled = (
        (1.0 - dx) * (1.0 - dy) * h00
        + dx * (1.0 - dy) * h10
        + (1.0 - dx) * dy * h01
        + dx * dy * h11
    )

    mx = np.clip(np.rint(x).astype(int), 0, cols - 1)
    my = np.clip(np.rint(y).astype(int), 0, rows - 1)
    masked = ~mask[my, mx]
    return heights_sampled, masked


def _mask_faces(mesh: Mesh, masked_vertices: np.ndarray) -> Mesh:
    faces = mesh.faces
    if faces.size == 0:
        return mesh
    keep = ~np.any(masked_vertices[faces], axis=1)
    faces = faces[keep]
    if faces.size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), color=mesh.color)

    used = np.unique(faces)
    remap = np.full(mesh.n_vertices, -1, dtype=int)
    remap[used] = np.arange(len(used))
    new_faces = remap[faces]
    new_vertices = mesh.vertices[used]

    result = Mesh(new_vertices, new_faces, color=mesh.color)
    if mesh.face_colors is not None:
        result.face_colors = mesh.face_colors[keep]
    return result


def displace_heightmap(
    mesh: Mesh,
    image: str | Path | Image.Image | ArrayLike,
    height: float = 1.0,
    projection: str = "planar",
    plane: str = "xy",
    direction: str | Sequence[float] = "normal",
    alpha_mode: str = "ignore",
    bounds: Sequence[float] | None = None,
    quality: MeshQuality | None = None,
) -> Mesh:
    """Displace a mesh using a heightmap with planar projection.

    alpha_mode:
        - "ignore": transparent pixels cause no displacement.
        - "mask": faces touching transparent samples are removed.
    """

    if projection != "planar":
        raise ValueError("Only planar projection is supported in this build.")
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'.")

    heights, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    if heights.size == 0 or mesh.n_vertices == 0:
        return mesh.copy()

    verts = mesh.vertices.copy()
    if plane == "xy":
        u = verts[:, 0]
        v = verts[:, 1]
        if bounds is None:
            xmin, xmax, ymin, ymax, _, _ = mesh.bounds
            bounds = (xmin, xmax, ymin, ymax)
    elif plane == "xz":
        u = verts[:, 0]
        v = verts[:, 2]
        if bounds is None:
            xmin, xmax, _, _, zmin, zmax = mesh.bounds
            bounds = (xmin, xmax, zmin, zmax)
    else:
        u = verts[:, 1]
        v = verts[:, 2]
        if bounds is None:
            _, _, ymin, ymax, zmin, zmax = mesh.bounds
            bounds = (ymin, ymax, zmin, zmax)

    umin, umax, vmin, vmax = np.asarray(bounds, dtype=float).reshape(4)
    if np.isclose(umax, umin) or np.isclose(vmax, vmin):
        raise ValueError("projection bounds are degenerate.")

    u_norm = (u - umin) / (umax - umin)
    v_norm = (v - vmin) / (vmax - vmin)

    sampled, masked = _sample_heightmap(heights, mask, u_norm, v_norm)
    if alpha_mode == "ignore":
        sampled = np.where(masked, 0.0, sampled)
    elif alpha_mode != "mask":
        raise ValueError("alpha_mode must be 'ignore' or 'mask'.")

    direction_vecs = _displace_direction(mesh, direction)
    displaced = verts + direction_vecs * (sampled[:, None] * float(height))

    result = Mesh(displaced, mesh.faces.copy(), color=mesh.color)
    if mesh.face_colors is not None:
        result.face_colors = mesh.face_colors.copy()

    if alpha_mode == "mask":
        result = _mask_faces(result, masked)

    return result


def _heightmap_cache_key(
    image: str | Path | Image.Image | ArrayLike,
    height: float,
    xy_scale: float | Sequence[float],
    center: Sequence[float],
    alpha_mode: str,
    quality: MeshQuality | None,
) -> tuple | None:
    if isinstance(image, (str, Path)):
        path = Path(image)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return (
            str(path),
            float(mtime),
            float(height),
            tuple(np.asarray(xy_scale, dtype=float).ravel()) if not isinstance(xy_scale, (int, float)) else float(xy_scale),
            tuple(np.asarray(center, dtype=float).ravel()),
            alpha_mode,
            quality.lod if quality is not None else None,
        )
    return None


__all__ = ["heightmap", "displace_heightmap"]
