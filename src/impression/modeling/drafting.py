from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from impression.mesh import Mesh, combine_meshes, triangulate_faces

from ._color import set_mesh_color
from ._legacy_mesh_deprecation import warn_mesh_primary_api
from .primitives import _orient_mesh, _normalize
from .text import make_text

Axis = Literal["x", "y", "z"]
Backend = Literal["mesh", "surface"]

if TYPE_CHECKING:
    from .surface import SurfaceBody
    from .tessellation import SurfaceConsumerCollection


def _surface_transform_from_direction(
    origin: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    origin_vec = np.asarray(origin, dtype=float).reshape(3)
    z_axis = _normalize(direction)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = np.cross(reference, z_axis)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    transform = np.eye(4, dtype=float)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin_vec
    return transform


def _combine_surface_bodies(
    bodies: Sequence["SurfaceBody"],
    *,
    metadata: dict[str, object] | None = None,
) -> "SurfaceBody":
    from .surface import make_surface_body

    shells = tuple(shell for body in bodies for shell in body.iter_shells(world=True))
    if not shells:
        raise ValueError("Expected at least one surface body to combine.")
    return make_surface_body(shells, metadata=metadata)


def make_line(
    start: Sequence[float],
    end: Sequence[float],
    thickness: float = 0.02,
    color: Sequence[float] | str | None = None,
    backend: Backend = "mesh",
) -> Mesh | SurfaceBody:
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        from ._surface_primitives import make_surface_box

        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("Line requires two distinct points.")
        midpoint = (np.asarray(start, dtype=float) + np.asarray(end, dtype=float)) / 2.0
        metadata = {"consumer": {"color": color}} if color is not None else None
        return make_surface_box(
            size=(thickness, thickness, float(length)),
            metadata=metadata,
        ).with_transform(_surface_transform_from_direction(midpoint, direction))

    warn_mesh_primary_api(
        "make_line",
        replacement="surface-native drafting annotations once that path is introduced",
    )
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Line requires two distinct points.")

    half = thickness / 2.0
    base = np.array(
        [
            (-half, -half, 0),
            (half, -half, 0),
            (half, half, 0),
            (-half, half, 0),
        ]
    )
    top = base + np.array((0, 0, length))
    points = np.vstack([base, top])
    faces = triangulate_faces([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ])
    mesh = Mesh(points, faces)
    mesh = _orient_mesh(mesh, _normalize(direction))
    mesh.translate(start, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_plane(
    size: Sequence[float] = (1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
    backend: Backend = "mesh",
) -> Mesh | SurfaceBody:
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        from .surface import PlanarSurfacePatch, make_surface_body, make_surface_shell

        sx, sy = np.asarray(size, dtype=float).reshape(2)
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("size components must both be positive.")
        metadata = {"consumer": {"color": color}} if color is not None else None
        patch = PlanarSurfacePatch(
            family="planar",
            origin=(-sx / 2.0, -sy / 2.0, 0.0),
            u_axis=(sx, 0.0, 0.0),
            v_axis=(0.0, sy, 0.0),
            metadata={"kernel": {"producer": "drafting", "kind": "plane"}},
        )
        body = make_surface_body(
            (make_surface_shell((patch,), connected=False, metadata={"kernel": {"producer": "drafting", "kind": "plane"}}),),
            metadata=metadata,
        )
        return body.with_transform(_surface_transform_from_direction(center, normal))

    warn_mesh_primary_api(
        "make_plane",
        replacement="PlanarSurfacePatch or SurfaceBody-native annotation geometry",
    )
    sx, sy = size
    half_x, half_y = sx / 2.0, sy / 2.0
    points = np.array(
        [
            (-half_x, -half_y, 0.0),
            (half_x, -half_y, 0.0),
            (half_x, half_y, 0.0),
            (-half_x, half_y, 0.0),
        ]
    )
    faces = triangulate_faces([[0, 1, 2, 3]])
    mesh = Mesh(points, faces)
    mesh = _orient_mesh(mesh, normal)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_arrow(
    start: Sequence[float],
    end: Sequence[float],
    shaft_diameter: float = 0.04,
    head_length: float = 0.15,
    head_diameter: float = 0.12,
    color: Sequence[float] | str | None = None,
    backend: Backend = "mesh",
) -> Mesh | SurfaceBody:
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        from ._surface_primitives import make_surface_cone

        start_vec = np.asarray(start, dtype=float)
        end_vec = np.asarray(end, dtype=float)
        direction = end_vec - start_vec
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("Arrow requires distinct start/end points.")
        head_length = min(head_length, length * 0.5)
        shaft_end = start_vec + direction * ((length - head_length) / length)
        shaft = make_line(
            start_vec,
            shaft_end,
            thickness=shaft_diameter,
            color=color,
            backend="surface",
        )
        head_center = shaft_end + (direction / length) * (head_length / 2.0)
        head = make_surface_cone(
            bottom_diameter=head_diameter,
            top_diameter=0.0,
            height=head_length,
            center=head_center,
            direction=direction,
            metadata={"consumer": {"color": color}} if color is not None else None,
        )
        return _combine_surface_bodies(
            [shaft, head],
            metadata={"consumer": {"color": color}} if color is not None else None,
        )

    warn_mesh_primary_api(
        "make_arrow",
        replacement="surface-native drafting annotations once that path is introduced",
    )
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Arrow requires distinct start/end points.")
    head_length = min(head_length, length * 0.5)
    shaft_length = length - head_length

    shaft = make_line(start, start + direction * (shaft_length / length), thickness=shaft_diameter, color=color)
    head_height = head_length
    base = np.array(
        [
            (-head_diameter / 2.0, -head_diameter / 2.0, 0),
            (head_diameter / 2.0, -head_diameter / 2.0, 0),
            (head_diameter / 2.0, head_diameter / 2.0, 0),
            (-head_diameter / 2.0, head_diameter / 2.0, 0),
            (0, 0, head_height),
        ]
    )
    faces = triangulate_faces(
        [
            [0, 1, 2, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
    )
    head = Mesh(base, faces)
    head = _orient_mesh(head, direction / length)
    head.translate(start + direction * (shaft_length / length), inplace=True)
    if color is not None:
        set_mesh_color(head, color)
    return combine_meshes([shaft, head])


def make_dimension(
    start: Sequence[float],
    end: Sequence[float],
    offset: float = 0.1,
    text: str | None = None,
    color: Sequence[float] | str | None = None,
    font: str = "Arial",
    font_path: str | None = None,
    backend: Backend = "mesh",
) -> list[Mesh] | SurfaceConsumerCollection:
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        from .tessellation import make_surface_consumer_collection

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("Dimension requires distinct points.")
        norm_dir = direction / length
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, norm_dir)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(up, norm_dir)
        right /= np.linalg.norm(right)
        offset_vec = right * offset
        arrow_start = start + offset_vec
        arrow_end = end + offset_vec

        bodies = [
            make_arrow(
                arrow_start,
                arrow_end,
                color=color,
                backend="surface",
            )
        ]
        if text:
            label_up = np.cross(right, norm_dir)
            up_norm = np.linalg.norm(label_up)
            if up_norm < 1e-9:
                label_up = up
            else:
                label_up = label_up / up_norm
            label_depth = max(length * 0.015, 0.02)
            label_size = max(length * 0.15, 0.08)
            label_gap = max(abs(offset) * 0.25, label_depth * 2.0)
            label_center = (arrow_start + arrow_end) / 2.0 + right * label_gap
            try:
                label = make_text(
                    text,
                    depth=label_depth,
                    center=tuple(label_center),
                    direction=tuple(right),
                    font_size=label_size,
                    justify="center",
                    valign="middle",
                    font=font,
                    font_path=font_path,
                    color=color,
                    backend="surface",
                )
                bodies.append(label)
            except FileNotFoundError:
                pass
        return make_surface_consumer_collection(
            bodies,
            source_prefix="drafting-dimension",
            metadata={"producer": "drafting", "kind": "dimension"},
        )

    warn_mesh_primary_api(
        "make_dimension",
        replacement="surface-native drafting annotations once that path is introduced",
    )
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Dimension requires distinct points.")
    norm_dir = direction / length
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, norm_dir)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, norm_dir)
    right /= np.linalg.norm(right)
    offset_vec = right * offset
    arrow_start = start + offset_vec
    arrow_end = end + offset_vec

    meshes = [make_arrow(arrow_start, arrow_end, color=color)]
    if text:
        label_up = np.cross(right, norm_dir)
        up_norm = np.linalg.norm(label_up)
        if up_norm < 1e-9:
            label_up = up
        else:
            label_up = label_up / up_norm

        label_depth = max(length * 0.015, 0.02)
        label_size = max(length * 0.15, 0.08)
        label_gap = max(abs(offset) * 0.25, label_depth * 2.0)
        label_center = (arrow_start + arrow_end) / 2.0 + right * label_gap

        try:
            label = make_text(
                text,
                depth=label_depth,
                center=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 1.0),
                font_size=label_size,
                justify="center",
                valign="middle",
                font=font,
                font_path=font_path,
                color=color,
            )
        except FileNotFoundError:
            return meshes

        # Re-center label thickness so it straddles the local annotation plane.
        zmin, zmax = label.bounds[4], label.bounds[5]
        label.translate((0.0, 0.0, -0.5 * (zmin + zmax)), inplace=True)

        transform = np.eye(4, dtype=float)
        transform[:3, 0] = norm_dir
        transform[:3, 1] = label_up
        transform[:3, 2] = right
        transform[:3, 3] = label_center
        label.transform(transform, inplace=True)
        meshes.append(label)

    return meshes
