from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class MeshAnalysis:
    n_vertices: int
    n_faces: int
    degenerate_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    invalid_vertices: int

    @property
    def is_manifold(self) -> bool:
        return self.nonmanifold_edges == 0

    @property
    def is_watertight(self) -> bool:
        return self.boundary_edges == 0 and self.is_manifold

    @property
    def has_degenerate_faces(self) -> bool:
        return self.degenerate_faces > 0

    @property
    def has_invalid_vertices(self) -> bool:
        return self.invalid_vertices > 0

    def issues(self) -> list[str]:
        issues: list[str] = []
        if self.has_invalid_vertices:
            issues.append(f"{self.invalid_vertices} invalid vertices (NaN/inf)")
        if self.has_degenerate_faces:
            issues.append(f"{self.degenerate_faces} degenerate faces")
        if self.boundary_edges > 0:
            issues.append(f"{self.boundary_edges} boundary edges (not watertight)")
        if self.nonmanifold_edges > 0:
            issues.append(f"{self.nonmanifold_edges} non-manifold edges")
        return issues


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float, float] | None = None
    face_colors: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    analysis: MeshAnalysis | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3).copy()
        self.faces = np.asarray(self.faces, dtype=int).reshape(-1, 3).copy()
        if self.color is not None and len(self.color) == 3:
            self.color = (self.color[0], self.color[1], self.color[2], 1.0)
        if self.face_colors is not None:
            self.face_colors = np.asarray(self.face_colors, dtype=float)

    def copy(self) -> "Mesh":
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            color=self.color,
            face_colors=None if self.face_colors is None else self.face_colors.copy(),
            metadata=dict(self.metadata),
            analysis=self.analysis,
        )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        if self.n_vertices == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))

    def transform(self, matrix: np.ndarray, inplace: bool = True) -> "Mesh":
        verts = np.hstack([self.vertices, np.ones((self.n_vertices, 1), dtype=float)])
        transformed = (matrix @ verts.T).T[:, :3]
        if inplace:
            self.vertices = transformed
            return self
        mesh = self.copy()
        mesh.vertices = transformed
        return mesh

    def translate(self, offset: Sequence[float], inplace: bool = True) -> "Mesh":
        vec = np.asarray(offset, dtype=float).reshape(3)
        if inplace:
            self.vertices = self.vertices + vec
            return self
        mesh = self.copy()
        mesh.vertices = mesh.vertices + vec
        return mesh

    def rotate_vector(
        self,
        axis: Sequence[float],
        angle_deg: float,
        point: Sequence[float] = (0.0, 0.0, 0.0),
        inplace: bool = True,
    ) -> "Mesh":
        axis_vec = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis_vec)
        if norm == 0:
            raise ValueError("Rotation axis must be non-zero.")
        axis_vec = axis_vec / norm
        angle_rad = np.deg2rad(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        x, y, z = axis_vec
        C = 1.0 - c
        rot = np.array(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ],
            dtype=float,
        )
        origin = np.asarray(point, dtype=float).reshape(3)
        shifted = self.vertices - origin
        rotated = (rot @ shifted.T).T + origin
        if inplace:
            self.vertices = rotated
            return self
        mesh = self.copy()
        mesh.vertices = rotated
        return mesh


@dataclass
class Polyline:
    points: np.ndarray
    closed: bool = False
    color: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3).copy()
        if self.color is not None and len(self.color) == 3:
            self.color = (self.color[0], self.color[1], self.color[2], 1.0)


def triangulate_faces(face_list: Iterable[Sequence[int]]) -> np.ndarray:
    triangles: list[list[int]] = []
    for face in face_list:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face) - 1):
            triangles.append([v0, face[i], face[i + 1]])
    if not triangles:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(triangles, dtype=int)


def combine_meshes(meshes: Iterable[Mesh]) -> Mesh:
    meshes_list = list(meshes)
    if not meshes_list:
        raise ValueError("combine_meshes requires at least one mesh.")

    vertices = []
    faces = []
    face_colors = []
    color = None
    offset = 0
    for mesh in meshes_list:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += mesh.n_vertices
        if mesh.face_colors is not None:
            face_colors.append(mesh.face_colors)
        elif mesh.n_faces:
            face_colors.append(np.tile(np.array(mesh.color or (0.8, 0.8, 0.8, 1.0)), (mesh.n_faces, 1)))
        if color is None and mesh.color is not None:
            color = mesh.color

    combined = Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces), color=color)
    if face_colors:
        combined.face_colors = np.vstack(face_colors)
    return combined


def analyze_mesh(mesh: Mesh, area_epsilon: float = 1e-12) -> MeshAnalysis:
    verts = mesh.vertices
    faces = mesh.faces
    invalid_vertices = int(np.count_nonzero(~np.isfinite(verts)))

    degenerate_faces = 0
    if faces.size > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) * 0.5
        degenerate_faces = int(np.count_nonzero(areas <= area_epsilon))

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in faces:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)

    analysis = MeshAnalysis(
        n_vertices=mesh.n_vertices,
        n_faces=mesh.n_faces,
        degenerate_faces=degenerate_faces,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        invalid_vertices=invalid_vertices,
    )
    mesh.analysis = analysis
    return analysis


def mesh_to_pyvista(mesh: Mesh):
    import pyvista as pv

    if mesh.n_faces == 0:
        return pv.PolyData(mesh.vertices, deep=True)
    faces = np.hstack([np.array([3, *tri], dtype=np.int64) for tri in mesh.faces])
    poly = pv.PolyData(mesh.vertices, faces, deep=True)
    return poly
