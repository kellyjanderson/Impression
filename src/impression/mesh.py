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


@dataclass(frozen=True)
class MeshSectionPlane:
    origin: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float).reshape(3)
        normal = np.asarray(self.normal, dtype=float).reshape(3)
        if not np.all(np.isfinite(origin)):
            raise ValueError("MeshSectionPlane.origin must be finite.")
        if not np.all(np.isfinite(normal)):
            raise ValueError("MeshSectionPlane.normal must be finite.")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("MeshSectionPlane.normal must be non-zero.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class MeshSectionPolyline:
    points: np.ndarray
    closed: bool

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        if points.shape[0] < 2:
            raise ValueError("MeshSectionPolyline requires at least two points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("MeshSectionPolyline.points must be finite.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "closed", bool(self.closed))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


@dataclass(frozen=True)
class MeshSectionResult:
    plane: MeshSectionPlane
    polylines: tuple[MeshSectionPolyline, ...]

    @property
    def polyline_count(self) -> int:
        return len(self.polylines)

    @property
    def closed_count(self) -> int:
        return sum(1 for polyline in self.polylines if polyline.closed)


@dataclass(frozen=True)
class MeshRepairReport:
    removed_invalid_faces: int = 0
    removed_degenerate_faces: int = 0
    removed_unreferenced_vertices: int = 0

    @property
    def changed(self) -> bool:
        return (
            self.removed_invalid_faces > 0
            or self.removed_degenerate_faces > 0
            or self.removed_unreferenced_vertices > 0
        )


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


def _plane_signed_distance(points: np.ndarray, plane: MeshSectionPlane) -> np.ndarray:
    return (points - plane.origin) @ plane.normal


def _segment_plane_intersection(
    a: np.ndarray,
    b: np.ndarray,
    da: float,
    db: float,
    *,
    epsilon: float,
) -> np.ndarray | None:
    if abs(da) <= epsilon and abs(db) <= epsilon:
        return None
    if abs(da) <= epsilon:
        return a
    if abs(db) <= epsilon:
        return b
    if da * db > 0.0:
        return None
    denom = da - db
    if abs(denom) <= epsilon:
        return None
    t = da / denom
    return a + ((b - a) * t)


def _dedupe_points(points: list[np.ndarray], *, epsilon: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= epsilon for existing in unique):
            unique.append(point)
    return unique


def _triangle_plane_segment(
    triangle: np.ndarray,
    plane: MeshSectionPlane,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    distances = _plane_signed_distance(triangle, plane)
    candidates: list[np.ndarray] = []
    for edge in ((0, 1), (1, 2), (2, 0)):
        point = _segment_plane_intersection(
            triangle[edge[0]],
            triangle[edge[1]],
            float(distances[edge[0]]),
            float(distances[edge[1]]),
            epsilon=epsilon,
        )
        if point is not None:
            candidates.append(point)
    unique = _dedupe_points(candidates, epsilon=epsilon)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        # Coplanar or numerically messy triangles are intentionally skipped in
        # this first retained analysis tool.
        return None
    return (unique[0], unique[1])


def _point_key(point: np.ndarray, *, epsilon: float) -> tuple[int, int, int]:
    scale = 1.0 / epsilon
    return (
        int(round(float(point[0]) * scale)),
        int(round(float(point[1]) * scale)),
        int(round(float(point[2]) * scale)),
    )


def _stitch_section_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    epsilon: float,
) -> tuple[MeshSectionPolyline, ...]:
    if not segments:
        return ()

    key_to_index: dict[tuple[int, int, int], int] = {}
    nodes: list[np.ndarray] = []
    adjacency: dict[int, list[int]] = {}

    def _node_index(point: np.ndarray) -> int:
        key = _point_key(point, epsilon=epsilon)
        if key not in key_to_index:
            key_to_index[key] = len(nodes)
            nodes.append(point)
        return key_to_index[key]

    edges: list[tuple[int, int]] = []
    for start, end in segments:
        a = _node_index(start)
        b = _node_index(end)
        if a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in edges:
            continue
        edges.append(edge)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited_edges: set[tuple[int, int]] = set()
    polylines: list[MeshSectionPolyline] = []

    def _walk(start_node: int, *, closed_hint: bool) -> MeshSectionPolyline:
        order = [start_node]
        prev = None
        current = start_node
        while True:
            neighbors = adjacency.get(current, [])
            next_node = None
            for neighbor in neighbors:
                edge = (current, neighbor) if current < neighbor else (neighbor, current)
                if edge in visited_edges:
                    continue
                next_node = neighbor
                visited_edges.add(edge)
                break
            if next_node is None:
                break
            order.append(next_node)
            prev, current = current, next_node
            if current == start_node:
                break
        points = np.asarray([nodes[index] for index in order], dtype=float)
        closed = bool(closed_hint and len(order) > 2 and order[0] == order[-1])
        if closed:
            points = points[:-1]
        return MeshSectionPolyline(points=points, closed=closed)

    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    for node in endpoints:
        if all(((node, neighbor) if node < neighbor else (neighbor, node)) in visited_edges for neighbor in adjacency[node]):
            continue
        polylines.append(_walk(node, closed_hint=False))

    for a, b in edges:
        edge = (a, b) if a < b else (b, a)
        if edge in visited_edges:
            continue
        visited_edges.add(edge)
        cycle = [a, b]
        prev = a
        current = b
        while True:
            neighbors = adjacency[current]
            candidates = [neighbor for neighbor in neighbors if neighbor != prev]
            if not candidates:
                break
            next_node = candidates[0]
            edge_key = (current, next_node) if current < next_node else (next_node, current)
            if edge_key in visited_edges:
                if next_node == a:
                    cycle.append(next_node)
                break
            visited_edges.add(edge_key)
            cycle.append(next_node)
            prev, current = current, next_node
        points = np.asarray([nodes[index] for index in cycle], dtype=float)
        closed = bool(len(cycle) > 3 and cycle[0] == cycle[-1])
        if closed:
            points = points[:-1]
        polylines.append(MeshSectionPolyline(points=points, closed=closed))

    return tuple(polylines)


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


def section_mesh_with_plane(
    mesh: Mesh,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    epsilon: float = 1e-9,
    stitch_epsilon: float = 1e-6,
) -> MeshSectionResult:
    """Intersect a triangle mesh with a plane for analysis and debugging.

    This is an explicit mesh-analysis tool. It does not mutate the input mesh
    and it is not canonical modeling truth.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if stitch_epsilon <= 0.0:
        raise ValueError("stitch_epsilon must be positive.")
    plane = MeshSectionPlane(origin=np.asarray(origin, dtype=float), normal=np.asarray(normal, dtype=float))
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be triangular for plane sectioning.")
    if mesh.n_faces == 0:
        return MeshSectionResult(plane=plane, polylines=())

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for face in mesh.faces:
        triangle = mesh.vertices[np.asarray(face, dtype=int)]
        segment = _triangle_plane_segment(triangle, plane, epsilon=epsilon)
        if segment is not None:
            segments.append(segment)
    polylines = _stitch_section_segments(segments, epsilon=stitch_epsilon)
    return MeshSectionResult(plane=plane, polylines=polylines)


def repair_mesh(
    mesh: Mesh,
    *,
    remove_invalid_faces: bool = True,
    remove_degenerate_faces: bool = True,
    remove_unreferenced_vertices: bool = True,
    area_epsilon: float = 1e-12,
) -> tuple[Mesh, MeshRepairReport]:
    """Apply bounded explicit mesh cleanup for downstream repair workflows."""

    if area_epsilon <= 0.0:
        raise ValueError("area_epsilon must be positive.")

    repaired = mesh.copy()
    vertices = repaired.vertices
    faces = repaired.faces
    face_colors = repaired.face_colors
    removed_invalid_faces = 0
    removed_degenerate_faces = 0
    removed_unreferenced_vertices = 0

    if faces.size:
        valid_face_mask = np.ones(faces.shape[0], dtype=bool)
        if remove_invalid_faces:
            in_range = (faces >= 0) & (faces < repaired.n_vertices)
            finite_vertices = np.all(np.isfinite(vertices), axis=1)
            referenced_finite = finite_vertices[faces].all(axis=1)
            valid_face_mask &= in_range.all(axis=1) & referenced_finite
            removed_invalid_faces = int(np.count_nonzero(~valid_face_mask))

        if remove_degenerate_faces:
            candidate_faces = faces[valid_face_mask]
            if candidate_faces.size:
                tri = vertices[candidate_faces]
                cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                areas = np.linalg.norm(cross, axis=1) * 0.5
                degenerate_mask = areas <= area_epsilon
                removed_degenerate_faces = int(np.count_nonzero(degenerate_mask))
                if removed_degenerate_faces:
                    indices = np.flatnonzero(valid_face_mask)
                    valid_face_mask[indices[degenerate_mask]] = False

        faces = faces[valid_face_mask]
        if face_colors is not None and face_colors.shape[0] == repaired.faces.shape[0]:
            face_colors = face_colors[valid_face_mask]

    if remove_unreferenced_vertices and faces.size:
        referenced = np.unique(faces.reshape(-1))
        keep_vertex_mask = np.zeros(repaired.n_vertices, dtype=bool)
        keep_vertex_mask[referenced] = True
        if np.any(~keep_vertex_mask):
            removed_unreferenced_vertices = int(np.count_nonzero(~keep_vertex_mask))
            remap = np.full(repaired.n_vertices, -1, dtype=int)
            remap[keep_vertex_mask] = np.arange(int(np.count_nonzero(keep_vertex_mask)), dtype=int)
            vertices = vertices[keep_vertex_mask]
            faces = remap[faces]
        else:
            removed_unreferenced_vertices = 0

    repaired.vertices = vertices
    repaired.faces = faces
    repaired.face_colors = face_colors
    analyze_mesh(repaired, area_epsilon=area_epsilon)
    return (
        repaired,
        MeshRepairReport(
            removed_invalid_faces=removed_invalid_faces,
            removed_degenerate_faces=removed_degenerate_faces,
            removed_unreferenced_vertices=removed_unreferenced_vertices,
        ),
    )


def mesh_to_pyvista(mesh: Mesh):
    import pyvista as pv

    if mesh.n_faces == 0:
        return pv.PolyData(mesh.vertices, deep=True)
    faces = np.hstack([np.array([3, *tri], dtype=np.int64) for tri in mesh.faces])
    poly = pv.PolyData(mesh.vertices, faces, deep=True)
    return poly
