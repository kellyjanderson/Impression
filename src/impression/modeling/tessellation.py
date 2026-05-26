from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from impression.mesh import Mesh, MeshAnalysis, analyze_mesh, combine_meshes

from ._legacy_mesh_deprecation import warn_mesh_primary_api
from .surface import (
    ImplicitFieldEvaluationDomain,
    ImplicitSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SUPPORTED_SURFACE_PATCH_FAMILIES,
    SubdivisionSurfacePatch,
    SurfaceBody,
    SurfacePatch,
    SurfaceShell,
    TrimLoop,
    _stable_hash,
)
from .topology import triangulate_loops

TessellationIntent = Literal["preview", "export", "analysis"]
TessellationOutput = Literal["mesh"]
TessellationQualityPreset = Literal["preview", "balanced", "fine", "analysis"]
SurfaceOutputClassification = Literal["open", "closed"]
AdapterLossiness = Literal["lossless", "lossy"]
SurfaceFamilySamplingKind = Literal["parametric-loop", "rectangular-grid", "subdivision", "implicit"]

_MAX_IMPLICIT_TESSELLATION_CELLS = 100_000

_QUALITY_PRESETS: dict[TessellationQualityPreset, dict[str, object]] = {
    "preview": {
        "chord_tolerance": 0.5,
        "angular_tolerance_deg": 15.0,
        "max_edge_length": 0.5,
        "weld_shared_edges": True,
        "require_watertight": False,
    },
    "balanced": {
        "chord_tolerance": 0.2,
        "angular_tolerance_deg": 8.0,
        "max_edge_length": 0.25,
        "weld_shared_edges": True,
        "require_watertight": False,
    },
    "fine": {
        "chord_tolerance": 0.05,
        "angular_tolerance_deg": 3.0,
        "max_edge_length": 0.1,
        "weld_shared_edges": True,
        "require_watertight": True,
    },
    "analysis": {
        "chord_tolerance": 0.02,
        "angular_tolerance_deg": 2.0,
        "max_edge_length": 0.05,
        "weld_shared_edges": True,
        "require_watertight": True,
    },
}

_DEFAULT_PRESET_BY_INTENT: dict[TessellationIntent, TessellationQualityPreset] = {
    "preview": "preview",
    "export": "fine",
    "analysis": "analysis",
}


def _validate_positive_or_none(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return scalar


def _normalize_patch_metadata(value: dict[str, object] | None) -> dict[str, object]:
    return {} if value is None else dict(value)


def _point_cache_key(point: np.ndarray, *, decimals: int = 12) -> tuple[object, ...]:
    rounded = tuple(float(value) for value in np.round(np.asarray(point, dtype=float), decimals=decimals))
    return ("point",) + rounded


def _clamp_patch_parameters(
    patch: SurfacePatch,
    u: float,
    v: float,
    *,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if not ((u0 - epsilon) <= u <= (u1 + epsilon) and (v0 - epsilon) <= v <= (v1 + epsilon)):
        raise ValueError(f"Parameters {(u, v)} are outside patch domain {patch.domain}.")
    return (
        float(np.clip(u, u0, u1)),
        float(np.clip(v, v0, v1)),
    )


def _rectangular_domain_loop(patch: SurfacePatch) -> np.ndarray:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    return np.asarray(
        [
            [u0, v0],
            [u1, v0],
            [u1, v1],
            [u0, v1],
        ],
        dtype=float,
    )


def _normalized_patch_loops(patch: SurfacePatch) -> list[np.ndarray]:
    if not patch.trim_loops:
        return [_rectangular_domain_loop(patch)]
    ordered: list[TrimLoop] = []
    outer = patch.outer_trim
    if outer is not None:
        ordered.append(outer.normalized())
    ordered.extend(trim.normalized() for trim in patch.inner_trims)
    return [trim.points_uv.copy() for trim in ordered]


def _patch_uv_mesh_data(patch: SurfacePatch) -> tuple[np.ndarray, np.ndarray]:
    loops = _normalized_patch_loops(patch)
    return triangulate_loops(loops)


def _rectangular_grid_uv_mesh_data(patch: SurfacePatch, *, u_count: int, v_count: int) -> tuple[np.ndarray, np.ndarray]:
    if u_count < 2 or v_count < 2:
        raise ValueError("Rectangular grid tessellation requires at least 2 samples on each axis.")
    us = np.linspace(patch.domain.u_range[0], patch.domain.u_range[1], int(u_count))
    vs = np.linspace(patch.domain.v_range[0], patch.domain.v_range[1], int(v_count))
    uv_vertices = np.asarray([(float(u), float(v)) for v in vs for u in us], dtype=float)
    faces: list[list[int]] = []
    row_width = len(us)
    for v_index in range(len(vs) - 1):
        row0 = v_index * row_width
        row1 = (v_index + 1) * row_width
        for u_index in range(len(us) - 1):
            a = row0 + u_index
            b = a + 1
            c = row1 + u_index
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return uv_vertices, np.asarray(faces, dtype=int)


def _rectangular_grid_counts(request: NormalizedTessellationRequest) -> tuple[int, int]:
    if request.intent == "preview":
        return (16, 12)
    if request.intent == "export":
        return (32, 20)
    return (48, 24)


def _uses_rectangular_grid_tessellation(patch: SurfacePatch) -> bool:
    return isinstance(patch, RevolutionSurfacePatch) and not patch.trim_loops


def _trim_boundary_loop(patch: SurfacePatch, boundary_id: str) -> np.ndarray | None:
    boundary_id = str(boundary_id).strip().lower()
    if boundary_id == "trim:outer":
        outer = patch.outer_trim
        return None if outer is None else outer.normalized().points_uv
    if boundary_id.startswith("trim:inner:"):
        index_text = boundary_id.split(":", 2)[2]
        try:
            inner_index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid trim boundary_id {boundary_id!r}.") from exc
        inner_trims = patch.inner_trims
        if inner_index < 0 or inner_index >= len(inner_trims):
            raise ValueError(f"Trim boundary_id {boundary_id!r} is out of range for the patch.")
        return inner_trims[inner_index].normalized().points_uv
    return None


def _patch_boundary_ids(patch: SurfacePatch) -> tuple[str, ...]:
    if patch.trim_loops:
        return ("trim:outer",) + tuple(f"trim:inner:{index}" for index, _trim in enumerate(patch.inner_trims))
    return ("left", "right", "bottom", "top")


def _patch_requires_shell_grid_tessellation(shell: SurfaceShell, patch_index: int, patch: SurfacePatch) -> bool:
    if _uses_rectangular_grid_tessellation(patch):
        return True
    if not isinstance(patch, RuledSurfacePatch) or patch.trim_loops:
        return False
    for seam in shell.seams:
        boundaries = tuple(
            boundary
            for boundary in seam.boundaries
            if boundary.patch_index == patch_index
        )
        if len(boundaries) == 2:
            boundary_ids = {boundary.boundary_id for boundary in boundaries}
            if boundary_ids in ({"bottom", "top"}, {"left", "right"}):
                return True
        elif len(boundaries) == 1:
            local_boundary = boundaries[0]
            if local_boundary.boundary_id not in {"left", "right"}:
                continue
            peer_boundaries = tuple(
                boundary
                for boundary in seam.boundaries
                if boundary.patch_index != patch_index
            )
            if any(_trim_boundary_loop(shell.patches[boundary.patch_index], boundary.boundary_id) is not None for boundary in peer_boundaries):
                return True
    return False


def _shell_grid_counts_for_patch(
    shell: SurfaceShell,
    patch_index: int,
    patch: SurfacePatch,
    request: NormalizedTessellationRequest,
) -> tuple[int, int] | None:
    if _uses_rectangular_grid_tessellation(patch):
        return _rectangular_grid_counts(request)
    if _patch_requires_shell_grid_tessellation(shell, patch_index, patch) and isinstance(patch, RuledSurfacePatch):
        u_count, _v_count = _rectangular_grid_counts(request)
        curve_count = int(patch.start_curve.shape[0])
        if curve_count >= 2 and np.allclose(patch.start_curve[0], patch.start_curve[-1]):
            curve_count -= 1
        return (u_count, max(2, curve_count))
    return None


def _boundary_samples(patch: SurfacePatch, boundary_id: str, *, sample_count: int = 9) -> np.ndarray:
    boundary_id = str(boundary_id).strip().lower()
    trim_loop = _trim_boundary_loop(patch, boundary_id)
    if trim_loop is not None:
        return np.asarray(
            [
                patch.point_at(*_clamp_patch_parameters(patch, float(u), float(v)))
                for u, v in trim_loop
            ],
            dtype=float,
        )
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if boundary_id == "left":
        parameters = [(u0, v) for v in np.linspace(v0, v1, sample_count)]
    elif boundary_id == "right":
        parameters = [(u1, v) for v in np.linspace(v0, v1, sample_count)]
    elif boundary_id == "bottom":
        parameters = [(u, v0) for u in np.linspace(u0, u1, sample_count)]
    elif boundary_id == "top":
        parameters = [(u, v1) for u in np.linspace(u0, u1, sample_count)]
    else:
        raise ValueError(f"Unsupported boundary_id for boundary sampling: {boundary_id!r}")
    return np.asarray([patch.point_at(*_clamp_patch_parameters(patch, float(u), float(v))) for u, v in parameters], dtype=float)


def _boundary_is_collapsed(patch: SurfacePatch, boundary_id: str, *, tolerance: float = 1e-9) -> bool:
    samples = _boundary_samples(patch, boundary_id)
    reference = samples[0]
    deltas = np.linalg.norm(samples - reference, axis=1)
    return bool(np.all(deltas <= tolerance))


def _rectangular_boundary_indices(
    patch: SurfacePatch,
    uv_vertices: np.ndarray,
    boundary_id: str,
    *,
    epsilon: float = 1e-9,
) -> tuple[int, ...]:
    boundary_id = str(boundary_id).strip().lower()
    trim_loop = _trim_boundary_loop(patch, boundary_id)
    if trim_loop is not None:
        matches: list[int] = []
        used: set[int] = set()
        for point in trim_loop:
            point = np.asarray(point, dtype=float)
            match_index: int | None = None
            for index, uv in enumerate(uv_vertices):
                if index in used:
                    continue
                if float(np.linalg.norm(np.asarray(uv, dtype=float) - point)) <= max(epsilon, 1e-6):
                    match_index = index
                    break
            if match_index is None:
                raise ValueError(f"Trim boundary {boundary_id!r} does not expose enough sampled vertices for seam reuse.")
            used.add(match_index)
            matches.append(match_index)
        if len(matches) < 2:
            raise ValueError(f"Trim boundary {boundary_id!r} does not expose enough sampled vertices for seam reuse.")
        return tuple(matches)
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if boundary_id == "left":
        indices = [index for index, (u, _v) in enumerate(uv_vertices) if abs(float(u) - u0) <= epsilon]
        key = lambda index: float(uv_vertices[index, 1])
    elif boundary_id == "right":
        indices = [index for index, (u, _v) in enumerate(uv_vertices) if abs(float(u) - u1) <= epsilon]
        key = lambda index: float(uv_vertices[index, 1])
    elif boundary_id == "bottom":
        indices = [index for index, (_u, v) in enumerate(uv_vertices) if abs(float(v) - v0) <= epsilon]
        key = lambda index: float(uv_vertices[index, 0])
    elif boundary_id == "top":
        indices = [index for index, (_u, v) in enumerate(uv_vertices) if abs(float(v) - v1) <= epsilon]
        key = lambda index: float(uv_vertices[index, 0])
    else:
        raise ValueError(f"Unsupported boundary_id for seam-aware tessellation: {boundary_id!r}")
    if len(indices) < 2:
        raise ValueError(f"Boundary {boundary_id!r} does not expose enough sampled vertices for seam reuse.")
    return tuple(sorted(indices, key=key))


def _seam_vertex_assignments(
    shell: SurfaceShell,
    world_patches: tuple[SurfacePatch, ...],
    patch_uv_vertices: tuple[np.ndarray, ...],
) -> dict[tuple[int, int], tuple[str, int, np.ndarray]]:
    assignments: dict[tuple[int, int], tuple[str, int, np.ndarray]] = {}
    for seam in shell.seams:
        if seam.is_open:
            continue
        owner_ref, peer_ref = seam.boundaries
        owner_patch = world_patches[owner_ref.patch_index]
        peer_patch = world_patches[peer_ref.patch_index]
        owner_uv = patch_uv_vertices[owner_ref.patch_index]
        peer_uv = patch_uv_vertices[peer_ref.patch_index]
        owner_indices = _rectangular_boundary_indices(owner_patch, owner_uv, owner_ref.boundary_id)
        peer_indices = _rectangular_boundary_indices(peer_patch, peer_uv, peer_ref.boundary_id)
        if len(owner_indices) != len(peer_indices):
            raise ValueError(f"Seam {seam.seam_id!r} boundary sample counts do not match.")

        owner_points = np.asarray(
            [
                owner_patch.point_at(
                    *_clamp_patch_parameters(owner_patch, float(owner_uv[index, 0]), float(owner_uv[index, 1]))
                )
                for index in owner_indices
            ],
            dtype=float,
        )
        peer_points = np.asarray(
            [
                peer_patch.point_at(
                    *_clamp_patch_parameters(peer_patch, float(peer_uv[index, 0]), float(peer_uv[index, 1]))
                )
                for index in peer_indices
            ],
            dtype=float,
        )
        forward_cost = float(np.linalg.norm(owner_points - peer_points, axis=1).sum())
        reversed_cost = float(np.linalg.norm(owner_points - peer_points[::-1], axis=1).sum())
        if reversed_cost < forward_cost:
            peer_indices = tuple(reversed(peer_indices))

        for sample_index, vertex_index in enumerate(owner_indices):
            assignments[(owner_ref.patch_index, vertex_index)] = (seam.seam_id, sample_index, owner_points[sample_index])
        for sample_index, vertex_index in enumerate(peer_indices):
            assignments[(peer_ref.patch_index, vertex_index)] = (seam.seam_id, sample_index, owner_points[sample_index])
    return assignments


def _classify_shell(shell: SurfaceShell) -> SurfaceOutputClassification:
    if not shell.seams:
        return "open"
    expected: dict[tuple[int, str], bool] = {}
    for patch_index, _patch in enumerate(shell.patches):
        for boundary_id in _patch_boundary_ids(shell.patches[patch_index]):
            expected[(patch_index, boundary_id)] = False
    referenced: set[tuple[int, str]] = set()
    for seam in shell.seams:
        if seam.is_open:
            return "open"
        for boundary in seam.boundaries:
            referenced.add((boundary.patch_index, boundary.boundary_id))
    for key in referenced:
        if key in expected:
            expected[key] = True
    for (patch_index, boundary_id), satisfied in list(expected.items()):
        if not satisfied and _boundary_is_collapsed(shell.patches[patch_index], boundary_id):
            expected[(patch_index, boundary_id)] = True
    return "closed" if all(expected.values()) else "open"


def _classify_body(body: SurfaceBody) -> SurfaceOutputClassification:
    return "closed" if all(_classify_shell(shell) == "closed" for shell in body.shells) else "open"


def _patch_mesh(patch: SurfacePatch) -> Mesh:
    uv_vertices, faces = _patch_uv_mesh_data(patch)
    if uv_vertices.size == 0:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), metadata={"surface_family": patch.family})
    vertices = np.asarray([patch.point_at(*_clamp_patch_parameters(patch, float(u), float(v))) for u, v in uv_vertices], dtype=float)
    metadata = {
        "surface_family": patch.family,
        "surface_patch_id": patch.stable_identity,
    }
    return Mesh(vertices=vertices, faces=faces, metadata=metadata)


def _family_adapter_for_patch(patch: SurfacePatch) -> SurfaceFamilyTessellationAdapter:
    adapter = SURFACE_FAMILY_TESSELLATION_ADAPTERS.get(patch.family)
    if adapter is None:
        supported = ", ".join(sorted(SURFACE_FAMILY_TESSELLATION_ADAPTERS))
        raise ValueError(f"Unsupported surface patch family for tessellation: {patch.family!r}. Supported families: {supported}.")
    return adapter


def _mesh_with_family_adapter_metadata(mesh: Mesh, adapter: SurfaceFamilyTessellationAdapter) -> Mesh:
    mesh.metadata.update(
        {
            "tessellation_family_adapter": adapter.canonical_payload(),
            "tessellation_family": adapter.family,
            "tessellation_sampling_kind": adapter.sampling_kind,
            "tessellation_adapter_boundary": "tessellation",
        }
    )
    if adapter.approximation_boundary is not None:
        mesh.metadata["tessellation_approximation_boundary"] = adapter.approximation_boundary
    return mesh


def _rectangular_grid_patch_mesh(patch: SurfacePatch, request: NormalizedTessellationRequest) -> Mesh:
    u_count, v_count = _rectangular_grid_counts(request)
    uv_vertices, faces = _rectangular_grid_uv_mesh_data(patch, u_count=u_count, v_count=v_count)
    vertices: list[np.ndarray] = []
    local_to_global: list[int] = []
    vertex_lookup: dict[tuple[object, ...], int] = {}
    for u, v in uv_vertices:
        world_point = patch.point_at(*_clamp_patch_parameters(patch, float(u), float(v)))
        key = _point_cache_key(world_point)
        global_index = vertex_lookup.get(key)
        if global_index is None:
            global_index = len(vertices)
            vertex_lookup[key] = global_index
            vertices.append(np.asarray(world_point, dtype=float))
        local_to_global.append(global_index)
    remapped_faces = []
    for tri in faces:
        face = [local_to_global[int(tri[0])], local_to_global[int(tri[1])], local_to_global[int(tri[2])]]
        if len(set(face)) == 3:
            remapped_faces.append(face)
    return Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(remapped_faces, dtype=int),
        metadata={"surface_family": patch.family, "surface_patch_id": patch.stable_identity},
    )


def _tessellate_patch_with_family_adapter(
    patch: SurfacePatch,
    request: NormalizedTessellationRequest,
) -> Mesh:
    adapter = _family_adapter_for_patch(patch)
    if adapter.sampling_kind == "implicit":
        if not isinstance(patch, ImplicitSurfacePatch):
            raise ValueError("Implicit tessellation adapter requires an ImplicitSurfacePatch.")
        mesh = _implicit_patch_mesh(patch, request)
    elif adapter.sampling_kind == "subdivision":
        if not isinstance(patch, SubdivisionSurfacePatch):
            raise ValueError("Subdivision tessellation adapter requires a SubdivisionSurfacePatch.")
        mesh = _subdivision_patch_mesh(patch, request)
    elif adapter.sampling_kind == "rectangular-grid":
        mesh = _rectangular_grid_patch_mesh(patch, request)
    else:
        mesh = _patch_mesh(patch)
    return _mesh_with_family_adapter_metadata(mesh, adapter)


def _subdivision_refinement_level(patch: SubdivisionSurfacePatch, request: NormalizedTessellationRequest) -> int:
    preset_minimum = {"preview": 1, "balanced": 1, "fine": 2, "analysis": 3}[request.quality_preset]
    return max(int(patch.subdivision_level), preset_minimum)


def _subdivision_patch_mesh(patch: SubdivisionSurfacePatch, request: NormalizedTessellationRequest) -> Mesh:
    result = patch.refined_cage(levels=_subdivision_refinement_level(patch, request))
    vertices = np.asarray([patch.transform_matrix[:3, :3] @ point + patch.transform_matrix[:3, 3] for point in result.control_points], dtype=float)
    faces: list[list[int]] = []
    for face in result.faces:
        if len(face) < 3:
            continue
        root = int(face[0])
        for index in range(1, len(face) - 1):
            tri = [root, int(face[index]), int(face[index + 1])]
            if len(set(tri)) == 3:
                faces.append(tri)
    return Mesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=int),
        metadata={
            "surface_family": patch.family,
            "surface_patch_id": patch.stable_identity,
            "subdivision_scheme": result.scheme,
            "subdivision_level": result.level,
            "subdivision_approximation": result.metadata["approximation"],
        },
    )


def _implicit_samples_for_request(
    patch: ImplicitSurfacePatch,
    request: NormalizedTessellationRequest,
) -> ImplicitTessellationBoundsDiagnostic:
    xmin, xmax, ymin, ymax, zmin, zmax = patch.bounds
    spans = np.asarray([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    samples = tuple(max(3, int(np.ceil(span / request.max_edge_length)) + 1) for span in spans)
    estimated_cells = int(np.prod([sample - 1 for sample in samples]))
    diagnostic = ImplicitTessellationBoundsDiagnostic(
        bounds=patch.bounds,
        samples=samples,
        estimated_cells=estimated_cells,
    )
    if not diagnostic.within_bounds:
        raise ValueError(
            "Implicit tessellation request exceeds bounded sampling limit: "
            f"{diagnostic.estimated_cells} cells requested, max {diagnostic.max_cells}."
        )
    return diagnostic


def _implicit_patch_mesh(patch: ImplicitSurfacePatch, request: NormalizedTessellationRequest) -> Mesh:
    diagnostic = _implicit_samples_for_request(patch, request)
    domain = ImplicitFieldEvaluationDomain(bounds=patch.bounds, samples=diagnostic.samples)
    values = patch.evaluate_domain(samples=diagnostic.samples)
    xmin, xmax, ymin, ymax, zmin, zmax = patch.bounds
    xs = np.linspace(xmin, xmax, diagnostic.samples[0])
    ys = np.linspace(ymin, ymax, diagnostic.samples[1])
    zs = np.linspace(zmin, zmax, diagnostic.samples[2])
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    active_cells = 0

    def add_quad(corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        start = len(vertices)
        vertices.extend(np.asarray(corner, dtype=float) for corner in corners)
        faces.append([start, start + 1, start + 2])
        faces.append([start, start + 2, start + 3])

    for z_index in range(diagnostic.samples[2] - 1):
        for y_index in range(diagnostic.samples[1] - 1):
            for x_index in range(diagnostic.samples[0] - 1):
                cell_values = values[z_index : z_index + 2, y_index : y_index + 2, x_index : x_index + 2]
                if not (np.min(cell_values) <= 0.0 <= np.max(cell_values)):
                    continue
                active_cells += 1
                x0, x1 = xs[x_index], xs[x_index + 1]
                y0, y1 = ys[y_index], ys[y_index + 1]
                z0, z1 = zs[z_index], zs[z_index + 1]
                center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=float)
                half = np.array([(x1 - x0), (y1 - y0), (z1 - z0)], dtype=float) * 0.45
                gradients = np.gradient(cell_values.astype(float))
                axis = int(np.argmax([float(np.mean(np.abs(gradient))) for gradient in gradients]))
                if axis == 0:
                    add_quad(
                        (
                            center + np.array([-half[0], -half[1], 0.0]),
                            center + np.array([half[0], -half[1], 0.0]),
                            center + np.array([half[0], half[1], 0.0]),
                            center + np.array([-half[0], half[1], 0.0]),
                        )
                    )
                elif axis == 1:
                    add_quad(
                        (
                            center + np.array([-half[0], 0.0, -half[2]]),
                            center + np.array([half[0], 0.0, -half[2]]),
                            center + np.array([half[0], 0.0, half[2]]),
                            center + np.array([-half[0], 0.0, half[2]]),
                        )
                    )
                else:
                    add_quad(
                        (
                            center + np.array([0.0, -half[1], -half[2]]),
                            center + np.array([0.0, half[1], -half[2]]),
                            center + np.array([0.0, half[1], half[2]]),
                            center + np.array([0.0, -half[1], half[2]]),
                        )
                    )

    transformed_vertices = np.asarray(
        [patch.transform_matrix[:3, :3] @ vertex + patch.transform_matrix[:3, 3] for vertex in vertices],
        dtype=float,
    )
    approximation = ImplicitApproximationMetadata(
        method="bounded_sampled_sign_change_quads",
        isovalue=0.0,
        samples=domain.samples,
        active_cells=active_cells,
    )
    return Mesh(
        vertices=transformed_vertices.reshape(-1, 3),
        faces=np.asarray(faces, dtype=int),
        metadata={
            "surface_family": patch.family,
            "surface_patch_id": patch.stable_identity,
            "implicit_bounds_diagnostic": diagnostic.canonical_payload(),
            "implicit_approximation": approximation.canonical_payload(),
            "implicit_approximation_boundary": "tessellation",
        },
    )


@dataclass(frozen=True)
class TessellationRequest:
    """Consumer-facing tessellation request with deterministic defaulting."""

    intent: TessellationIntent = "preview"
    output: TessellationOutput = "mesh"
    quality_preset: TessellationQualityPreset | None = None
    chord_tolerance: float | None = None
    angular_tolerance_deg: float | None = None
    max_edge_length: float | None = None
    weld_shared_edges: bool | None = None
    require_watertight: bool | None = None

    def __post_init__(self) -> None:
        if self.intent not in {"preview", "export", "analysis"}:
            raise ValueError("intent must be 'preview', 'export', or 'analysis'.")
        if self.output != "mesh":
            raise ValueError("Only 'mesh' tessellation output is currently supported.")
        if self.quality_preset is not None and self.quality_preset not in _QUALITY_PRESETS:
            raise ValueError("quality_preset must be one of: preview, balanced, fine, analysis.")
        object.__setattr__(self, "chord_tolerance", _validate_positive_or_none(self.chord_tolerance, name="chord_tolerance"))
        object.__setattr__(
            self,
            "angular_tolerance_deg",
            _validate_positive_or_none(self.angular_tolerance_deg, name="angular_tolerance_deg"),
        )
        object.__setattr__(self, "max_edge_length", _validate_positive_or_none(self.max_edge_length, name="max_edge_length"))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "output": self.output,
            "quality_preset": self.quality_preset,
            "chord_tolerance": self.chord_tolerance,
            "angular_tolerance_deg": self.angular_tolerance_deg,
            "max_edge_length": self.max_edge_length,
            "weld_shared_edges": self.weld_shared_edges,
            "require_watertight": self.require_watertight,
        }


@dataclass(frozen=True)
class NormalizedTessellationRequest:
    """Executor-facing canonical tessellation request."""

    intent: TessellationIntent
    output: TessellationOutput
    quality_preset: TessellationQualityPreset
    chord_tolerance: float
    angular_tolerance_deg: float
    max_edge_length: float
    weld_shared_edges: bool
    require_watertight: bool

    def __post_init__(self) -> None:
        _validate_positive_or_none(self.chord_tolerance, name="chord_tolerance")
        _validate_positive_or_none(self.angular_tolerance_deg, name="angular_tolerance_deg")
        _validate_positive_or_none(self.max_edge_length, name="max_edge_length")

    @property
    def cache_key(self) -> str:
        return _stable_hash(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "output": self.output,
            "quality_preset": self.quality_preset,
            "chord_tolerance": self.chord_tolerance,
            "angular_tolerance_deg": self.angular_tolerance_deg,
            "max_edge_length": self.max_edge_length,
            "weld_shared_edges": self.weld_shared_edges,
            "require_watertight": self.require_watertight,
        }


@dataclass(frozen=True)
class SurfaceTessellationResult:
    mesh: Mesh
    request: NormalizedTessellationRequest
    classification: SurfaceOutputClassification
    body_identity: str
    analysis: MeshAnalysis


@dataclass(frozen=True)
class SurfaceFamilyTessellationAdapter:
    """Family-specific tessellation policy behind the common request contract."""

    family: str
    sampling_kind: SurfaceFamilySamplingKind
    supports_seam_boundaries: bool = True
    approximation_boundary: str | None = None

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not family:
            raise ValueError("SurfaceFamilyTessellationAdapter.family must be non-empty.")
        if self.sampling_kind not in {"parametric-loop", "rectangular-grid", "subdivision", "implicit"}:
            raise ValueError("SurfaceFamilyTessellationAdapter.sampling_kind is not supported.")
        approximation_boundary = None if self.approximation_boundary is None else str(self.approximation_boundary).strip()
        if approximation_boundary == "":
            raise ValueError("SurfaceFamilyTessellationAdapter.approximation_boundary must be non-empty when provided.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "approximation_boundary", approximation_boundary)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "sampling_kind": self.sampling_kind,
            "supports_seam_boundaries": self.supports_seam_boundaries,
            "approximation_boundary": self.approximation_boundary,
        }


@dataclass(frozen=True)
class ImplicitTessellationBoundsDiagnostic:
    """Bounded sampling decision for an implicit patch tessellation request."""

    bounds: tuple[float, float, float, float, float, float]
    samples: tuple[int, int, int]
    estimated_cells: int
    max_cells: int = _MAX_IMPLICIT_TESSELLATION_CELLS

    def __post_init__(self) -> None:
        samples = tuple(int(value) for value in self.samples)
        if len(samples) != 3 or any(value < 2 for value in samples):
            raise ValueError("ImplicitTessellationBoundsDiagnostic.samples must contain three values >= 2.")
        estimated_cells = int(self.estimated_cells)
        max_cells = int(self.max_cells)
        if estimated_cells < 1 or max_cells < 1:
            raise ValueError("Implicit tessellation cell counts must be positive.")
        object.__setattr__(self, "bounds", tuple(float(value) for value in self.bounds))
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "estimated_cells", estimated_cells)
        object.__setattr__(self, "max_cells", max_cells)

    @property
    def within_bounds(self) -> bool:
        return self.estimated_cells <= self.max_cells

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bounds": self.bounds,
            "samples": self.samples,
            "estimated_cells": self.estimated_cells,
            "max_cells": self.max_cells,
            "within_bounds": self.within_bounds,
        }


@dataclass(frozen=True)
class ImplicitApproximationMetadata:
    """Metadata describing an implicit patch mesh approximation."""

    method: str
    isovalue: float
    samples: tuple[int, int, int]
    active_cells: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "isovalue": float(self.isovalue),
            "samples": tuple(int(value) for value in self.samples),
            "active_cells": int(self.active_cells),
            "exact": False,
        }


SURFACE_FAMILY_TESSELLATION_ADAPTERS: dict[str, SurfaceFamilyTessellationAdapter] = {
    "planar": SurfaceFamilyTessellationAdapter("planar", "parametric-loop"),
    "ruled": SurfaceFamilyTessellationAdapter("ruled", "parametric-loop"),
    "revolution": SurfaceFamilyTessellationAdapter("revolution", "rectangular-grid"),
    "bspline": SurfaceFamilyTessellationAdapter("bspline", "parametric-loop"),
    "nurbs": SurfaceFamilyTessellationAdapter("nurbs", "parametric-loop"),
    "sweep": SurfaceFamilyTessellationAdapter("sweep", "parametric-loop"),
    "subdivision": SurfaceFamilyTessellationAdapter(
        "subdivision",
        "subdivision",
        supports_seam_boundaries=False,
        approximation_boundary="tessellation",
    ),
    "implicit": SurfaceFamilyTessellationAdapter(
        "implicit",
        "implicit",
        supports_seam_boundaries=False,
        approximation_boundary="tessellation",
    ),
    "heightmap": SurfaceFamilyTessellationAdapter("heightmap", "rectangular-grid"),
}

_missing_surface_tessellation_adapters = set(SUPPORTED_SURFACE_PATCH_FAMILIES) - set(SURFACE_FAMILY_TESSELLATION_ADAPTERS)
if _missing_surface_tessellation_adapters:
    missing = ", ".join(sorted(_missing_surface_tessellation_adapters))
    raise RuntimeError(f"Missing surface family tessellation adapters for: {missing}")


@dataclass(frozen=True)
class CrossModeDriftReport:
    baseline_intent: TessellationIntent
    comparison_intent: TessellationIntent
    baseline_body_identity: str
    comparison_body_identity: str
    bounds_max_delta: float
    classification_changed: bool
    watertightness_changed: bool

    @property
    def within_default_bounds(self) -> bool:
        return (
            self.baseline_body_identity == self.comparison_body_identity
            and self.bounds_max_delta <= 1e-9
            and not self.classification_changed
            and not self.watertightness_changed
        )


@dataclass(frozen=True)
class SurfaceMeshAdapter:
    """Compatibility adapter from canonical surface bodies to legacy meshes."""

    request: NormalizedTessellationRequest
    visibility: Literal["internal"] = "internal"
    supported_consumers: tuple[str, ...] = ("preview", "export", "analysis")
    sunset_condition: str = "Retire once all canonical consumers accept SurfaceBody directly."
    lossiness: AdapterLossiness = "lossy"

    def __post_init__(self) -> None:
        warn_mesh_primary_api(
            "SurfaceMeshAdapter",
            replacement="SurfaceBody-native consumers",
        )

    def convert(self, body: SurfaceBody) -> SurfaceTessellationResult:
        return tessellate_surface_body(body, self.request)


@dataclass(frozen=True)
class TessellationBoundaryViolationDiagnostic:
    helper_name: str
    received_type: str
    expected_inputs: tuple[str, ...]
    message: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "helper_name": self.helper_name,
            "received_type": self.received_type,
            "expected_inputs": self.expected_inputs,
            "message": self.message,
        }


@dataclass(frozen=True)
class TessellationHelperContract:
    helper_name: str
    owner_module: str = "impression.modeling.tessellation"
    accepted_inputs: tuple[str, ...] = ("SurfaceBody", "SurfacePatch", "SurfaceShell")
    boundary: str = "tessellation"
    consumes_authored_primitive_arguments: bool = False

    def validate_source(self, source: object) -> TessellationBoundaryViolationDiagnostic | None:
        received_type = type(source).__name__
        if isinstance(source, (SurfaceBody, SurfacePatch, SurfaceShell)):
            return None
        return TessellationBoundaryViolationDiagnostic(
            helper_name=self.helper_name,
            received_type=received_type,
            expected_inputs=self.accepted_inputs,
            message=(
                f"{self.helper_name} belongs to the tessellation boundary and accepts "
                "SurfaceBody, SurfacePatch, or SurfaceShell inputs only."
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "helper_name": self.helper_name,
            "owner_module": self.owner_module,
            "accepted_inputs": self.accepted_inputs,
            "boundary": self.boundary,
            "consumes_authored_primitive_arguments": self.consumes_authored_primitive_arguments,
        }


@dataclass(frozen=True)
class SurfaceToMeshAdapterRecord:
    source_type: str
    source_identity: str
    request_cache_key: str
    helper_contract: TessellationHelperContract

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "source_identity": self.source_identity,
            "request_cache_key": self.request_cache_key,
            "helper_contract": self.helper_contract.canonical_payload(),
        }


SURFACE_TO_MESH_HELPER_CONTRACT = TessellationHelperContract("surface-to-mesh-adapter")


def validate_tessellation_helper_boundary_input(
    source: object,
    helper_contract: TessellationHelperContract = SURFACE_TO_MESH_HELPER_CONTRACT,
) -> TessellationBoundaryViolationDiagnostic | None:
    return helper_contract.validate_source(source)


def make_surface_to_mesh_adapter_record(
    source: SurfaceBody | SurfacePatch | SurfaceShell,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> SurfaceToMeshAdapterRecord:
    diagnostic = validate_tessellation_helper_boundary_input(source)
    if diagnostic is not None:
        raise ValueError(diagnostic.message)
    normalized = request if isinstance(request, NormalizedTessellationRequest) else normalize_tessellation_request(request)
    return SurfaceToMeshAdapterRecord(
        source_type=type(source).__name__,
        source_identity=source.stable_identity,
        request_cache_key=normalized.cache_key,
        helper_contract=SURFACE_TO_MESH_HELPER_CONTRACT,
    )


@dataclass(frozen=True)
class SurfaceConsumerRecord:
    body: SurfaceBody
    source_id: str
    order: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "metadata", _normalize_patch_metadata(self.metadata))


@dataclass(frozen=True)
class SurfaceConsumerCollection:
    items: tuple[SurfaceConsumerRecord, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.items, key=lambda item: (item.order, item.source_id, item.body.stable_identity)))
        object.__setattr__(self, "items", ordered)
        object.__setattr__(self, "metadata", _normalize_patch_metadata(self.metadata))

    @property
    def body_identities(self) -> tuple[str, ...]:
        return tuple(item.body.stable_identity for item in self.items)


def normalize_tessellation_request(request: TessellationRequest | None = None) -> NormalizedTessellationRequest:
    if request is None:
        request = TessellationRequest()
    preset_name = request.quality_preset or _DEFAULT_PRESET_BY_INTENT[request.intent]
    preset = _QUALITY_PRESETS[preset_name]
    chord_tolerance = float(request.chord_tolerance if request.chord_tolerance is not None else preset["chord_tolerance"])
    angular_tolerance_deg = float(
        request.angular_tolerance_deg if request.angular_tolerance_deg is not None else preset["angular_tolerance_deg"]
    )
    max_edge_length = float(request.max_edge_length if request.max_edge_length is not None else preset["max_edge_length"])
    weld_shared_edges = bool(request.weld_shared_edges if request.weld_shared_edges is not None else preset["weld_shared_edges"])
    require_watertight = bool(
        request.require_watertight if request.require_watertight is not None else preset["require_watertight"]
    )
    return NormalizedTessellationRequest(
        intent=request.intent,
        output=request.output,
        quality_preset=preset_name,
        chord_tolerance=chord_tolerance,
        angular_tolerance_deg=angular_tolerance_deg,
        max_edge_length=max_edge_length,
        weld_shared_edges=weld_shared_edges,
        require_watertight=require_watertight,
    )


def preview_tessellation_request(**overrides: object) -> TessellationRequest:
    return TessellationRequest(intent="preview", quality_preset="preview", **overrides)


def export_tessellation_request(**overrides: object) -> TessellationRequest:
    return TessellationRequest(intent="export", quality_preset="fine", **overrides)


def analysis_tessellation_request(**overrides: object) -> TessellationRequest:
    return TessellationRequest(intent="analysis", quality_preset="analysis", **overrides)


def tessellate_surface_patch(
    patch: SurfacePatch,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> Mesh:
    normalized = request if isinstance(request, NormalizedTessellationRequest) else normalize_tessellation_request(request)
    mesh = _tessellate_patch_with_family_adapter(patch, normalized)
    mesh.metadata.update(
        {
            "tessellation_request": normalized.canonical_payload(),
            "tessellation_quality_preset": normalized.quality_preset,
        }
    )
    return mesh


def tessellate_surface_shell(
    shell: SurfaceShell,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> Mesh:
    normalized = request if isinstance(request, NormalizedTessellationRequest) else normalize_tessellation_request(request)
    world_patches = shell.iter_patches(world=True)
    if not world_patches:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), metadata={"surface_shell_id": shell.stable_identity})
    patch_family_adapters = tuple(_family_adapter_for_patch(patch) for patch in world_patches)
    if any(not adapter.supports_seam_boundaries for adapter in patch_family_adapters):
        mesh = combine_meshes([tessellate_surface_patch(patch, normalized) for patch in world_patches])
        approximation_boundaries = {
            f"{adapter.family}_approximation_boundary": adapter.approximation_boundary
            for adapter in patch_family_adapters
            if adapter.approximation_boundary is not None
        }
        mesh.metadata.update(
            {
                "surface_shell_id": shell.stable_identity,
                "tessellation_request": normalized.canonical_payload(),
                "surface_shell_classification": _classify_shell(shell),
                "tessellation_family_adapters": [adapter.canonical_payload() for adapter in patch_family_adapters],
                **approximation_boundaries,
            }
        )
        return mesh
    patch_grid_counts = tuple(
        _shell_grid_counts_for_patch(shell, patch_index, patch, normalized)
        for patch_index, patch in enumerate(world_patches)
    )
    patch_uv_faces = tuple(
        _rectangular_grid_uv_mesh_data(patch, u_count=patch_grid_counts[patch_index][0], v_count=patch_grid_counts[patch_index][1])
        if patch_grid_counts[patch_index] is not None
        else _patch_uv_mesh_data(patch)
        for patch_index, patch in enumerate(world_patches)
    )
    patch_uv_vertices = tuple(item[0] for item in patch_uv_faces)
    patch_faces = tuple(item[1] for item in patch_uv_faces)
    seam_assignments = _seam_vertex_assignments(shell, world_patches, patch_uv_vertices) if normalized.weld_shared_edges else {}

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_lookup: dict[tuple[object, ...], int] = {}

    for patch_index, patch in enumerate(world_patches):
        uv_vertices = patch_uv_vertices[patch_index]
        local_to_global: list[int] = []
        for local_index, (u, v) in enumerate(uv_vertices):
            seam_assignment = seam_assignments.get((patch_index, local_index))
            if seam_assignment is not None:
                _seam_id, _sample_index, point = seam_assignment
                world_point = np.asarray(point, dtype=float)
                key = _point_cache_key(world_point)
            else:
                world_point = patch.point_at(*_clamp_patch_parameters(patch, float(u), float(v)))
                key = _point_cache_key(world_point) if patch_grid_counts[patch_index] is not None else ("patch", patch_index, local_index)
            global_index = vertex_lookup.get(key)
            if global_index is None:
                global_index = len(vertices)
                vertex_lookup[key] = global_index
                vertices.append(np.asarray(world_point, dtype=float))
            local_to_global.append(global_index)
        for tri in patch_faces[patch_index]:
            face = [local_to_global[int(tri[0])], local_to_global[int(tri[1])], local_to_global[int(tri[2])]]
            if len(set(face)) == 3:
                faces.append(np.asarray(face, dtype=int))

    mesh = Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
    )
    mesh.metadata.update(
        {
            "surface_shell_id": shell.stable_identity,
            "tessellation_request": normalized.canonical_payload(),
            "surface_shell_classification": _classify_shell(shell),
            "tessellation_family_adapters": [adapter.canonical_payload() for adapter in patch_family_adapters],
        }
    )
    return mesh


def tessellate_surface_body(
    body: SurfaceBody,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> SurfaceTessellationResult:
    normalized = request if isinstance(request, NormalizedTessellationRequest) else normalize_tessellation_request(request)
    shell_meshes = [tessellate_surface_shell(shell, normalized) for shell in body.iter_shells(world=True)]
    if shell_meshes:
        mesh = combine_meshes(shell_meshes)
    else:
        mesh = Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))
    classification = _classify_body(body)
    mesh.metadata.update(
        {
            "surface_body_id": body.stable_identity,
            "surface_output_classification": classification,
            "tessellation_request": normalized.canonical_payload(),
            "adapter_lossiness": "lossy",
        }
    )
    analysis = analyze_mesh(mesh)
    if normalized.require_watertight and classification != "closed":
        raise ValueError("Surface body is not closed-valid under shell seam/boundary truth.")
    if normalized.require_watertight and classification == "closed" and not analysis.is_watertight:
        raise ValueError("Closed surface body tessellation is not watertight under the current request.")
    return SurfaceTessellationResult(
        mesh=mesh,
        request=normalized,
        classification=classification,
        body_identity=body.stable_identity,
        analysis=analysis,
    )


def mesh_from_surface_body(
    body: SurfaceBody,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> Mesh:
    warn_mesh_primary_api(
        "mesh_from_surface_body",
        replacement="tessellate_surface_body(...) or direct SurfaceBody-native consumers",
    )
    return tessellate_surface_body(body, request).mesh


def compare_tessellation_modes(
    body: SurfaceBody,
    baseline: TessellationRequest | NormalizedTessellationRequest | None,
    comparison: TessellationRequest | NormalizedTessellationRequest | None,
) -> CrossModeDriftReport:
    baseline_result = tessellate_surface_body(body, baseline)
    comparison_result = tessellate_surface_body(body, comparison)
    bounds_a = np.asarray(baseline_result.mesh.bounds, dtype=float)
    bounds_b = np.asarray(comparison_result.mesh.bounds, dtype=float)
    return CrossModeDriftReport(
        baseline_intent=baseline_result.request.intent,
        comparison_intent=comparison_result.request.intent,
        baseline_body_identity=baseline_result.body_identity,
        comparison_body_identity=comparison_result.body_identity,
        bounds_max_delta=float(np.max(np.abs(bounds_a - bounds_b))),
        classification_changed=baseline_result.classification != comparison_result.classification,
        watertightness_changed=baseline_result.analysis.is_watertight != comparison_result.analysis.is_watertight,
    )


def make_surface_mesh_adapter(
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
) -> SurfaceMeshAdapter:
    normalized = request if isinstance(request, NormalizedTessellationRequest) else normalize_tessellation_request(request)
    return SurfaceMeshAdapter(request=normalized)


def make_surface_consumer_collection(
    bodies: list[SurfaceBody] | tuple[SurfaceBody, ...],
    *,
    source_prefix: str = "surface",
    metadata: dict[str, object] | None = None,
) -> SurfaceConsumerCollection:
    items = tuple(
        SurfaceConsumerRecord(
            body=body,
            source_id=f"{source_prefix}-{index}",
            order=index,
            metadata={"body_identity": body.stable_identity, **body.consumer_metadata()},
        )
        for index, body in enumerate(bodies)
    )
    return SurfaceConsumerCollection(items=items, metadata=_normalize_patch_metadata(metadata))


__all__ = [
    "AdapterLossiness",
    "CrossModeDriftReport",
    "ImplicitApproximationMetadata",
    "ImplicitTessellationBoundsDiagnostic",
    "NormalizedTessellationRequest",
    "SurfaceConsumerCollection",
    "SurfaceConsumerRecord",
    "SurfaceFamilySamplingKind",
    "SurfaceFamilyTessellationAdapter",
    "SurfaceMeshAdapter",
    "SurfaceOutputClassification",
    "SurfaceToMeshAdapterRecord",
    "SURFACE_TO_MESH_HELPER_CONTRACT",
    "TessellationBoundaryViolationDiagnostic",
    "TessellationHelperContract",
    "SurfaceTessellationResult",
    "SURFACE_FAMILY_TESSELLATION_ADAPTERS",
    "TessellationIntent",
    "TessellationOutput",
    "TessellationQualityPreset",
    "TessellationRequest",
    "analysis_tessellation_request",
    "compare_tessellation_modes",
    "export_tessellation_request",
    "make_surface_to_mesh_adapter_record",
    "make_surface_consumer_collection",
    "make_surface_mesh_adapter",
    "mesh_from_surface_body",
    "normalize_tessellation_request",
    "preview_tessellation_request",
    "tessellate_surface_body",
    "tessellate_surface_patch",
    "tessellate_surface_shell",
    "validate_tessellation_helper_boundary_input",
]
