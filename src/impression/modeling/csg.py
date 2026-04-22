from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping, Sequence, Union

import warnings

import numpy as np

from impression.mesh import Mesh, analyze_mesh
from impression.modeling.group import MeshGroup

from ._color import get_mesh_color, set_mesh_color
from ._legacy_mesh_deprecation import warn_mesh_primary_api
from .surface import (
    SurfaceBoundaryRef,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    SurfaceBody,
    SurfaceSeam,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
)

BooleanBackend = Literal["manifold", "surface"]
SurfaceBooleanOperation = Literal["union", "difference", "intersection"]
SurfaceBooleanStatus = Literal["succeeded", "invalid", "unsupported"]
SurfaceBooleanClassification = Literal["open", "closed", "empty"]
SurfaceBooleanBodyRelation = Literal["disjoint", "touching", "overlap", "containment", "equal"]
SurfaceBooleanPatchRelation = Literal["inside", "outside", "on"]
SurfaceBooleanSplitRole = Literal["survive", "cut_cap", "discard"]


class BooleanOperationError(RuntimeError):
    """Raised when a boolean operation cannot produce a valid solid."""


class SurfaceBooleanEligibilityError(BooleanOperationError):
    """Raised when surfaced boolean inputs violate the current v1 contract."""


class SurfaceBooleanExecutionUnavailableError(BooleanOperationError):
    """Raised when a caller explicitly requires execution that is not implemented yet."""

    def __init__(self, operation: SurfaceBooleanOperation, operand_ids: tuple[str, ...]) -> None:
        self.operation = operation
        self.operand_ids = operand_ids
        super().__init__(
            f"Surface boolean {operation} execution is not implemented yet after canonical input preparation."
        )


@dataclass(frozen=True)
class SurfaceBooleanOperands:
    """Canonical surfaced boolean operands ready for execution."""

    operation: SurfaceBooleanOperation
    bodies: tuple[SurfaceBody, ...]

    @property
    def operand_count(self) -> int:
        return len(self.bodies)

    @property
    def body_ids(self) -> tuple[str, ...]:
        return tuple(body.stable_identity for body in self.bodies)


@dataclass(frozen=True)
class SurfaceBooleanResult:
    """Structured surfaced boolean result contract."""

    operation: SurfaceBooleanOperation
    operands: SurfaceBooleanOperands
    status: SurfaceBooleanStatus
    body: SurfaceBody | None = None
    classification: SurfaceBooleanClassification | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "succeeded":
            if self.classification is None:
                raise ValueError("Succeeded surface boolean results require classification.")
            if self.classification == "empty":
                if self.body is not None:
                    raise ValueError("Empty succeeded surface boolean results may not carry body.")
            elif self.body is None:
                raise ValueError("Non-empty succeeded surface boolean results require body.")
            if self.failure_reason is not None:
                raise ValueError("Succeeded surface boolean results may not carry failure_reason.")
        else:
            if self.body is not None or self.classification is not None:
                raise ValueError("Invalid or unsupported surface boolean results may not carry body or classification.")
            if not self.failure_reason:
                raise ValueError("Invalid or unsupported surface boolean results require failure_reason.")

    @property
    def body_id(self) -> str | None:
        return None if self.body is None else self.body.stable_identity


@dataclass(frozen=True)
class SurfaceBooleanPatchRef:
    """Stable reference to one patch in one boolean operand."""

    operand_index: int
    patch_index: int


@dataclass(frozen=True)
class SurfaceBooleanTrimFragment:
    """One cut fragment expressed in patch-local UV coordinates."""

    patch: SurfaceBooleanPatchRef
    points_uv: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SurfaceBooleanCutCurve:
    """One surfaced cut curve shared by two operand patches."""

    cut_curve_id: str
    points_3d: tuple[tuple[float, float, float], ...]
    patches: tuple[SurfaceBooleanPatchRef, SurfaceBooleanPatchRef]
    trim_fragments: tuple[SurfaceBooleanTrimFragment, SurfaceBooleanTrimFragment]


@dataclass(frozen=True)
class SurfaceBooleanPatchClassification:
    """One deterministic patch classification relative to the opposing operand."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceBooleanSplitRecord:
    """One deterministic surfaced split-selection record for a source patch fragment."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    role: SurfaceBooleanSplitRole
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceBooleanTrimmedPatchFragment:
    """One reconstructed surfaced patch fragment before shell assembly."""

    source_patch: SurfaceBooleanPatchRef
    patch: PlanarSurfacePatch
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceBooleanIntersectionStage:
    """Deterministic surfaced intersection/classification stage output."""

    operation: SurfaceBooleanOperation
    operands: SurfaceBooleanOperands
    supported: bool
    body_relation: SurfaceBooleanBodyRelation
    cut_curves: tuple[SurfaceBooleanCutCurve, ...] = ()
    patch_classifications: tuple[SurfaceBooleanPatchClassification, ...] = ()
    split_records: tuple[SurfaceBooleanSplitRecord, ...] = ()
    support_reason: str | None = None

    def __post_init__(self) -> None:
        if self.supported and self.support_reason is not None:
            raise ValueError("Supported intersection stages may not carry support_reason.")
        if not self.supported and not self.support_reason:
            raise ValueError("Unsupported intersection stages require support_reason.")


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend not in {"manifold", "surface"}:
        raise ValueError(f"Unsupported backend '{backend}'. Only 'manifold' and 'surface' are available right now.")


def _load_manifold():
    try:
        from manifold3d import Manifold, Mesh as ManifoldMesh
    except ImportError as exc:  # pragma: no cover - runtime dep
        raise BooleanOperationError(
            "manifold3d is required for boolean operations. Install it with `pip install manifold3d`."
        ) from exc
    return Manifold, ManifoldMesh


def _mesh_from_manifold(manifold_mesh) -> Mesh:
    vertices = None
    faces = None
    for attr in ("vertices", "vert_properties", "verts"):
        if hasattr(manifold_mesh, attr):
            vertices = np.asarray(getattr(manifold_mesh, attr), dtype=float)
            break
    for attr in ("triangles", "tri_verts", "faces"):
        if hasattr(manifold_mesh, attr):
            faces = np.asarray(getattr(manifold_mesh, attr), dtype=int)
            break
    if vertices is None or faces is None:
        raise BooleanOperationError("manifold3d returned an unexpected mesh format.")
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise BooleanOperationError("manifold3d returned invalid vertex data.")
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BooleanOperationError("manifold3d returned non-triangular faces.")
    return Mesh(vertices, faces)


def _manifold_from_mesh(mesh: Mesh, face_id: int | None = None):
    Manifold, ManifoldMesh = _load_manifold()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    face_ids = None
    if face_id is not None:
        face_ids = np.full(mesh.n_faces, face_id, dtype=np.uint32)
    try:
        manifold_mesh = ManifoldMesh(vertices, faces, face_id=face_ids)
    except TypeError:
        try:
            manifold_mesh = ManifoldMesh(vertices=vertices, triangles=faces, face_id=face_ids)
        except TypeError as exc:  # pragma: no cover - defensive
            raise BooleanOperationError("Unable to build manifold mesh from input data.") from exc
    try:
        return Manifold(manifold_mesh)
    except Exception as exc:
        raise BooleanOperationError("manifold3d failed to create a solid from the provided mesh.") from exc


def _flatten_meshes(meshes: Iterable[Mesh | MeshGroup]) -> list[Mesh]:
    flattened: list[Mesh] = []
    for mesh in meshes:
        if isinstance(mesh, MeshGroup):
            flattened.extend(mesh.to_meshes())
        elif isinstance(mesh, Mesh):
            flattened.append(mesh)
        else:
            raise TypeError("Boolean operations require Mesh or MeshGroup inputs.")
    return flattened


def _check_mesh(mesh: Mesh) -> Mesh:
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be a (N, 3) triangle array.")
    if mesh.n_faces == 0:
        raise ValueError("Mesh has no faces.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= mesh.n_vertices:
        raise ValueError("Mesh faces contain out-of-range vertex indices.")
    analyze_mesh(mesh)
    issues = mesh.analysis.issues() if mesh.analysis else []
    if issues:
        warnings.warn(
            f"Mesh analysis warnings: {', '.join(issues)}",
            RuntimeWarning,
        )
    return mesh


def _combine_color(result: Mesh, sources: list[Mesh]) -> None:
    for mesh in sources:
        color = get_mesh_color(mesh)
        if color is not None:
            set_mesh_color(result, (*color[0], color[1]))
            return


def _resolve_mesh_rgba(mesh: Mesh) -> tuple[float, float, float, float] | None:
    if mesh.color is not None:
        return mesh.color
    if mesh.face_colors is None or mesh.face_colors.size == 0:
        return None
    colors = np.asarray(mesh.face_colors, dtype=float)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    rgb = colors[:, :3].mean(axis=0)
    alpha = colors[:, 3].mean() if colors.shape[1] >= 4 else 1.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def _apply_face_colors(result: Mesh, face_ids: np.ndarray, color_map: dict[int, tuple[float, float, float, float] | None]) -> None:
    if face_ids.size == 0:
        return
    face_colors = np.zeros((face_ids.size, 4), dtype=float)
    for idx, face_id in enumerate(face_ids):
        rgba = color_map.get(int(face_id))
        if rgba is None:
            rgba = (0.8, 0.8, 0.8, 1.0)
        face_colors[idx] = rgba
    result.face_colors = face_colors


def _apply_boolean(
    meshes: Iterable[Mesh],
    operation: str,
) -> Mesh:
    meshes_list = [_check_mesh(mesh) for mesh in meshes]
    if not meshes_list:
        raise ValueError(f"boolean_{operation} requires at least one mesh.")

    color_map: dict[int, tuple[float, float, float, float] | None] = {}
    explicit_color = False
    manifold_meshes = []
    for idx, mesh in enumerate(meshes_list, start=1):
        color = _resolve_mesh_rgba(mesh)
        if color is not None:
            explicit_color = True
        color_map[idx] = color
        manifold_meshes.append(_manifold_from_mesh(mesh, face_id=idx))

    base = manifold_meshes[0]
    for other in manifold_meshes[1:]:
        if hasattr(base, operation):
            base = getattr(base, operation)(other)
        elif operation == "union" and hasattr(base, "__add__"):
            base = base + other
        elif operation == "difference" and hasattr(base, "__sub__"):
            base = base - other
        elif operation == "intersection":
            if hasattr(base, "__and__"):
                base = base & other
            elif hasattr(base, "__sub__"):
                # Intersection = A - (A - B) when direct op is unavailable.
                base = base - (base - other)
            else:
                raise BooleanOperationError(
                    "manifold3d does not support intersection on this version."
                )
        else:
            raise BooleanOperationError(f"manifold3d does not support '{operation}' on this version.")

    if hasattr(base, "to_mesh"):
        result_mesh = base.to_mesh()
    elif hasattr(base, "mesh"):
        result_mesh = base.mesh
    else:
        raise BooleanOperationError("manifold3d returned an unexpected result type.")

    result = _mesh_from_manifold(result_mesh)
    if explicit_color and hasattr(result_mesh, "face_id"):
        face_ids = np.asarray(result_mesh.face_id, dtype=int)
        if face_ids.shape[0] == result.n_faces:
            _apply_face_colors(result, face_ids, color_map)
        _combine_color(result, meshes_list)
    else:
        _combine_color(result, meshes_list)
    return result


def _classify_surface_body(body: SurfaceBody) -> Literal["open", "closed"]:
    # Local import keeps the surfaced preparation layer decoupled from the
    # mesh-primary boolean path while still using the canonical shell truth.
    from .tessellation import _classify_body

    return _classify_body(body)


def _aabb_overlap(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return (
        max(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        min(left[3], right[3]),
        max(left[4], right[4]),
        min(left[5], right[5]),
    )


def _surface_boolean_body_relation(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanBodyRelation:
    overlap = _aabb_overlap(left, right)
    spans = (
        overlap[1] - overlap[0],
        overlap[3] - overlap[2],
        overlap[5] - overlap[4],
    )
    if any(span < -epsilon for span in spans):
        return "disjoint"
    if all(abs(left[idx] - right[idx]) <= epsilon for idx in range(6)):
        return "equal"
    if any(abs(span) <= epsilon for span in spans):
        return "touching"

    def _contains(container: tuple[float, ...], candidate: tuple[float, ...]) -> bool:
        return (
            container[0] <= candidate[0] + epsilon
            and container[1] >= candidate[1] - epsilon
            and container[2] <= candidate[2] + epsilon
            and container[3] >= candidate[3] - epsilon
            and container[4] <= candidate[4] + epsilon
            and container[5] >= candidate[5] - epsilon
        )

    if _contains(left, right) or _contains(right, left):
        return "containment"
    return "overlap"


def _contains_bounds(
    container: tuple[float, float, float, float, float, float],
    candidate: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    return (
        container[0] <= candidate[0] + epsilon
        and container[1] >= candidate[1] - epsilon
        and container[2] <= candidate[2] + epsilon
        and container[3] >= candidate[3] - epsilon
        and container[4] <= candidate[4] + epsilon
        and container[5] >= candidate[5] - epsilon
    )


def _bounds_size(bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    return (
        float(bounds[1] - bounds[0]),
        float(bounds[3] - bounds[2]),
        float(bounds[5] - bounds[4]),
    )


def _bounds_center(bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    return (
        float((bounds[0] + bounds[1]) * 0.5),
        float((bounds[2] + bounds[3]) * 0.5),
        float((bounds[4] + bounds[5]) * 0.5),
    )


def _surface_boolean_provenance_payload(operands: SurfaceBooleanOperands) -> dict[str, object]:
    return {
        "backend": "surface",
        "operation": operands.operation,
        "operand_ids": operands.body_ids,
    }


def _surface_boolean_result_metadata(operands: SurfaceBooleanOperands) -> dict[str, object]:
    inherited_kernel = dict(operands.bodies[0].kernel_metadata())
    inherited_consumer = dict(operands.bodies[0].consumer_metadata())
    for body in operands.bodies[1:]:
        inherited_kernel.update(body.kernel_metadata())
        inherited_consumer.update(body.consumer_metadata())
    provenance = _surface_boolean_provenance_payload(operands)
    inherited_kernel["boolean_backend"] = "surface"
    inherited_kernel["boolean_operation"] = operands.operation
    inherited_kernel["boolean_operand_ids"] = operands.body_ids
    inherited_kernel["boolean_provenance"] = provenance
    inherited_consumer["boolean_backend"] = "surface"
    inherited_consumer["boolean_operation"] = operands.operation
    inherited_consumer["boolean_operand_ids"] = operands.body_ids
    inherited_consumer["boolean_provenance"] = provenance
    return {
        "kernel": inherited_kernel,
        "consumer": inherited_consumer,
    }


def _surface_boolean_boundary_key(boundary) -> tuple[int, str]:
    return (boundary.patch_index, boundary.boundary_id)


def _surface_boolean_trim_key(trim_loop: TrimLoop) -> tuple[object, ...]:
    normalized = trim_loop.normalized()
    return (
        normalized.category,
        *(tuple(float(value) for value in point) for point in np.round(normalized.points_uv, decimals=12)),
    )


def _surface_boolean_cleanup_patch(
    patch,
    *,
    epsilon: float = 1e-12,
):
    if not patch.trim_loops:
        return patch

    cleaned_trim_loops: list[TrimLoop] = []
    seen: set[tuple[object, ...]] = set()
    for trim_loop in patch.trim_loops:
        normalized = trim_loop.normalized()
        if abs(normalized.area) <= epsilon:
            continue
        trim_key = _surface_boolean_trim_key(normalized)
        if trim_key in seen:
            continue
        seen.add(trim_key)
        cleaned_trim_loops.append(normalized)
    ordered_trim_loops = tuple(
        sorted(
            cleaned_trim_loops,
            key=lambda loop: (0 if loop.category == "outer" else 1, _surface_boolean_trim_key(loop)),
        )
    )
    return replace(patch, trim_loops=ordered_trim_loops)


def _surface_boolean_cleanup_shell(shell) -> object:
    cleaned_patches = tuple(
        _surface_boolean_cleanup_patch(patch)
        for patch in shell.iter_patches(world=True)
    )
    cleaned_seams = []
    seen_seam_keys: set[tuple[object, ...]] = set()
    for seam in sorted(
        shell.seams,
        key=lambda item: (
            tuple(sorted(_surface_boolean_boundary_key(boundary) for boundary in item.boundaries)),
            item.continuity,
            item.seam_id,
        ),
    ):
        canonical_boundaries = tuple(sorted(seam.boundaries, key=_surface_boolean_boundary_key))
        seam_key = (
            tuple(_surface_boolean_boundary_key(boundary) for boundary in canonical_boundaries),
            seam.continuity,
        )
        if seam_key in seen_seam_keys:
            continue
        seen_seam_keys.add(seam_key)
        cleaned_seams.append(replace(seam, boundaries=canonical_boundaries))

    cleaned_adjacency = tuple(
        sorted(
            shell.adjacency,
            key=lambda record: (
                record.source.patch_index,
                record.source.boundary_id,
                -1 if record.target is None else record.target.patch_index,
                "" if record.target is None else record.target.boundary_id,
                "" if record.seam_id is None else record.seam_id,
                record.continuity,
            ),
        )
    )
    return make_surface_shell(
        cleaned_patches,
        connected=shell.connected,
        seams=tuple(cleaned_seams),
        adjacency=cleaned_adjacency,
        metadata=shell.metadata,
    )


def _surface_boolean_cleanup_body(body: SurfaceBody) -> SurfaceBody:
    cleaned_shells = tuple(_surface_boolean_cleanup_shell(shell) for shell in body.iter_shells(world=True))
    return make_surface_body(cleaned_shells, metadata=body.metadata)


def _surface_boolean_shell_invalid_reason(
    shell,
    *,
    epsilon: float = 1e-9,
) -> str | None:
    from .tessellation import _boundary_is_collapsed, _patch_boundary_ids

    if not shell.connected:
        return "Surface boolean validity gate rejected a disconnected shell."

    boundary_use_counts: dict[tuple[int, str], int] = {}
    for patch_index, patch in enumerate(shell.patches):
        for boundary_id in _patch_boundary_ids(patch):
            boundary_use_counts[(patch_index, boundary_id)] = 0

    for seam in shell.seams:
        if seam.is_open:
            return f"Surface boolean validity gate rejected open seam {seam.seam_id!r}."
        for boundary in seam.boundaries:
            boundary_key = _surface_boolean_boundary_key(boundary)
            if boundary_key not in boundary_use_counts:
                return (
                    "Surface boolean validity gate found a seam boundary outside the reconstructed "
                    "patch boundary set."
                )
            boundary_use_counts[boundary_key] += 1

    for (patch_index, boundary_id), use_count in boundary_use_counts.items():
        patch = shell.patches[patch_index]
        if use_count == 0 and _boundary_is_collapsed(patch, boundary_id, tolerance=epsilon):
            continue
        if use_count == 0:
            return (
                "Surface boolean validity gate rejected a shell with missing seam coverage on "
                f"patch {patch_index} boundary {boundary_id!r}."
            )
        if use_count > 1:
            return (
                "Surface boolean validity gate rejected duplicate seam use on "
                f"patch {patch_index} boundary {boundary_id!r}."
            )
    return None


def _surface_boolean_finalize_body_result(
    operation: SurfaceBooleanOperation,
    operands: SurfaceBooleanOperands,
    body: SurfaceBody,
) -> SurfaceBooleanResult:
    cleaned_body = _surface_boolean_cleanup_body(body)
    for shell in cleaned_body.iter_shells(world=True):
        invalid_reason = _surface_boolean_shell_invalid_reason(shell)
        if invalid_reason is not None:
            return SurfaceBooleanResult(
                operation=operation,
                operands=operands,
                status="invalid",
                failure_reason=invalid_reason,
            )

    classification = _classify_surface_body(cleaned_body)
    if classification != "closed":
        return SurfaceBooleanResult(
            operation=operation,
            operands=operands,
            status="invalid",
            failure_reason="Surface boolean validity gate rejected a non-closed reconstructed result.",
        )
    return SurfaceBooleanResult(
        operation=operation,
        operands=operands,
        status="succeeded",
        body=cleaned_body,
        classification=classification,
    )


def _clone_surface_shell(shell) -> object:
    return make_surface_shell(
        shell.iter_patches(world=True),
        connected=shell.connected,
        seams=shell.seams,
        adjacency=shell.adjacency,
        metadata=shell.metadata,
    )


def _clone_surface_body_with_metadata(
    body: SurfaceBody,
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    shells = tuple(_clone_surface_shell(shell) for shell in body.iter_shells(world=True))
    return make_surface_body(shells, metadata=metadata)


def _combine_surface_bodies_with_metadata(
    bodies: Sequence[SurfaceBody],
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    shells = tuple(_clone_surface_shell(shell) for body in bodies for shell in body.iter_shells(world=True))
    return make_surface_body(shells, metadata=metadata)


def _surface_body_primitive_family(body: SurfaceBody) -> str | None:
    if body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    kernel = shell.metadata.get("kernel", {})
    family = kernel.get("primitive_family")
    if isinstance(family, str) and family:
        return family
    if shell.patch_count == 1:
        patch_family = shell.iter_patches(world=True)[0].kernel_metadata().get("primitive_family")
        if isinstance(patch_family, str) and patch_family:
            return patch_family
    return None


def _bounds_corners(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.asarray(
        [
            (bounds[0], bounds[2], bounds[4]),
            (bounds[0], bounds[2], bounds[5]),
            (bounds[0], bounds[3], bounds[4]),
            (bounds[0], bounds[3], bounds[5]),
            (bounds[1], bounds[2], bounds[4]),
            (bounds[1], bounds[2], bounds[5]),
            (bounds[1], bounds[3], bounds[4]),
            (bounds[1], bounds[3], bounds[5]),
        ],
        dtype=float,
    )


def _surface_body_sphere_parameters(
    body: SurfaceBody,
    *,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, float] | None:
    if _surface_body_primitive_family(body) != "sphere" or body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    if shell.patch_count != 1:
        return None
    patch = shell.iter_patches(world=True)[0]
    if not isinstance(patch, RevolutionSurfacePatch):
        return None
    bounds = body.bounds_estimate()
    spans = _bounds_size(bounds)
    if not (abs(spans[0] - spans[1]) <= epsilon and abs(spans[1] - spans[2]) <= epsilon):
        return None
    radius = spans[0] * 0.5
    if radius <= epsilon:
        return None
    return np.asarray(_bounds_center(bounds), dtype=float), float(radius)


def _surface_body_contains_exact(
    container: SurfaceBody,
    candidate: SurfaceBody,
    *,
    epsilon: float = 1e-9,
) -> bool:
    container_family = _surface_body_primitive_family(container)
    candidate_family = _surface_body_primitive_family(candidate)
    container_bounds = container.bounds_estimate()
    candidate_bounds = candidate.bounds_estimate()

    if container_family == "box" and candidate_family == "box":
        return _contains_bounds(container_bounds, candidate_bounds, epsilon=epsilon)
    if container_family == "box" and candidate_family == "sphere":
        return _contains_bounds(container_bounds, candidate_bounds, epsilon=epsilon)
    if container_family == "sphere" and candidate_family == "sphere":
        container_sphere = _surface_body_sphere_parameters(container, epsilon=epsilon)
        candidate_sphere = _surface_body_sphere_parameters(candidate, epsilon=epsilon)
        if container_sphere is None or candidate_sphere is None:
            return False
        container_center, container_radius = container_sphere
        candidate_center, candidate_radius = candidate_sphere
        return (
            float(np.linalg.norm(candidate_center - container_center)) + candidate_radius
            <= container_radius + epsilon
        )
    if container_family == "sphere" and candidate_family == "box":
        container_sphere = _surface_body_sphere_parameters(container, epsilon=epsilon)
        if container_sphere is None:
            return False
        container_center, container_radius = container_sphere
        corners = _bounds_corners(candidate_bounds)
        distances = np.linalg.norm(corners - container_center, axis=1)
        return bool(np.all(distances <= container_radius + epsilon))
    return False


def _surface_boolean_trivial_result(operands: SurfaceBooleanOperands) -> SurfaceBooleanResult | None:
    metadata = _surface_boolean_result_metadata(operands)

    if operands.operation == "difference":
        base = operands.bodies[0]
        cutters = operands.bodies[1:]
        if all(
            _surface_boolean_body_relation(base.bounds_estimate(), cutter.bounds_estimate()) in {"disjoint", "touching"}
            for cutter in cutters
        ):
            body = _clone_surface_body_with_metadata(base, metadata=metadata)
            return _surface_boolean_finalize_body_result("difference", operands, body)
        if any(
            _surface_boolean_body_relation(base.bounds_estimate(), cutter.bounds_estimate()) == "equal"
            or _surface_body_contains_exact(cutter, base)
            for cutter in cutters
        ):
            return SurfaceBooleanResult(
                operation="difference",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        return None

    if operands.operand_count != 2:
        return None

    left, right = operands.bodies
    relation = _surface_boolean_body_relation(left.bounds_estimate(), right.bounds_estimate())
    left_contains_right = relation in {"containment", "equal"} and _surface_body_contains_exact(left, right)
    right_contains_left = relation in {"containment", "equal"} and _surface_body_contains_exact(right, left)

    if operands.operation == "union":
        if relation == "disjoint":
            body = _combine_surface_bodies_with_metadata((left, right), metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "equal" or (left_contains_right and right_contains_left):
            body = _clone_surface_body_with_metadata(left, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if left_contains_right:
            body = _clone_surface_body_with_metadata(left, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if right_contains_left:
            body = _clone_surface_body_with_metadata(right, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        return None

    if relation in {"disjoint", "touching"}:
        return SurfaceBooleanResult(
            operation="intersection",
            operands=operands,
            status="succeeded",
            classification="empty",
        )
    if relation == "equal" or (left_contains_right and right_contains_left):
        body = _clone_surface_body_with_metadata(left, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    if left_contains_right:
        body = _clone_surface_body_with_metadata(right, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    if right_contains_left:
        body = _clone_surface_body_with_metadata(left, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    return None


def _surface_box_body_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    from ._surface_primitives import make_surface_box

    return make_surface_box(
        size=_bounds_size(bounds),
        center=_bounds_center(bounds),
        metadata=metadata,
    )


def _sorted_unique_axis_values(
    values: Iterable[float],
    *,
    epsilon: float = 1e-9,
) -> tuple[float, ...]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if not unique or abs(value - unique[-1]) > epsilon:
            unique.append(value)
    return tuple(unique)


def _point_inside_bounds(
    point: Sequence[float],
    bounds: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    x, y, z = (float(coord) for coord in point)
    return (
        (bounds[0] - epsilon) <= x <= (bounds[1] + epsilon)
        and (bounds[2] - epsilon) <= y <= (bounds[3] + epsilon)
        and (bounds[4] - epsilon) <= z <= (bounds[5] + epsilon)
    )


@dataclass(frozen=True)
class _RectilinearBoundaryFace:
    axis: int
    side: Literal["min", "max"]
    cell_index: tuple[int, int, int]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]


def _rectilinear_boundary_sort_key(face: _RectilinearBoundaryFace) -> tuple[object, ...]:
    return (
        face.axis,
        0 if face.side == "min" else 1,
        *(round(value, 12) for value in face.mins),
        *(round(value, 12) for value in face.maxs),
        *face.cell_index,
    )


def _surface_rectangular_patch_from_face(
    face: _RectilinearBoundaryFace,
    *,
    operation: SurfaceBooleanOperation,
) -> PlanarSurfacePatch:
    x0, y0, z0 = face.mins
    x1, y1, z1 = face.maxs
    axis_name = _axis_name(face.axis)
    metadata = {
        "kernel": {
            "primitive_family": "orthogonal_boolean",
            "boolean_operation": operation,
            "axis": axis_name,
            "side": face.side,
        }
    }
    if face.axis == 0 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x1, y0, z0),
            u_axis=(0.0, 0.0, z1 - z0),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    if face.axis == 0 and face.side == "min":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y0, z1),
            u_axis=(0.0, 0.0, z0 - z1),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    if face.axis == 1 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y1, z0),
            u_axis=(x1 - x0, 0.0, 0.0),
            v_axis=(0.0, 0.0, z1 - z0),
            metadata=metadata,
        )
    if face.axis == 1 and face.side == "min":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y0, z1),
            u_axis=(x1 - x0, 0.0, 0.0),
            v_axis=(0.0, 0.0, z0 - z1),
            metadata=metadata,
        )
    if face.axis == 2 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x1, y0, z1),
            u_axis=(x0 - x1, 0.0, 0.0),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    return PlanarSurfacePatch(
        family="planar",
        origin=(x0, y0, z0),
        u_axis=(x1 - x0, 0.0, 0.0),
        v_axis=(0.0, y1 - y0, 0.0),
        metadata=metadata,
    )


def _patch_boundary_endpoints(
    patch: PlanarSurfacePatch,
    boundary_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if boundary_id == "left":
        return patch.point_at(u0, v0), patch.point_at(u0, v1)
    if boundary_id == "right":
        return patch.point_at(u1, v0), patch.point_at(u1, v1)
    if boundary_id == "bottom":
        return patch.point_at(u0, v0), patch.point_at(u1, v0)
    if boundary_id == "top":
        return patch.point_at(u0, v1), patch.point_at(u1, v1)
    raise ValueError(f"Unsupported boundary_id {boundary_id!r} for planar boundary endpoints.")


def _segment_key(
    first: np.ndarray,
    second: np.ndarray,
    *,
    decimals: int = 12,
) -> tuple[object, ...]:
    points = tuple(
        sorted(
            (
                tuple(float(value) for value in np.round(np.asarray(first, dtype=float), decimals=decimals)),
                tuple(float(value) for value in np.round(np.asarray(second, dtype=float), decimals=decimals)),
            )
        )
    )
    return ("segment", *points[0], *points[1])


def _result_cell_occupied(
    operation: SurfaceBooleanOperation,
    point: Sequence[float],
    left_bounds: tuple[float, float, float, float, float, float],
    right_bounds: tuple[float, float, float, float, float, float],
) -> bool:
    in_left = _point_inside_bounds(point, left_bounds)
    in_right = _point_inside_bounds(point, right_bounds)
    if operation == "union":
        return in_left or in_right
    if operation == "difference":
        return in_left and not in_right
    raise ValueError(f"Unsupported orthogonal surfaced boolean operation {operation!r}.")


def _occupied_cell_components(
    occupied: set[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    remaining = set(occupied)
    components: list[tuple[tuple[int, int, int], ...]] = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        component: list[tuple[int, int, int]] = []
        remaining.remove(seed)
        while stack:
            ix, iy, iz = stack.pop()
            component.append((ix, iy, iz))
            for neighbor in (
                (ix - 1, iy, iz),
                (ix + 1, iy, iz),
                (ix, iy - 1, iz),
                (ix, iy + 1, iz),
                (ix, iy, iz - 1),
                (ix, iy, iz + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _surface_orthogonal_box_boolean_body(
    left_bounds: tuple[float, float, float, float, float, float],
    right_bounds: tuple[float, float, float, float, float, float],
    *,
    operation: SurfaceBooleanOperation,
    metadata: dict[str, object],
) -> SurfaceBody | None:
    xs = _sorted_unique_axis_values((left_bounds[0], left_bounds[1], right_bounds[0], right_bounds[1]))
    ys = _sorted_unique_axis_values((left_bounds[2], left_bounds[3], right_bounds[2], right_bounds[3]))
    zs = _sorted_unique_axis_values((left_bounds[4], left_bounds[5], right_bounds[4], right_bounds[5]))
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        return None

    occupied: set[tuple[int, int, int]] = set()
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            for iz in range(len(zs) - 1):
                midpoint = (
                    (xs[ix] + xs[ix + 1]) * 0.5,
                    (ys[iy] + ys[iy + 1]) * 0.5,
                    (zs[iz] + zs[iz + 1]) * 0.5,
                )
                if _result_cell_occupied(operation, midpoint, left_bounds, right_bounds):
                    occupied.add((ix, iy, iz))
    if not occupied or len(_occupied_cell_components(occupied)) != 1:
        return None

    faces: list[_RectilinearBoundaryFace] = []
    for ix, iy, iz in sorted(occupied):
        mins = (xs[ix], ys[iy], zs[iz])
        maxs = (xs[ix + 1], ys[iy + 1], zs[iz + 1])
        neighbors = {
            (0, "min"): (ix - 1, iy, iz),
            (0, "max"): (ix + 1, iy, iz),
            (1, "min"): (ix, iy - 1, iz),
            (1, "max"): (ix, iy + 1, iz),
            (2, "min"): (ix, iy, iz - 1),
            (2, "max"): (ix, iy, iz + 1),
        }
        for (axis, side), neighbor in neighbors.items():
            if neighbor in occupied:
                continue
            faces.append(
                _RectilinearBoundaryFace(
                    axis=axis,
                    side=side,
                    cell_index=(ix, iy, iz),
                    mins=mins,
                    maxs=maxs,
                )
            )

    ordered_faces = tuple(sorted(faces, key=_rectilinear_boundary_sort_key))
    patches = tuple(
        _surface_rectangular_patch_from_face(face, operation=operation)
        for face in ordered_faces
    )

    segment_map: dict[tuple[object, ...], list[SurfaceBoundaryRef]] = {}
    for patch_index, patch in enumerate(patches):
        for boundary_id in ("left", "right", "bottom", "top"):
            first, second = _patch_boundary_endpoints(patch, boundary_id)
            segment_map.setdefault(_segment_key(first, second), []).append(
                SurfaceBoundaryRef(patch_index, boundary_id)
            )

    seam_objects = []
    for seam_index, segment_key in enumerate(sorted(segment_map)):
        boundaries = tuple(sorted(segment_map[segment_key], key=lambda ref: (ref.patch_index, ref.boundary_id)))
        if len(boundaries) != 2:
            return None
        seam_objects.append(
            SurfaceSeam(
                seam_id=f"orthogonal-boolean-seam-{seam_index:03d}",
                boundaries=boundaries,
            )
        )
    seam_objects = tuple(seam_objects)

    adjacency: dict[int, set[int]] = {patch_index: set() for patch_index in range(len(patches))}
    for seam in seam_objects:
        if len(seam.boundaries) != 2:
            return None
        first, second = seam.boundaries
        adjacency[first.patch_index].add(second.patch_index)
        adjacency[second.patch_index].add(first.patch_index)
    if patches:
        visited = set()
        stack = [0]
        while stack:
            patch_index = stack.pop()
            if patch_index in visited:
                continue
            visited.add(patch_index)
            stack.extend(sorted(adjacency[patch_index] - visited))
        if len(visited) != len(patches):
            return None

    shell = make_surface_shell(
        patches,
        connected=True,
        seams=seam_objects,
        metadata={"kernel": {"primitive_family": "orthogonal_boolean", "boolean_operation": operation}},
    )
    return make_surface_body((shell,), metadata=metadata)


@dataclass(frozen=True)
class _AxisAlignedPlanarPatch:
    operand_index: int
    patch_index: int
    patch: PlanarSurfacePatch
    axis: int
    coordinate: float
    min_corner: np.ndarray
    max_corner: np.ndarray


def _axis_name(axis_index: int) -> str:
    return ("x", "y", "z")[axis_index]


def _planar_patch_point_to_uv(patch: PlanarSurfacePatch, point: Sequence[float]) -> tuple[float, float]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    anchor = patch.point_at(u0, v0)
    du, dv = patch.derivatives_at(u0, v0)
    basis = np.column_stack((du, dv))
    delta = np.asarray(point, dtype=float).reshape(3) - anchor
    params, *_rest = np.linalg.lstsq(basis, delta, rcond=None)
    return (float(u0 + params[0]), float(v0 + params[1]))


def _extract_axis_aligned_planar_patches(body: SurfaceBody, *, operand_index: int) -> tuple[_AxisAlignedPlanarPatch, ...] | None:
    if body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    if shell.patch_count != 6:
        return None
    extracted: list[_AxisAlignedPlanarPatch] = []
    for patch_index, patch in enumerate(shell.iter_patches(world=True)):
        if not isinstance(patch, PlanarSurfacePatch) or patch.trim_loops:
            return None
        u0, u1 = patch.domain.u_range
        v0, v1 = patch.domain.v_range
        corners = np.asarray(
            [
                patch.point_at(u0, v0),
                patch.point_at(u0, v1),
                patch.point_at(u1, v0),
                patch.point_at(u1, v1),
            ],
            dtype=float,
        )
        du, dv = patch.derivatives_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)
        normal = np.cross(du, dv)
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            return None
        normal /= norm
        axis = int(np.argmax(np.abs(normal)))
        if not np.allclose(np.abs(normal), np.eye(3, dtype=float)[axis], atol=1e-8):
            return None
        coordinate = float(corners[:, axis].mean())
        if not np.allclose(corners[:, axis], coordinate, atol=1e-8):
            return None
        extracted.append(
            _AxisAlignedPlanarPatch(
                operand_index=operand_index,
                patch_index=patch_index,
                patch=patch,
                axis=axis,
                coordinate=coordinate,
                min_corner=corners.min(axis=0),
                max_corner=corners.max(axis=0),
            )
        )
    return tuple(extracted)


def _classify_patch_against_bounds(
    patch: PlanarSurfacePatch,
    bounds: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanPatchRelation:
    u = float(sum(patch.domain.u_range) * 0.5)
    v = float(sum(patch.domain.v_range) * 0.5)
    point = patch.point_at(u, v)
    inside = (
        (bounds[0] + epsilon) < point[0] < (bounds[1] - epsilon)
        and (bounds[2] + epsilon) < point[1] < (bounds[3] - epsilon)
        and (bounds[4] + epsilon) < point[2] < (bounds[5] - epsilon)
    )
    if inside:
        return "inside"
    on = (
        (bounds[0] - epsilon) <= point[0] <= (bounds[1] + epsilon)
        and (bounds[2] - epsilon) <= point[1] <= (bounds[3] + epsilon)
        and (bounds[4] - epsilon) <= point[2] <= (bounds[5] + epsilon)
        and (
            abs(point[0] - bounds[0]) <= epsilon
            or abs(point[0] - bounds[1]) <= epsilon
            or abs(point[1] - bounds[2]) <= epsilon
            or abs(point[1] - bounds[3]) <= epsilon
            or abs(point[2] - bounds[4]) <= epsilon
            or abs(point[2] - bounds[5]) <= epsilon
        )
    )
    return "on" if on else "outside"


def _surface_boolean_split_role(
    operation: SurfaceBooleanOperation,
    *,
    operand_index: int,
    relation: SurfaceBooleanPatchRelation,
) -> SurfaceBooleanSplitRole:
    if operation == "union":
        return "discard" if relation == "inside" else "survive"
    if operation == "intersection":
        return "discard" if relation == "outside" else "survive"
    if operand_index == 0:
        return "discard" if relation == "inside" else "survive"
    return "discard" if relation == "outside" else "cut_cap"


def _cut_curve_id(
    first: SurfaceBooleanPatchRef,
    second: SurfaceBooleanPatchRef,
    points_3d: tuple[tuple[float, float, float], ...],
) -> str:
    return (
        f"operand{first.operand_index}:patch{first.patch_index}|"
        f"operand{second.operand_index}:patch{second.patch_index}|"
        f"{points_3d[0]}->{points_3d[-1]}"
    )


def _intersect_axis_aligned_patch_pair(
    first: _AxisAlignedPlanarPatch,
    second: _AxisAlignedPlanarPatch,
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanCutCurve | None:
    if first.axis == second.axis:
        if abs(first.coordinate - second.coordinate) <= epsilon:
            other_axes = tuple(axis for axis in (0, 1, 2) if axis != first.axis)
            overlap_a = min(first.max_corner[other_axes[0]], second.max_corner[other_axes[0]]) - max(
                first.min_corner[other_axes[0]], second.min_corner[other_axes[0]]
            )
            overlap_b = min(first.max_corner[other_axes[1]], second.max_corner[other_axes[1]]) - max(
                first.min_corner[other_axes[1]], second.min_corner[other_axes[1]]
            )
            if overlap_a > epsilon and overlap_b > epsilon:
                return None
        return None

    remaining_axis = next(axis for axis in (0, 1, 2) if axis not in {first.axis, second.axis})
    if not (second.min_corner[first.axis] - epsilon <= first.coordinate <= second.max_corner[first.axis] + epsilon):
        return None
    if not (first.min_corner[second.axis] - epsilon <= second.coordinate <= first.max_corner[second.axis] + epsilon):
        return None
    seg_min = max(first.min_corner[remaining_axis], second.min_corner[remaining_axis])
    seg_max = min(first.max_corner[remaining_axis], second.max_corner[remaining_axis])
    if seg_max - seg_min <= epsilon:
        return None

    start = np.zeros(3, dtype=float)
    end = np.zeros(3, dtype=float)
    start[first.axis] = first.coordinate
    start[second.axis] = second.coordinate
    start[remaining_axis] = seg_min
    end[:] = start
    end[remaining_axis] = seg_max
    points_3d = (
        (float(start[0]), float(start[1]), float(start[2])),
        (float(end[0]), float(end[1]), float(end[2])),
    )
    first_ref = SurfaceBooleanPatchRef(first.operand_index, first.patch_index)
    second_ref = SurfaceBooleanPatchRef(second.operand_index, second.patch_index)
    first_trim = SurfaceBooleanTrimFragment(
        patch=first_ref,
        points_uv=(
            _planar_patch_point_to_uv(first.patch, start),
            _planar_patch_point_to_uv(first.patch, end),
        ),
    )
    second_trim = SurfaceBooleanTrimFragment(
        patch=second_ref,
        points_uv=(
            _planar_patch_point_to_uv(second.patch, start),
            _planar_patch_point_to_uv(second.patch, end),
        ),
    )
    return SurfaceBooleanCutCurve(
        cut_curve_id=_cut_curve_id(first_ref, second_ref, points_3d),
        points_3d=points_3d,
        patches=(first_ref, second_ref),
        trim_fragments=(first_trim, second_trim),
    )


def _sorted_cut_curve_ids_for_patch(
    cut_curves: Sequence[SurfaceBooleanCutCurve],
    patch: SurfaceBooleanPatchRef,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            curve.cut_curve_id
            for curve in cut_curves
            if any(
                ref.operand_index == patch.operand_index and ref.patch_index == patch.patch_index
                for ref in curve.patches
            )
        )
    )


def _trim_loop_for_overlap_fragment(
    fragment: _AxisAlignedPlanarPatch,
    overlap_bounds: tuple[float, float, float, float, float, float],
) -> TrimLoop:
    corners: list[tuple[float, float, float]] = []
    for first in (0, 1):
        for second in (0, 1):
            point = [0.0, 0.0, 0.0]
            point[fragment.axis] = fragment.coordinate
            other_axes = [axis for axis in (0, 1, 2) if axis != fragment.axis]
            point[other_axes[0]] = overlap_bounds[(other_axes[0] * 2) + first]
            point[other_axes[1]] = overlap_bounds[(other_axes[1] * 2) + second]
            corners.append((point[0], point[1], point[2]))
    ordered_corners = (corners[0], corners[1], corners[3], corners[2])
    u0, v0 = _planar_patch_point_to_uv(fragment.patch, ordered_corners[0])
    points_uv = [(u0, v0)]
    for point in ordered_corners[1:]:
        points_uv.append(_planar_patch_point_to_uv(fragment.patch, point))
    return TrimLoop(points_uv, category="outer").normalized()


def _boundary_contributor_patch(
    patches: Sequence[_AxisAlignedPlanarPatch],
    *,
    axis: int,
    coordinate: float,
    epsilon: float = 1e-9,
) -> _AxisAlignedPlanarPatch | None:
    matches = [
        patch
        for patch in patches
        if patch.axis == axis and abs(patch.coordinate - coordinate) <= epsilon
    ]
    if not matches:
        return None
    return min(matches, key=lambda patch: (patch.operand_index, patch.patch_index))


def surface_boolean_overlap_fragments(operands: SurfaceBooleanOperands) -> tuple[SurfaceBooleanTrimmedPatchFragment, ...]:
    """Reconstruct trimmed planar overlap fragments for the initial box intersection slice."""

    if operands.operation != "intersection":
        return ()
    stage = surface_boolean_intersection_stage(operands)
    if not stage.supported or stage.body_relation != "overlap" or operands.operand_count != 2:
        return ()

    left, right = operands.bodies
    left_patches = _extract_axis_aligned_planar_patches(left, operand_index=0)
    right_patches = _extract_axis_aligned_planar_patches(right, operand_index=1)
    if left_patches is None or right_patches is None:
        return ()

    overlap_bounds = _aabb_overlap(left.bounds_estimate(), right.bounds_estimate())
    all_patches = (*left_patches, *right_patches)
    fragments: list[SurfaceBooleanTrimmedPatchFragment] = []
    for axis in (0, 1, 2):
        for coordinate in (overlap_bounds[axis * 2], overlap_bounds[(axis * 2) + 1]):
            contributor = _boundary_contributor_patch(all_patches, axis=axis, coordinate=coordinate)
            if contributor is None:
                continue
            source_ref = SurfaceBooleanPatchRef(contributor.operand_index, contributor.patch_index)
            trimmed_patch = replace(
                contributor.patch,
                trim_loops=(_trim_loop_for_overlap_fragment(contributor, overlap_bounds),),
            )
            fragments.append(
                SurfaceBooleanTrimmedPatchFragment(
                    source_patch=source_ref,
                    patch=trimmed_patch,
                    cut_curve_ids=_sorted_cut_curve_ids_for_patch(stage.cut_curves, source_ref),
                )
            )
    return tuple(
        sorted(
            fragments,
            key=lambda fragment: (fragment.source_patch.operand_index, fragment.source_patch.patch_index),
        )
    )


def surface_boolean_intersection_stage(operands: SurfaceBooleanOperands) -> SurfaceBooleanIntersectionStage:
    """Compute the first bounded surfaced intersection/classification stage."""

    if operands.operand_count != 2:
        return SurfaceBooleanIntersectionStage(
            operation=operands.operation,
            operands=operands,
            supported=False,
            body_relation="disjoint",
            support_reason="The initial surfaced boolean intersection stage supports exactly two operands.",
        )

    left, right = operands.bodies
    left_bounds = left.bounds_estimate()
    right_bounds = right.bounds_estimate()
    relation = _surface_boolean_body_relation(left_bounds, right_bounds)
    left_patches = _extract_axis_aligned_planar_patches(left, operand_index=0)
    right_patches = _extract_axis_aligned_planar_patches(right, operand_index=1)
    if left_patches is None or right_patches is None:
        return SurfaceBooleanIntersectionStage(
            operation=operands.operation,
            operands=operands,
            supported=False,
            body_relation=relation,
            support_reason=(
                "The initial surfaced boolean intersection stage currently supports only simple "
                "single-shell axis-aligned planar box-style operands without trims."
            ),
        )

    cut_curves = tuple(
        curve
        for left_patch in left_patches
        for right_patch in right_patches
        for curve in (_intersect_axis_aligned_patch_pair(left_patch, right_patch),)
        if curve is not None
    )

    cut_curve_ids_by_patch: dict[tuple[int, int], list[str]] = {}
    for curve in cut_curves:
        for patch_ref in curve.patches:
            cut_curve_ids_by_patch.setdefault((patch_ref.operand_index, patch_ref.patch_index), []).append(curve.cut_curve_id)

    patch_classifications = tuple(
        SurfaceBooleanPatchClassification(
            patch=SurfaceBooleanPatchRef(operand_index, patch.patch_index),
            relation=_classify_patch_against_bounds(
                patch.patch,
                right_bounds if operand_index == 0 else left_bounds,
            ),
            cut_curve_ids=tuple(sorted(cut_curve_ids_by_patch.get((operand_index, patch.patch_index), ()))),
        )
        for operand_index, patch_set in ((0, left_patches), (1, right_patches))
        for patch in patch_set
    )
    split_records = tuple(
        SurfaceBooleanSplitRecord(
            patch=classification.patch,
            relation=classification.relation,
            role=_surface_boolean_split_role(
                operands.operation,
                operand_index=classification.patch.operand_index,
                relation=classification.relation,
            ),
            cut_curve_ids=classification.cut_curve_ids,
        )
        for classification in patch_classifications
    )
    return SurfaceBooleanIntersectionStage(
        operation=operands.operation,
        operands=operands,
        supported=True,
        body_relation=relation,
        cut_curves=tuple(sorted(cut_curves, key=lambda curve: curve.cut_curve_id)),
        patch_classifications=patch_classifications,
        split_records=split_records,
    )


def _surface_boolean_supported_box_result(
    operands: SurfaceBooleanOperands,
    stage: SurfaceBooleanIntersectionStage,
) -> SurfaceBooleanResult | None:
    if not stage.supported or operands.operand_count != 2:
        return None
    left, right = operands.bodies
    left_bounds = left.bounds_estimate()
    right_bounds = right.bounds_estimate()
    relation = stage.body_relation
    metadata = _surface_boolean_result_metadata(operands)

    if operands.operation == "intersection":
        overlap = _aabb_overlap(left_bounds, right_bounds)
        spans = _bounds_size(overlap)
        if relation in {"disjoint", "touching"} or any(span <= 1e-9 for span in spans):
            return SurfaceBooleanResult(
                operation="intersection",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        body = _surface_box_body_from_bounds(overlap, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)

    if operands.operation == "union":
        if relation == "disjoint":
            left_shell = left.iter_shells(world=True)[0]
            right_shell = right.iter_shells(world=True)[0]
            body = make_surface_body((left_shell, right_shell), metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "equal":
            body = _surface_box_body_from_bounds(left_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "containment":
            container_bounds = left_bounds if _contains_bounds(left_bounds, right_bounds) else right_bounds
            body = _surface_box_body_from_bounds(container_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "overlap":
            body = _surface_orthogonal_box_boolean_body(
                left_bounds,
                right_bounds,
                operation="union",
                metadata=metadata,
            )
            if body is not None:
                return _surface_boolean_finalize_body_result("union", operands, body)
        return None

    if operands.operation == "difference":
        if relation in {"disjoint", "touching"}:
            body = _surface_box_body_from_bounds(left_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("difference", operands, body)
        if relation == "equal" or _contains_bounds(right_bounds, left_bounds):
            return SurfaceBooleanResult(
                operation="difference",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        if relation == "overlap":
            body = _surface_orthogonal_box_boolean_body(
                left_bounds,
                right_bounds,
                operation="difference",
                metadata=metadata,
            )
            if body is not None:
                return _surface_boolean_finalize_body_result("difference", operands, body)
        return None

    return None


def _canonicalize_surface_boolean_body(body: SurfaceBody, *, role: str) -> SurfaceBody:
    if not isinstance(body, SurfaceBody):
        raise TypeError(f"{role} must be a SurfaceBody.")
    if body.shell_count != 1:
        raise SurfaceBooleanEligibilityError(f"{role} must contain exactly one shell for surfaced booleans.")

    shell = body.iter_shells(world=True)[0]
    if not shell.connected:
        raise SurfaceBooleanEligibilityError(f"{role} shell must be connected for surfaced booleans.")

    canonical_shell = make_surface_shell(
        shell.iter_patches(world=True),
        connected=shell.connected,
        seams=shell.seams,
        adjacency=shell.adjacency,
        metadata=shell.metadata,
    )
    canonical_body = make_surface_body([canonical_shell], metadata=body.metadata)
    classification = _classify_surface_body(canonical_body)
    if classification != "closed":
        raise SurfaceBooleanEligibilityError(
            f"{role} must be closed-valid under shell seam and boundary truth for surfaced booleans."
        )
    return canonical_body


def prepare_surface_boolean_operands(
    operation: Literal["union", "intersection"],
    bodies: Iterable[SurfaceBody],
) -> SurfaceBooleanOperands:
    if operation not in {"union", "intersection"}:
        raise ValueError("operation must be 'union' or 'intersection'.")
    canonical = tuple(
        _canonicalize_surface_boolean_body(body, role=f"{operation} operand {index}")
        for index, body in enumerate(bodies)
    )
    if len(canonical) < 2:
        raise ValueError(f"surface boolean {operation} requires at least two SurfaceBody operands.")
    return SurfaceBooleanOperands(operation=operation, bodies=canonical)


def prepare_surface_boolean_difference_operands(
    base: SurfaceBody,
    cutters: Iterable[SurfaceBody],
) -> SurfaceBooleanOperands:
    canonical_base = _canonicalize_surface_boolean_body(base, role="difference base")
    canonical_cutters = tuple(
        _canonicalize_surface_boolean_body(body, role=f"difference cutter {index}")
        for index, body in enumerate(cutters)
    )
    if not canonical_cutters:
        raise ValueError("surface boolean difference requires at least one cutter SurfaceBody.")
    return SurfaceBooleanOperands(operation="difference", bodies=(canonical_base, *canonical_cutters))


def surface_boolean_result(operation: SurfaceBooleanOperation, operands: SurfaceBooleanOperands) -> SurfaceBooleanResult:
    """Return the structured surfaced boolean result for the current v1 implementation."""

    if operands.operation != operation:
        raise ValueError("Surface boolean result operation must match prepared operands.")
    trivial_result = _surface_boolean_trivial_result(operands)
    if trivial_result is not None:
        return trivial_result
    stage = surface_boolean_intersection_stage(operands)
    supported_result = _surface_boolean_supported_box_result(operands, stage)
    if supported_result is not None:
        return supported_result
    return SurfaceBooleanResult(
        operation=operation,
        operands=operands,
        status="unsupported",
        failure_reason=(
            f"Surface boolean {operation} execution is not implemented yet after canonical input preparation."
        ),
    )


def _raise_surface_boolean_execution_unavailable(operands: SurfaceBooleanOperands) -> None:
    result = surface_boolean_result(operands.operation, operands)
    raise SurfaceBooleanExecutionUnavailableError(result.operation, result.operands.body_ids)


def boolean_union(
    meshes: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        operands = prepare_surface_boolean_operands("union", meshes)  # type: ignore[arg-type]
        return surface_boolean_result("union", operands)
    warn_mesh_primary_api(
        "boolean_union",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    return _apply_boolean(_flatten_meshes(meshes), "union")


def boolean_difference(
    base: Mesh | MeshGroup | SurfaceBody,
    cutters: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        operands = prepare_surface_boolean_difference_operands(base, cutters)  # type: ignore[arg-type]
        return surface_boolean_result("difference", operands)
    warn_mesh_primary_api(
        "boolean_difference",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    if isinstance(base, MeshGroup):
        base_mesh = base.to_mesh()
    else:
        base_mesh = base
    meshes = _flatten_meshes([base_mesh]) + _flatten_meshes(cutters)
    return _apply_boolean(meshes, "difference")


def boolean_intersection(
    meshes: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        operands = prepare_surface_boolean_operands("intersection", meshes)  # type: ignore[arg-type]
        return surface_boolean_result("intersection", operands)
    warn_mesh_primary_api(
        "boolean_intersection",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    return _apply_boolean(_flatten_meshes(meshes), "intersection")


def union_meshes(
    meshes: Union[Iterable[Mesh | MeshGroup], Mapping[object, Mesh | MeshGroup]],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    """Retained standalone mesh union tool.

    This helper remains useful for mesh analysis, repair, and debugging
    workflows, but it is not canonical surfaced modeling truth.
    """

    warn_mesh_primary_api(
        "union_meshes",
        replacement="SurfaceBody-native composition plus tessellation at the boundary",
    )
    if isinstance(meshes, Mapping):
        meshes = meshes.values()
    return boolean_union(meshes, tolerance=tolerance, backend=backend)
