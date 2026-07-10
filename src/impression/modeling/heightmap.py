from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from PIL import Image

from impression.mesh import Mesh
from impression.cache import LRUCache
from impression.mesh_quality import MeshQuality, apply_lod

from ._legacy_mesh_deprecation import warn_mesh_primary_api

if TYPE_CHECKING:
    from .surface import HeightmapSurfacePatch, SurfaceBody


ArrayLike = np.ndarray
Backend = Literal["mesh", "surface"]

_HEIGHTMAP_CACHE = LRUCache(max_size=32)


@dataclass(frozen=True)
class HeightmapAuthoringRequest:
    """Validated native finite-grid heightmap authoring request."""

    height_samples: np.ndarray
    alpha_mask: np.ndarray | None = None
    alpha_mode: Literal["mask", "ignore"] = "mask"
    xy_scale: float | Sequence[float] = 1.0
    center: Sequence[float] = (0.0, 0.0, 0.0)
    height_scale: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = validate_heightmap_finite_samples(self.height_samples)
        mask = np.ones(samples.shape, dtype=bool) if self.alpha_mask is None else np.asarray(self.alpha_mask, dtype=bool)
        if mask.shape != samples.shape:
            raise ValueError("HeightmapAuthoringRequest.alpha_mask must match height_samples shape.")
        alpha_mode = str(self.alpha_mode)
        if alpha_mode not in {"mask", "ignore"}:
            raise ValueError("HeightmapAuthoringRequest.alpha_mode must be 'mask' or 'ignore'.")
        height_scale = float(self.height_scale)
        if not np.isfinite(height_scale):
            raise ValueError("HeightmapAuthoringRequest.height_scale must be finite.")
        object.__setattr__(self, "height_samples", samples)
        object.__setattr__(self, "alpha_mask", mask)
        object.__setattr__(self, "alpha_mode", alpha_mode)
        object.__setattr__(self, "xy_scale", _as_scale(self.xy_scale))
        object.__setattr__(self, "center", tuple(float(value) for value in np.asarray(self.center, dtype=float).reshape(3)))
        object.__setattr__(self, "height_scale", height_scale)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class HeightmapSampleGridProvenanceRecord:
    """Inspectable provenance for an embedded native heightmap sample grid."""

    family: str
    operation: str
    sample_shape: tuple[int, int]
    masked_sample_count: int
    total_sample_count: int
    authoring_boundary: str = "surface-native"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "operation": self.operation,
            "sample_shape": self.sample_shape,
            "masked_sample_count": self.masked_sample_count,
            "total_sample_count": self.total_sample_count,
            "authoring_boundary": self.authoring_boundary,
        }


@dataclass(frozen=True)
class HeightmapNoDataDiagnostic:
    """Diagnostic for mask/no-data behavior in a finite heightmap grid."""

    code: str
    message: str
    valid: bool
    masked_sample_count: int
    total_sample_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "valid": self.valid,
            "masked_sample_count": self.masked_sample_count,
            "total_sample_count": self.total_sample_count,
        }


@dataclass(frozen=True)
class HeightmapImportRequest:
    """Optional image/array import request that embeds samples as surface truth."""

    source: str | Path | Image.Image | ArrayLike
    height: float = 1.0
    xy_scale: float | Sequence[float] = 1.0
    center: Sequence[float] = (0.0, 0.0, 0.0)
    alpha_mode: Literal["mask", "ignore"] = "mask"
    quality: MeshQuality | None = None
    embed_samples: bool = True
    max_sample_count: int = 1_000_000
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        max_sample_count = int(self.max_sample_count)
        if max_sample_count <= 0:
            raise ValueError("HeightmapImportRequest.max_sample_count must be positive.")
        object.__setattr__(self, "max_sample_count", max_sample_count)
        object.__setattr__(self, "embed_samples", bool(self.embed_samples))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class HeightmapImportDiagnostic:
    """Diagnostic for optional heightmap import dependency and payload checks."""

    code: str
    message: str
    supported: bool
    source_kind: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "supported": self.supported,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class HeightmapAlphaMaskPolicy:
    alpha_mode: Literal["mask", "ignore"]
    masked_sample_count: int
    total_sample_count: int

    @property
    def has_masked_samples(self) -> bool:
        return self.masked_sample_count > 0

    def canonical_payload(self) -> dict[str, object]:
        return {
            "alpha_mode": self.alpha_mode,
            "masked_sample_count": self.masked_sample_count,
            "total_sample_count": self.total_sample_count,
            "has_masked_samples": self.has_masked_samples,
        }


@dataclass(frozen=True)
class HeightmapCacheKeyRecord:
    cache_key: tuple | None
    reason: str

    @property
    def cacheable(self) -> bool:
        return self.cache_key is not None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "cacheable": self.cacheable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeightmapMeshCompatibilityResult:
    mesh: Mesh
    alpha_policy: HeightmapAlphaMaskPolicy
    cache_policy: HeightmapCacheKeyRecord
    boundary: Literal["explicit-mesh-compatibility"] = "explicit-mesh-compatibility"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "mesh_vertices": self.mesh.n_vertices,
            "mesh_faces": self.mesh.n_faces,
            "alpha_policy": self.alpha_policy.canonical_payload(),
            "cache_key_policy": self.cache_policy.canonical_payload(),
        }


@dataclass(frozen=True)
class HeightmapProjectionBoundsPolicy:
    projection: Literal["planar"]
    plane: Literal["xy", "xz", "yz"]
    bounds: tuple[float, float, float, float]
    source: Literal["explicit", "source-bounds"]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "projection": self.projection,
            "plane": self.plane,
            "bounds": self.bounds,
            "source": self.source,
        }


@dataclass(frozen=True)
class HeightmapSampleCoordinateRecord:
    projection: Literal["planar"]
    plane: Literal["xy", "xz", "yz"]
    bounds: tuple[float, float, float, float]
    u: np.ndarray
    v: np.ndarray
    u_normalized: np.ndarray
    v_normalized: np.ndarray

    def canonical_payload(self) -> dict[str, object]:
        return {
            "projection": self.projection,
            "plane": self.plane,
            "bounds": self.bounds,
            "sample_count": int(np.size(self.u_normalized)),
            "u_normalized_range": (
                float(np.min(self.u_normalized)) if self.u_normalized.size else 0.0,
                float(np.max(self.u_normalized)) if self.u_normalized.size else 0.0,
            ),
            "v_normalized_range": (
                float(np.min(self.v_normalized)) if self.v_normalized.size else 0.0,
                float(np.max(self.v_normalized)) if self.v_normalized.size else 0.0,
            ),
        }


@dataclass(frozen=True)
class HeightmapProjectionDomainRecord:
    """Projected domain facts for one heightmap patch used by preserving CSG."""

    patch_id: str
    projection: Literal["planar"]
    plane: Literal["xy", "xz", "yz"]
    bounds: tuple[float, float, float, float]
    sample_shape: tuple[int, int]
    sample_spacing: tuple[float, float]
    origin: tuple[float, float]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "projection": self.projection,
            "plane": self.plane,
            "bounds": self.bounds,
            "sample_shape": self.sample_shape,
            "sample_spacing": self.sample_spacing,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class HeightmapClippingRecord:
    """Projected overlap and sample-index clipping facts for two heightmap domains."""

    overlap_bounds: tuple[float, float, float, float]
    left_index_window: tuple[int, int, int, int]
    right_index_window: tuple[int, int, int, int]

    @property
    def has_overlap(self) -> bool:
        umin, umax, vmin, vmax = self.overlap_bounds
        return umax > umin and vmax > vmin

    def canonical_payload(self) -> dict[str, object]:
        return {
            "overlap_bounds": self.overlap_bounds,
            "left_index_window": self.left_index_window,
            "right_index_window": self.right_index_window,
            "has_overlap": self.has_overlap,
        }


@dataclass(frozen=True)
class HeightmapProjectionRefusalDiagnostic:
    """Deterministic refusal for heightmap-preserving CSG projection planning."""

    code: Literal["projection-mismatch", "disjoint-domain", "invalid-heightmap-domain"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapGridAlignmentRecord:
    """Heightmap-preserving CSG grid alignment plan."""

    supported: bool
    alignment: Literal["aligned", "resample-required", "refused"]
    left: HeightmapProjectionDomainRecord
    right: HeightmapProjectionDomainRecord
    clipping: HeightmapClippingRecord | None = None
    result_shape: tuple[int, int] | None = None
    resample_kernel: Literal["none", "bilinear"] = "none"
    diagnostics: tuple[HeightmapProjectionRefusalDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "alignment": self.alignment,
            "left": self.left.canonical_payload(),
            "right": self.right.canonical_payload(),
            "clipping": None if self.clipping is None else self.clipping.canonical_payload(),
            "result_shape": self.result_shape,
            "resample_kernel": self.resample_kernel,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class HeightmapEvaluationDiagnostic:
    code: str
    message: str
    sample: tuple[float, float]

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "sample": self.sample}


@dataclass(frozen=True)
class HeightmapMaskTessellationRecord:
    alpha_mode: Literal["mask", "ignore"]
    cell_count: int
    emitted_face_count: int
    skipped_cell_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "alpha_mode": self.alpha_mode,
            "cell_count": self.cell_count,
            "emitted_face_count": self.emitted_face_count,
            "skipped_cell_count": self.skipped_cell_count,
        }


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


def resolve_heightmap_alpha_mask_policy(
    mask: np.ndarray,
    *,
    alpha_mode: str,
) -> HeightmapAlphaMaskPolicy:
    mode = str(alpha_mode).lower()
    if mode not in {"mask", "ignore"}:
        raise ValueError("alpha_mode must be 'mask' or 'ignore'.")
    mask_arr = np.asarray(mask, dtype=bool)
    return HeightmapAlphaMaskPolicy(
        alpha_mode=mode,  # type: ignore[arg-type]
        masked_sample_count=int(np.size(mask_arr) - np.count_nonzero(mask_arr)),
        total_sample_count=int(np.size(mask_arr)),
    )


def heightmap_cache_key_record(
    image: str | Path | Image.Image | ArrayLike,
    height: float,
    xy_scale: float | Sequence[float],
    center: Sequence[float],
    alpha_mode: str,
    quality: MeshQuality | None,
) -> HeightmapCacheKeyRecord:
    cache_key = _heightmap_cache_key(image, height, xy_scale, center, alpha_mode, quality)
    if cache_key is None:
        reason = "uncacheable-source"
        if isinstance(image, (str, Path)):
            reason = "path-stat-unavailable"
        return HeightmapCacheKeyRecord(cache_key=None, reason=reason)
    return HeightmapCacheKeyRecord(cache_key=cache_key, reason="cache-key-valid")


def heightmap_mesh_compatibility_result(
    image: str | Path | Image.Image | ArrayLike,
    *,
    height: float = 1.0,
    xy_scale: float | Sequence[float] = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    alpha_mode: str = "mask",
    quality: MeshQuality | None = None,
) -> HeightmapMeshCompatibilityResult:
    """Create an explicit mesh-compatibility heightmap result.

    This helper keeps mesh heightfield generation visible as compatibility data
    instead of allowing callers to mistake the mesh for authored surface truth.
    """

    _, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            mask = mask[::2, ::2]
    alpha_policy = resolve_heightmap_alpha_mask_policy(mask, alpha_mode=alpha_mode)
    cache_policy = heightmap_cache_key_record(image, height, xy_scale, center, alpha_mode, quality)
    mesh = _heightmap_mesh_impl(
        image,
        height=height,
        xy_scale=xy_scale,
        center=center,
        alpha_mode=alpha_mode,
        quality=quality,
    )
    result = HeightmapMeshCompatibilityResult(mesh=mesh, alpha_policy=alpha_policy, cache_policy=cache_policy)
    mesh.metadata.update({"heightmap_mesh_compatibility": result.canonical_payload()})
    return result


def _as_planar_projection(projection: str) -> Literal["planar"]:
    projection_name = str(projection).lower()
    if projection_name != "planar":
        raise ValueError("Only planar projection is supported in this build.")
    return "planar"


def _as_projection_plane(plane: str) -> Literal["xy", "xz", "yz"]:
    plane_name = str(plane).lower()
    if plane_name not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'.")
    return plane_name  # type: ignore[return-value]


def _project_bounds_from_source(
    source_bounds: Sequence[float],
    plane: Literal["xy", "xz", "yz"],
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in np.asarray(source_bounds, dtype=float).ravel())
    if len(values) == 4:
        return values
    if len(values) != 6:
        raise ValueError("source_bounds must contain 4 projected values or 6 xyz bounds values.")
    xmin, xmax, ymin, ymax, zmin, zmax = values
    if plane == "xy":
        return xmin, xmax, ymin, ymax
    if plane == "xz":
        return xmin, xmax, zmin, zmax
    return ymin, ymax, zmin, zmax


def resolve_heightmap_projection_bounds_policy(
    *,
    projection: str = "planar",
    plane: str = "xy",
    bounds: Sequence[float] | None = None,
    source_bounds: Sequence[float] | None = None,
) -> HeightmapProjectionBoundsPolicy:
    projection_name = _as_planar_projection(projection)
    plane_name = _as_projection_plane(plane)
    source: Literal["explicit", "source-bounds"]
    if bounds is None:
        if source_bounds is None:
            raise ValueError("projection bounds are required when source bounds are unavailable.")
        resolved = _project_bounds_from_source(source_bounds, plane_name)
        source = "source-bounds"
    else:
        resolved = tuple(float(value) for value in np.asarray(bounds, dtype=float).reshape(4))
        source = "explicit"
    if not np.all(np.isfinite(np.asarray(resolved, dtype=float))):
        raise ValueError("projection bounds must be finite.")
    umin, umax, vmin, vmax = resolved
    if np.isclose(umax, umin) or np.isclose(vmax, vmin):
        raise ValueError("projection bounds are degenerate.")
    return HeightmapProjectionBoundsPolicy(projection_name, plane_name, resolved, source)


def heightmap_sample_coordinate_record(
    points: np.ndarray,
    policy: HeightmapProjectionBoundsPolicy,
) -> HeightmapSampleCoordinateRecord:
    point_arr = np.asarray(points, dtype=float)
    if point_arr.size == 0:
        point_arr = point_arr.reshape(0, 3)
    if point_arr.ndim != 2 or point_arr.shape[1] != 3:
        raise ValueError("heightmap sample coordinates require Nx3 points.")
    if policy.plane == "xy":
        u = point_arr[:, 0]
        v = point_arr[:, 1]
    elif policy.plane == "xz":
        u = point_arr[:, 0]
        v = point_arr[:, 2]
    else:
        u = point_arr[:, 1]
        v = point_arr[:, 2]
    umin, umax, vmin, vmax = policy.bounds
    u_norm = (u - umin) / (umax - umin)
    v_norm = (v - vmin) / (vmax - vmin)
    return HeightmapSampleCoordinateRecord(
        projection=policy.projection,
        plane=policy.plane,
        bounds=policy.bounds,
        u=u,
        v=v,
        u_normalized=u_norm,
        v_normalized=v_norm,
    )


def heightmap_projection_domain_record(
    patch: "HeightmapSurfacePatch",
    *,
    projection: str = "planar",
    plane: str = "xy",
) -> HeightmapProjectionDomainRecord:
    """Return projected domain facts for a native heightmap patch."""

    from .surface import HeightmapSurfacePatch

    if not isinstance(patch, HeightmapSurfacePatch):
        raise TypeError("heightmap projection planning requires a HeightmapSurfacePatch.")
    projection_name = _as_planar_projection(projection)
    plane_name = _as_projection_plane(plane)
    if plane_name != "xy":
        raise ValueError("HeightmapSurfacePatch preserves CSG only in its native xy projection plane.")
    rows, cols = patch.height_samples.shape
    sx, sy = patch.xy_scale
    half_width = (cols - 1) * sx * 0.5
    half_height = (rows - 1) * sy * 0.5
    xmin = float(patch.center[0] - half_width)
    xmax = float(patch.center[0] + half_width)
    ymin = float(patch.center[1] - half_height)
    ymax = float(patch.center[1] + half_height)
    return HeightmapProjectionDomainRecord(
        patch_id=patch.stable_identity,
        projection=projection_name,
        plane=plane_name,
        bounds=(xmin, xmax, ymin, ymax),
        sample_shape=(int(rows), int(cols)),
        sample_spacing=(float(sx), float(sy)),
        origin=(xmin, ymin),
    )


def _heightmap_overlap_bounds(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (max(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), min(left[3], right[3]))


def _heightmap_index_window(
    domain: HeightmapProjectionDomainRecord,
    overlap: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> tuple[int, int, int, int]:
    u0, u1, v0, v1 = overlap
    sx, sy = domain.sample_spacing
    ox, oy = domain.origin
    cols = domain.sample_shape[1]
    rows = domain.sample_shape[0]
    c0 = max(0, int(np.floor(((u0 - ox) / sx) + tolerance)))
    c1 = min(cols - 1, int(np.ceil(((u1 - ox) / sx) - tolerance)))
    r0 = max(0, int(np.floor(((v0 - oy) / sy) + tolerance)))
    r1 = min(rows - 1, int(np.ceil(((v1 - oy) / sy) - tolerance)))
    return (r0, r1, c0, c1)


def _heightmap_grid_lines_align(
    left: HeightmapProjectionDomainRecord,
    right: HeightmapProjectionDomainRecord,
    *,
    tolerance: float,
) -> bool:
    if not np.allclose(left.sample_spacing, right.sample_spacing, atol=tolerance):
        return False
    sx, sy = left.sample_spacing
    dx = abs((left.origin[0] - right.origin[0]) / sx)
    dy = abs((left.origin[1] - right.origin[1]) / sy)
    return abs(dx - round(dx)) <= tolerance and abs(dy - round(dy)) <= tolerance


def plan_heightmap_grid_alignment(
    left_patch: "HeightmapSurfacePatch",
    right_patch: "HeightmapSurfacePatch",
    *,
    left_projection: str = "planar",
    right_projection: str = "planar",
    left_plane: str = "xy",
    right_plane: str = "xy",
    tolerance: float = 1e-9,
) -> HeightmapGridAlignmentRecord:
    """Plan projection agreement, XY overlap, clipping, and grid alignment for heightmap CSG."""

    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("heightmap grid alignment tolerance must be finite and non-negative.")
    try:
        left = heightmap_projection_domain_record(left_patch, projection=left_projection, plane=left_plane)
        right = heightmap_projection_domain_record(right_patch, projection=right_projection, plane=right_plane)
    except Exception as exc:
        fallback_left = heightmap_projection_domain_record(left_patch)
        fallback_right = heightmap_projection_domain_record(right_patch)
        return HeightmapGridAlignmentRecord(
            supported=False,
            alignment="refused",
            left=fallback_left,
            right=fallback_right,
            diagnostics=(
                HeightmapProjectionRefusalDiagnostic(
                    code="projection-mismatch",
                    message=f"Heightmap projection planning refused: {exc}; no mesh fallback was attempted.",
                ),
            ),
        )
    if left.projection != right.projection or left.plane != right.plane:
        return HeightmapGridAlignmentRecord(
            supported=False,
            alignment="refused",
            left=left,
            right=right,
            diagnostics=(
                HeightmapProjectionRefusalDiagnostic(
                    code="projection-mismatch",
                    message="Heightmap projection frames do not match; no mesh fallback was attempted.",
                ),
            ),
        )
    overlap = _heightmap_overlap_bounds(left.bounds, right.bounds)
    clipping = HeightmapClippingRecord(
        overlap_bounds=overlap,
        left_index_window=_heightmap_index_window(left, overlap, tolerance=tol),
        right_index_window=_heightmap_index_window(right, overlap, tolerance=tol),
    )
    if not clipping.has_overlap:
        return HeightmapGridAlignmentRecord(
            supported=False,
            alignment="refused",
            left=left,
            right=right,
            clipping=clipping,
            diagnostics=(
                HeightmapProjectionRefusalDiagnostic(
                    code="disjoint-domain",
                    message="Heightmap projected XY domains are disjoint; no mesh fallback was attempted.",
                ),
            ),
        )
    aligned = _heightmap_grid_lines_align(left, right, tolerance=tol)
    if aligned:
        result_shape = (
            max(1, clipping.left_index_window[1] - clipping.left_index_window[0] + 1),
            max(1, clipping.left_index_window[3] - clipping.left_index_window[2] + 1),
        )
        return HeightmapGridAlignmentRecord(
            supported=True,
            alignment="aligned",
            left=left,
            right=right,
            clipping=clipping,
            result_shape=result_shape,
            resample_kernel="none",
        )
    min_sx = min(left.sample_spacing[0], right.sample_spacing[0])
    min_sy = min(left.sample_spacing[1], right.sample_spacing[1])
    result_shape = (
        int(np.floor((overlap[3] - overlap[2]) / min_sy + tol)) + 1,
        int(np.floor((overlap[1] - overlap[0]) / min_sx + tol)) + 1,
    )
    return HeightmapGridAlignmentRecord(
        supported=True,
        alignment="resample-required",
        left=left,
        right=right,
        clipping=clipping,
        result_shape=(max(2, result_shape[0]), max(2, result_shape[1])),
        resample_kernel="bilinear",
    )


def estimate_heightmap_normal(
    patch: "HeightmapSurfacePatch",
    u: float,
    v: float,
) -> tuple[np.ndarray, HeightmapEvaluationDiagnostic]:
    du, dv = patch.derivatives_at(float(u), float(v))
    normal = np.cross(du, dv)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        diagnostic = HeightmapEvaluationDiagnostic(
            code="degenerate-heightmap-normal",
            message="Heightmap normal estimate has zero length at the requested sample.",
            sample=(float(u), float(v)),
        )
        return np.array([0.0, 0.0, 1.0], dtype=float), diagnostic
    return normal / norm, HeightmapEvaluationDiagnostic(
        code="heightmap-normal-estimated",
        message="Heightmap normal estimated from finite surface derivatives.",
        sample=(float(u), float(v)),
    )


def heightmap_mask_tessellation_record(patch: "HeightmapSurfacePatch") -> HeightmapMaskTessellationRecord:
    rows, cols = patch.height_samples.shape
    cell_count = int(max(rows - 1, 0) * max(cols - 1, 0))
    skipped = 0
    if patch.alpha_mode == "mask":
        for row in range(rows - 1):
            for col in range(cols - 1):
                if not (
                    patch.alpha_mask[row, col]
                    and patch.alpha_mask[row, col + 1]
                    and patch.alpha_mask[row + 1, col]
                    and patch.alpha_mask[row + 1, col + 1]
                ):
                    skipped += 1
    emitted = (cell_count - skipped) * 2
    return HeightmapMaskTessellationRecord(
        alpha_mode=patch.alpha_mode,
        cell_count=cell_count,
        emitted_face_count=int(emitted),
        skipped_cell_count=int(skipped),
    )


def validate_heightmap_finite_samples(samples: ArrayLike) -> np.ndarray:
    """Normalize embedded height samples and refuse non-finite grids."""

    grid = np.asarray(samples, dtype=float)
    if grid.ndim != 2:
        raise ValueError("Heightmap finite-grid samples must be a 2D array.")
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ValueError("Heightmap finite-grid samples must be at least 2x2.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("Heightmap finite-grid samples must be finite.")
    return grid


def build_heightmap_mask_no_data_diagnostic(
    request: HeightmapAuthoringRequest,
) -> HeightmapNoDataDiagnostic:
    """Return an explicit mask/no-data diagnostic for a native heightmap grid."""

    mask = np.asarray(request.alpha_mask, dtype=bool)
    total = int(mask.size)
    masked = int(total - np.count_nonzero(mask))
    if request.alpha_mode == "mask" and masked == total:
        return HeightmapNoDataDiagnostic(
            code="heightmap-all-samples-masked",
            message="Heightmap native grid masks every sample and cannot emit visible sampled surface cells.",
            valid=False,
            masked_sample_count=masked,
            total_sample_count=total,
        )
    return HeightmapNoDataDiagnostic(
        code="heightmap-mask-valid",
        message="Heightmap native grid mask is valid for finite-grid authoring.",
        valid=True,
        masked_sample_count=masked,
        total_sample_count=total,
    )


def heightmap_sample_grid_provenance_record(
    request: HeightmapAuthoringRequest,
) -> HeightmapSampleGridProvenanceRecord:
    """Return producer provenance for an embedded heightmap sample grid."""

    mask = np.asarray(request.alpha_mask, dtype=bool)
    total = int(mask.size)
    return HeightmapSampleGridProvenanceRecord(
        family="heightmap",
        operation="heightmap-finite-grid-authoring",
        sample_shape=tuple(int(value) for value in request.height_samples.shape),
        masked_sample_count=int(total - np.count_nonzero(mask)),
        total_sample_count=total,
    )


def make_heightmap_surface_from_grid(request: HeightmapAuthoringRequest) -> "SurfaceBody":
    """Create a surface body from an embedded finite heightmap grid."""

    diagnostic = build_heightmap_mask_no_data_diagnostic(request)
    if not diagnostic.valid:
        raise ValueError(diagnostic.message)
    from .surface import HeightmapSurfacePatch, make_surface_body, make_surface_shell

    provenance = heightmap_sample_grid_provenance_record(request).canonical_payload()
    kernel_metadata = {
        "producer": "heightmap",
        "operation": "heightmap-finite-grid-authoring",
        "sample_shape": tuple(int(value) for value in request.height_samples.shape),
    }
    kernel_metadata.update({"producer_provenance": provenance, **dict(request.metadata.get("kernel", {}))})
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=request.height_samples,
        alpha_mask=request.alpha_mask,
        alpha_mode=request.alpha_mode,
        xy_scale=request.xy_scale,
        center=request.center,
        height_scale=request.height_scale,
        metadata={"kernel": kernel_metadata},
    )
    return make_surface_body(
        (make_surface_shell((patch,), connected=False, metadata={"kernel": {"surface_family": "heightmap", "producer_provenance": provenance}}),),
        metadata={"kernel": {"surface_family": "heightmap", "authoring_boundary": "surface-native", "producer_provenance": provenance}},
    )


def heightmap_import_dependency_boundary() -> HeightmapImportDiagnostic:
    """Report optional image import availability without making it required for native grids."""

    supported = Image is not None
    return HeightmapImportDiagnostic(
        code="heightmap-import-dependency-available" if supported else "heightmap-import-dependency-unavailable",
        message=(
            "Heightmap optional import adapter can load images and arrays into embedded grids."
            if supported
            else "Heightmap optional import adapter is unavailable; use embedded finite grids."
        ),
        supported=supported,
        source_kind="dependency",
    )


def build_heightmap_import_diagnostic(request: HeightmapImportRequest) -> HeightmapImportDiagnostic:
    """Inspect a heightmap import request before producing surface truth."""

    source_kind = type(request.source).__name__
    if not request.embed_samples:
        return HeightmapImportDiagnostic(
            code="heightmap-import-external-reference-refused",
            message="Heightmap imports must embed finite samples; external references are not surface truth.",
            supported=False,
            source_kind=source_kind,
        )
    dependency = heightmap_import_dependency_boundary()
    if not dependency.supported:
        return dependency
    try:
        heights, _mask = _load_heightmap(request.source)
        sample_count = int(heights.size)
        if sample_count > request.max_sample_count:
            raise ValueError(
                f"heightmap import sample count {sample_count} exceeds max_sample_count={request.max_sample_count}"
            )
        validate_heightmap_finite_samples(heights)
    except Exception as exc:
        return HeightmapImportDiagnostic(
            code="heightmap-import-invalid-payload",
            message=f"Heightmap import payload is invalid: {exc}",
            supported=False,
            source_kind=source_kind,
        )
    return HeightmapImportDiagnostic(
        code="heightmap-import-supported",
        message="Heightmap import can be normalized into an embedded finite grid.",
        supported=True,
        source_kind=source_kind,
    )


def import_heightmap_surface(request: HeightmapImportRequest) -> "SurfaceBody":
    """Import an image or array into an embedded native heightmap surface body."""

    diagnostic = build_heightmap_import_diagnostic(request)
    if not diagnostic.supported:
        raise ValueError(diagnostic.message)
    heights, mask = _load_heightmap(request.source)
    if request.quality is not None:
        quality = apply_lod(request.quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    metadata = dict(request.metadata)
    kernel = dict(metadata.get("kernel", {})) if isinstance(metadata.get("kernel"), dict) else {}
    kernel.update({"operation": "heightmap-import", "import_source_kind": diagnostic.source_kind})
    metadata["kernel"] = kernel
    return make_heightmap_surface_from_grid(
        HeightmapAuthoringRequest(
            height_samples=heights,
            alpha_mask=mask,
            alpha_mode=request.alpha_mode,
            xy_scale=request.xy_scale,
            center=request.center,
            height_scale=request.height,
            metadata=metadata,
        )
    )


def _triangle_surface_body_from_mesh(mesh: Mesh, *, metadata: dict[str, object] | None = None) -> "SurfaceBody":
    from .surface import ParameterDomain, PlanarSurfacePatch, TrimLoop, make_surface_body, make_surface_shell

    if mesh.n_faces == 0:
        from ._surface_primitives import make_surface_box

        return make_surface_box(size=(1e-6, 1e-6, 1e-6), metadata={"consumer": {"hidden_placeholder": True}})

    patches = []
    for face_index, tri in enumerate(mesh.faces):
        points = np.asarray(mesh.vertices[np.asarray(tri, dtype=int)], dtype=float).reshape(3, 3)
        origin = points[0]
        u_seed = points[1] - origin
        u_norm = float(np.linalg.norm(u_seed))
        if u_norm == 0.0:
            continue
        u_dir = u_seed / u_norm
        normal = np.cross(points[1] - origin, points[2] - origin)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm == 0.0:
            continue
        normal = normal / normal_norm
        v_dir = np.cross(normal, u_dir)
        v_dir = v_dir / float(np.linalg.norm(v_dir))
        uv_points = np.asarray(
            [
                (
                    float(np.dot(point - origin, u_dir)),
                    float(np.dot(point - origin, v_dir)),
                )
                for point in points
            ],
            dtype=float,
        )
        xmin, ymin = uv_points.min(axis=0)
        xmax, ymax = uv_points.max(axis=0)
        if np.isclose(xmax, xmin):
            xmax = xmin + 1e-9
        if np.isclose(ymax, ymin):
            ymax = ymin + 1e-9
        patches.append(
            PlanarSurfacePatch(
                family="planar",
                domain=ParameterDomain((xmin, xmax), (ymin, ymax)),
                trim_loops=(TrimLoop(uv_points, category="outer"),),
                origin=origin,
                u_axis=u_dir,
                v_axis=v_dir,
                metadata={"kernel": {"producer": "heightmap", "triangle_face_index": face_index}},
            )
        )
    return make_surface_body(
        (make_surface_shell(tuple(patches), connected=False, metadata={"kernel": {"producer": "heightmap"}}),),
        metadata=metadata,
    )


def _heightmap_mesh_impl(
    image: str | Path | Image.Image | ArrayLike,
    *,
    height: float,
    xy_scale: float | Sequence[float],
    center: Sequence[float],
    alpha_mode: str,
    quality: MeshQuality | None,
) -> Mesh:
    cache_record = heightmap_cache_key_record(image, height, xy_scale, center, alpha_mode, quality)
    cached = _HEIGHTMAP_CACHE.get(cache_record.cache_key) if cache_record.cache_key is not None else None
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
    mesh.metadata.update({"heightmap_cache_key_policy": cache_record.canonical_payload()})
    if cache_record.cache_key is not None:
        _HEIGHTMAP_CACHE.set(cache_record.cache_key, mesh.copy())
    return mesh


def make_heightmap_surface_patch(
    image: str | Path | Image.Image | ArrayLike,
    *,
    height: float = 1.0,
    xy_scale: float | Sequence[float] = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    alpha_mode: str = "mask",
    quality: MeshQuality | None = None,
):
    from .surface import HeightmapSurfacePatch

    heights, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    if heights.shape[0] < 2 or heights.shape[1] < 2:
        raise ValueError("Heightmap surface payload requires at least a 2x2 sample grid.")
    alpha_policy = resolve_heightmap_alpha_mask_policy(mask, alpha_mode=alpha_mode)
    cache_record = heightmap_cache_key_record(image, height, xy_scale, center, alpha_mode, quality)
    if alpha_mode == "ignore":
        heights = np.where(mask, heights, 0.0)
    elif alpha_mode == "mask":
        heights = np.where(mask, heights, 0.0)
    else:
        raise ValueError("alpha_mode must be 'mask' or 'ignore'.")
    return HeightmapSurfacePatch(
        family="heightmap",
        height_samples=heights,
        alpha_mask=mask,
        alpha_mode=alpha_policy.alpha_mode,
        xy_scale=_as_scale(xy_scale),
        center=np.asarray(center, dtype=float).reshape(3),
        height_scale=float(height),
        metadata={
            "kernel": {
                "producer": "heightmap",
                "sample_shape": tuple(int(value) for value in heights.shape),
                "alpha_policy": alpha_policy.canonical_payload(),
                "cache_key_policy": cache_record.canonical_payload(),
            }
        },
    )


def heightmap(
    image: str | Path | Image.Image | ArrayLike,
    height: float = 1.0,
    xy_scale: float | Sequence[float] = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    alpha_mode: str = "mask",
    quality: MeshQuality | None = None,
    backend: Backend = "mesh",
) -> Mesh | SurfaceBody:
    """Create a heightfield mesh from an image.

    alpha_mode:
        - "mask": skip faces that touch fully transparent pixels (holes).
        - "ignore": treat transparent pixels as zero height (no holes).
    """
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        patch = make_heightmap_surface_patch(
            image,
            height=height,
            xy_scale=xy_scale,
            center=center,
            alpha_mode=alpha_mode,
            quality=quality,
        )
        from .surface import make_surface_body, make_surface_shell

        return make_surface_body(
            (make_surface_shell((patch,), connected=False, metadata={"kernel": {"producer": "heightmap"}}),),
            metadata={"kernel": {"producer": "heightmap"}, "consumer": {"source": "heightmap"}},
        )

    warn_mesh_primary_api(
        "heightmap",
        replacement="a future surface-native heightfield path",
    )
    return heightmap_mesh_compatibility_result(
        image,
        height=height,
        xy_scale=xy_scale,
        center=center,
        alpha_mode=alpha_mode,
        quality=quality,
    ).mesh


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
    mesh: Mesh | SurfaceBody,
    image: str | Path | Image.Image | ArrayLike,
    height: float = 1.0,
    projection: str = "planar",
    plane: str = "xy",
    direction: str | Sequence[float] = "normal",
    alpha_mode: str = "ignore",
    bounds: Sequence[float] | None = None,
    quality: MeshQuality | None = None,
    backend: Backend = "mesh",
) -> Mesh | SurfaceBody:
    """Displace a mesh using a heightmap with planar projection.

    alpha_mode:
        - "ignore": transparent pixels cause no displacement.
        - "mask": faces touching transparent samples are removed.
    """
    if backend not in {"mesh", "surface"}:
        raise ValueError("backend must be 'mesh' or 'surface'.")
    if backend == "surface":
        if isinstance(mesh, Mesh):
            raise ValueError("Surface displacement requires a SurfaceBody input; use backend='mesh' for Mesh inputs.")
        return _displace_heightmap_surface_body(
            mesh,
            image=image,
            height=height,
            projection=projection,
            plane=plane,
            direction=direction,
            alpha_mode=alpha_mode,
            bounds=bounds,
            quality=quality,
        )

    warn_mesh_primary_api(
        "displace_heightmap",
        replacement="surface-native displacement once SurfaceBody deformation lands",
    )
    return _displace_heightmap_mesh_impl(
        mesh,
        image=image,
        height=height,
        projection=projection,
        plane=plane,
        direction=direction,
        alpha_mode=alpha_mode,
        bounds=bounds,
        quality=quality,
    )


def _displace_heightmap_surface_body(
    body: "SurfaceBody",
    *,
    image: str | Path | Image.Image | ArrayLike,
    height: float,
    projection: str,
    plane: str,
    direction: str | Sequence[float],
    alpha_mode: str,
    bounds: Sequence[float] | None,
    quality: MeshQuality | None,
) -> "SurfaceBody":
    projection_name = _as_planar_projection(projection)
    plane_name = _as_projection_plane(plane)
    heights, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    if heights.shape[0] < 2 or heights.shape[1] < 2:
        raise ValueError("Surface displacement requires at least a 2x2 heightmap sample grid.")
    alpha_policy = resolve_heightmap_alpha_mask_policy(mask, alpha_mode=alpha_mode)
    cache_record = heightmap_cache_key_record(image, height, 1.0, (0.0, 0.0, 0.0), alpha_mode, quality)
    if alpha_policy.alpha_mode == "ignore":
        heights = np.where(mask, heights, 0.0)
    else:
        heights = np.where(mask, heights, 0.0)

    from .surface import DisplacementSurfacePatch, make_surface_body, make_surface_shell

    displaced_shells = []
    for shell in body.iter_shells(world=True):
        patches = []
        for patch in shell.iter_patches(world=True):
            policy = resolve_heightmap_projection_bounds_policy(
                projection=projection_name,
                plane=plane_name,
                bounds=bounds,
                source_bounds=_surface_patch_projection_source_bounds(patch, plane_name) if bounds is None else None,
            )
            patches.append(
                DisplacementSurfacePatch(
                    family="displacement",
                    source_patch=patch,
                    displacement_samples=heights,
                    alpha_mask=mask,
                    alpha_mode=alpha_policy.alpha_mode,
                    height_scale=float(height),
                    direction=direction,
                    projection=policy.projection,
                    plane=policy.plane,
                    projection_bounds=policy.bounds,
                    metadata={
                        "kernel": {
                            "producer": "heightmap",
                            "operation": "displace",
                            "projection_policy": policy.canonical_payload(),
                            "alpha_policy": alpha_policy.canonical_payload(),
                            "cache_key_policy": cache_record.canonical_payload(),
                        }
                    },
                )
            )
        displaced_shells.append(
            make_surface_shell(
                tuple(patches),
                connected=shell.connected,
                metadata={"kernel": {"producer": "heightmap", "operation": "displace"}},
            )
        )
    return make_surface_body(
        tuple(displaced_shells),
        metadata={"kernel": {"producer": "heightmap", "operation": "displace"}, "consumer": {"source": "heightmap"}},
    )


def _surface_patch_projection_source_bounds(
    patch: object,
    plane: Literal["xy", "xz", "yz"],
) -> tuple[float, float, float, float]:
    domain = getattr(patch, "domain")
    u0, u1 = domain.u_range
    v0, v1 = domain.v_range
    points = np.asarray(
        [
            patch.point_at(u0, v0),
            patch.point_at(u1, v0),
            patch.point_at(u1, v1),
            patch.point_at(u0, v1),
        ],
        dtype=float,
    )
    coord_record = heightmap_sample_coordinate_record(
        points,
        HeightmapProjectionBoundsPolicy("planar", plane, (0.0, 1.0, 0.0, 1.0), "source-bounds"),
    )
    return (
        float(np.min(coord_record.u)),
        float(np.max(coord_record.u)),
        float(np.min(coord_record.v)),
        float(np.max(coord_record.v)),
    )


def _displace_heightmap_mesh_impl(
    mesh: Mesh,
    *,
    image: str | Path | Image.Image | ArrayLike,
    height: float,
    projection: str,
    plane: str,
    direction: str | Sequence[float],
    alpha_mode: str,
    bounds: Sequence[float] | None,
    quality: MeshQuality | None,
) -> Mesh:
    heights, mask = _load_heightmap(image)
    if quality is not None:
        quality = apply_lod(quality)
        if quality.lod == "preview":
            heights = heights[::2, ::2]
            mask = mask[::2, ::2]
    if heights.size == 0 or mesh.n_vertices == 0:
        return mesh.copy()

    verts = mesh.vertices.copy()
    policy = resolve_heightmap_projection_bounds_policy(
        projection=projection,
        plane=plane,
        bounds=bounds,
        source_bounds=mesh.bounds if bounds is None else None,
    )
    coord_record = heightmap_sample_coordinate_record(verts, policy)

    sampled, masked = _sample_heightmap(heights, mask, coord_record.u_normalized, coord_record.v_normalized)
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


__all__ = [
    "HeightmapAlphaMaskPolicy",
    "HeightmapAuthoringRequest",
    "HeightmapCacheKeyRecord",
    "HeightmapMeshCompatibilityResult",
    "HeightmapEvaluationDiagnostic",
    "HeightmapImportDiagnostic",
    "HeightmapImportRequest",
    "HeightmapMaskTessellationRecord",
    "HeightmapNoDataDiagnostic",
    "HeightmapClippingRecord",
    "HeightmapGridAlignmentRecord",
    "HeightmapProjectionDomainRecord",
    "HeightmapProjectionRefusalDiagnostic",
    "HeightmapProjectionBoundsPolicy",
    "HeightmapSampleGridProvenanceRecord",
    "HeightmapSampleCoordinateRecord",
    "build_heightmap_mask_no_data_diagnostic",
    "build_heightmap_import_diagnostic",
    "heightmap",
    "displace_heightmap",
    "heightmap_cache_key_record",
    "heightmap_import_dependency_boundary",
    "heightmap_mesh_compatibility_result",
    "heightmap_projection_domain_record",
    "estimate_heightmap_normal",
    "heightmap_mask_tessellation_record",
    "heightmap_sample_grid_provenance_record",
    "heightmap_sample_coordinate_record",
    "import_heightmap_surface",
    "make_heightmap_surface_from_grid",
    "make_heightmap_surface_patch",
    "plan_heightmap_grid_alignment",
    "resolve_heightmap_projection_bounds_policy",
    "resolve_heightmap_alpha_mask_policy",
    "validate_heightmap_finite_samples",
]
