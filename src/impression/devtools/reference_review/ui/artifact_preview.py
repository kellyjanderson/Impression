"""Service-owned artifact preview generation for the review workbench."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

_RENDER_CONTRACT_VERSION = "stl-preview-v6"
_PREVIEW_BACKGROUND_COLOR = "#071426"
_PREVIEW_OBJECT_COLOR = "#ffb56b"
_PREVIEW_EDGE_COLOR = "#3d210f"


@dataclass(frozen=True)
class PreviewCameraState:
    azimuth_deg: float = 45.0
    elevation_deg: float = 28.0
    roll_deg: float = 0.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def normalized(self) -> "PreviewCameraState":
        return PreviewCameraState(
            azimuth_deg=self.azimuth_deg % 360.0,
            elevation_deg=self.elevation_deg % 360.0,
            roll_deg=self.roll_deg % 360.0,
            zoom=max(0.2, min(6.0, self.zoom)),
            pan_x=max(-5.0, min(5.0, self.pan_x)),
            pan_y=max(-5.0, min(5.0, self.pan_y)),
        )


@dataclass(frozen=True)
class ArtifactPreviewRecord:
    artifact_path: Path
    preview_path: Path | None
    diagnostic: str | None = None

    @property
    def preview_url(self) -> str:
        if self.preview_path is None:
            return ""
        return self.preview_path.resolve().as_uri()


@dataclass(frozen=True)
class _LoadedPreviewScene:
    artifact_key: tuple[Path, int, int]
    mesh: object
    feature_edges: object


class ArtifactPreviewRenderer:
    """Persistent PyVista renderer for a preview surface/session."""

    def __init__(self, *, cache_root: Path) -> None:
        self._cache_root = cache_root
        self._pv = None
        self._plotter = None
        self._window_size: tuple[int, int] | None = None
        self._scene: _LoadedPreviewScene | None = None

    def render(
        self,
        artifact_path: Path,
        *,
        window_size: tuple[int, int] = (720, 520),
        camera: PreviewCameraState | None = None,
    ) -> ArtifactPreviewRecord:
        artifact_path = Path(artifact_path)
        if artifact_path.suffix.lower() not in {".stl", ".impress"}:
            return ArtifactPreviewRecord(artifact_path, None, "unsupported-artifact-kind")
        if not artifact_path.is_file():
            return ArtifactPreviewRecord(artifact_path, None, "missing-artifact")
        camera = (camera or PreviewCameraState()).normalized()
        self._cache_root.mkdir(parents=True, exist_ok=True)
        preview_path = self._cache_root / f"{_preview_cache_key(artifact_path, window_size, camera)}.png"
        if preview_path.is_file():
            return ArtifactPreviewRecord(artifact_path, preview_path)
        try:
            pv = self._ensure_pyvista()
            scene = self._ensure_scene(artifact_path, pv)
            plotter = self._ensure_plotter(pv, window_size)
            _apply_camera(plotter, scene.mesh.bounds, camera)
            plotter.screenshot(str(preview_path), return_img=False)
        except Exception as exc:
            return ArtifactPreviewRecord(
                artifact_path,
                None,
                f"artifact-preview-failed:{exc.__class__.__name__}",
            )
        if not preview_path.is_file():
            return ArtifactPreviewRecord(artifact_path, None, "artifact-preview-missing-output")
        return ArtifactPreviewRecord(artifact_path, preview_path)

    def close(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        self._plotter = None
        self._scene = None
        self._window_size = None

    def _ensure_pyvista(self):
        if self._pv is None:
            import pyvista as pv

            self._pv = pv
        return self._pv

    def _ensure_plotter(self, pv, window_size: tuple[int, int]):
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=True, window_size=window_size)
            self._plotter.set_background(_PREVIEW_BACKGROUND_COLOR)
            self._window_size = window_size
            self._load_scene_into_plotter()
        elif self._window_size != window_size:
            self._plotter.window_size = window_size
            self._window_size = window_size
        return self._plotter

    def _ensure_scene(self, artifact_path: Path, pv) -> _LoadedPreviewScene:
        stat = artifact_path.stat()
        artifact_key = (artifact_path.resolve(), stat.st_mtime_ns, stat.st_size)
        if self._scene is not None and self._scene.artifact_key == artifact_key:
            return self._scene
        mesh = _preview_mesh_for_artifact(artifact_path, pv)
        self._scene = _LoadedPreviewScene(artifact_key, mesh, _object_feature_edges(mesh))
        if self._plotter is not None:
            self._load_scene_into_plotter()
        return self._scene

    def _load_scene_into_plotter(self) -> None:
        if self._plotter is None or self._scene is None:
            return
        self._plotter.clear()
        self._plotter.set_background(_PREVIEW_BACKGROUND_COLOR)
        self._plotter.add_mesh(
            self._scene.mesh,
            color=_PREVIEW_OBJECT_COLOR,
            smooth_shading=False,
            show_edges=False,
        )
        if self._scene.feature_edges.n_cells > 0:
            self._plotter.add_mesh(self._scene.feature_edges, color=_PREVIEW_EDGE_COLOR, line_width=2)


def render_stl_preview(
    artifact_path: Path,
    *,
    cache_root: Path,
    window_size: tuple[int, int] = (720, 520),
    camera: PreviewCameraState | None = None,
) -> ArtifactPreviewRecord:
    """Render an STL artifact to a cached PNG preview."""

    renderer = ArtifactPreviewRenderer(cache_root=cache_root)
    try:
        return renderer.render(artifact_path, window_size=window_size, camera=camera)
    finally:
        renderer.close()


def _preview_mesh_for_artifact(artifact_path: Path, pv):
    if artifact_path.suffix.lower() == ".stl":
        return pv.read(artifact_path)
    from impression.io import load_impress
    from impression.mesh import combine_meshes, mesh_to_pyvista
    from impression.modeling import preview_tessellation_request, tessellate_surface_body

    loaded = load_impress(artifact_path)
    meshes = [
        tessellate_surface_body(body, preview_tessellation_request(require_watertight=False)).mesh
        for body in loaded.bodies
    ]
    if not meshes:
        raise ValueError("empty-impress-artifact")
    return mesh_to_pyvista(meshes[0] if len(meshes) == 1 else combine_meshes(meshes))


def _object_feature_edges(mesh):
    return mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        non_manifold_edges=True,
        manifold_edges=False,
        feature_angle=30.0,
    )


def _apply_camera(plotter, bounds, camera: PreviewCameraState) -> None:
    x_center = (bounds[0] + bounds[1]) / 2.0
    y_center = (bounds[2] + bounds[3]) / 2.0
    z_center = (bounds[4] + bounds[5]) / 2.0
    span = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )
    azimuth = math.radians(camera.azimuth_deg)
    elevation = math.radians(camera.elevation_deg)
    distance = span * 1.95 / camera.zoom
    pan_scale = span * 0.35
    focal_point = (
        x_center + camera.pan_x * pan_scale,
        y_center,
        z_center + camera.pan_y * pan_scale,
    )
    position = (
        focal_point[0] + distance * math.cos(elevation) * math.cos(azimuth),
        focal_point[1] + distance * math.cos(elevation) * math.sin(azimuth),
        focal_point[2] + distance * math.sin(elevation),
    )
    up = _camera_up_vector(camera.azimuth_deg, camera.elevation_deg, camera.roll_deg)
    plotter.camera_position = [position, focal_point, up]


def _camera_up_vector(
    azimuth_deg: float,
    elevation_deg: float,
    roll_deg: float,
) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    axis = (
        -math.cos(elevation) * math.cos(azimuth),
        -math.cos(elevation) * math.sin(azimuth),
        -math.sin(elevation),
    )
    base_up = (
        -math.sin(elevation) * math.cos(azimuth),
        -math.sin(elevation) * math.sin(azimuth),
        math.cos(elevation),
    )
    if roll_deg == 0.0:
        return base_up
    ux, uy, uz = axis
    x, y, z = base_up
    angle = math.radians(roll_deg)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    dot = ux * x + uy * y + uz * z
    cross = (
        uy * z - uz * y,
        uz * x - ux * z,
        ux * y - uy * x,
    )
    return (
        x * cos_angle + cross[0] * sin_angle + ux * dot * (1.0 - cos_angle),
        y * cos_angle + cross[1] * sin_angle + uy * dot * (1.0 - cos_angle),
        z * cos_angle + cross[2] * sin_angle + uz * dot * (1.0 - cos_angle),
    )


def _preview_cache_key(
    artifact_path: Path,
    window_size: tuple[int, int],
    camera: PreviewCameraState,
) -> str:
    stat = artifact_path.stat()
    payload = "|".join(
        (
            str(artifact_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            f"{window_size[0]}x{window_size[1]}",
            f"{camera.azimuth_deg:.2f}",
            f"{camera.elevation_deg:.2f}",
            f"{camera.roll_deg:.2f}",
            f"{camera.zoom:.3f}",
            f"{camera.pan_x:.3f}",
            f"{camera.pan_y:.3f}",
            _RENDER_CONTRACT_VERSION,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
