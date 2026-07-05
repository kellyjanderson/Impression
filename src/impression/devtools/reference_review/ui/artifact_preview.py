"""Service-owned artifact preview generation for the review workbench."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

_RENDER_CONTRACT_VERSION = "stl-preview-v3"


@dataclass(frozen=True)
class PreviewCameraState:
    azimuth_deg: float = 45.0
    elevation_deg: float = 28.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def normalized(self) -> "PreviewCameraState":
        return PreviewCameraState(
            azimuth_deg=self.azimuth_deg % 360.0,
            elevation_deg=max(-85.0, min(85.0, self.elevation_deg)),
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


def render_stl_preview(
    artifact_path: Path,
    *,
    cache_root: Path,
    window_size: tuple[int, int] = (720, 520),
    camera: PreviewCameraState | None = None,
) -> ArtifactPreviewRecord:
    """Render an STL artifact to a cached PNG preview."""

    artifact_path = Path(artifact_path)
    if artifact_path.suffix.lower() != ".stl":
        return ArtifactPreviewRecord(artifact_path, None, "unsupported-artifact-kind")
    if not artifact_path.is_file():
        return ArtifactPreviewRecord(artifact_path, None, "missing-artifact")
    camera = (camera or PreviewCameraState()).normalized()
    cache_root.mkdir(parents=True, exist_ok=True)
    preview_path = cache_root / f"{_preview_cache_key(artifact_path, window_size, camera)}.png"
    if preview_path.is_file():
        return ArtifactPreviewRecord(artifact_path, preview_path)
    try:
        import pyvista as pv

        mesh = pv.read(artifact_path)
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        plotter.add_mesh(mesh, color="#5b84b1", smooth_shading=False, show_edges=True)
        _apply_camera(plotter, mesh.bounds, camera)
        plotter.show(screenshot=str(preview_path), auto_close=True, interactive=False)
    except Exception as exc:
        return ArtifactPreviewRecord(artifact_path, None, f"artifact-preview-failed:{exc.__class__.__name__}")
    if not preview_path.is_file():
        return ArtifactPreviewRecord(artifact_path, None, "artifact-preview-missing-output")
    return ArtifactPreviewRecord(artifact_path, preview_path)


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
    distance = span * 2.8 / camera.zoom
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
    plotter.camera_position = [position, focal_point, (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.45)


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
            f"{camera.zoom:.3f}",
            f"{camera.pan_x:.3f}",
            f"{camera.pan_y:.3f}",
            _RENDER_CONTRACT_VERSION,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
