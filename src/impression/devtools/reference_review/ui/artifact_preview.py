"""Service-owned artifact preview generation for the review workbench."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

_RENDER_CONTRACT_VERSION = "stl-preview-v2"


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
) -> ArtifactPreviewRecord:
    """Render an STL artifact to a cached PNG preview."""

    artifact_path = Path(artifact_path)
    if artifact_path.suffix.lower() != ".stl":
        return ArtifactPreviewRecord(artifact_path, None, "unsupported-artifact-kind")
    if not artifact_path.is_file():
        return ArtifactPreviewRecord(artifact_path, None, "missing-artifact")
    cache_root.mkdir(parents=True, exist_ok=True)
    preview_path = cache_root / f"{_preview_cache_key(artifact_path, window_size)}.png"
    if preview_path.is_file():
        return ArtifactPreviewRecord(artifact_path, preview_path)
    try:
        import pyvista as pv

        mesh = pv.read(artifact_path)
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        plotter.add_mesh(mesh, color="#5b84b1", smooth_shading=False, show_edges=True)
        plotter.reset_camera()
        plotter.camera.zoom(1.45)
        plotter.show(screenshot=str(preview_path), auto_close=True, interactive=False)
    except Exception as exc:
        return ArtifactPreviewRecord(artifact_path, None, f"artifact-preview-failed:{exc.__class__.__name__}")
    if not preview_path.is_file():
        return ArtifactPreviewRecord(artifact_path, None, "artifact-preview-missing-output")
    return ArtifactPreviewRecord(artifact_path, preview_path)


def _preview_cache_key(artifact_path: Path, window_size: tuple[int, int]) -> str:
    stat = artifact_path.stat()
    payload = "|".join(
        (
            str(artifact_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            f"{window_size[0]}x{window_size[1]}",
            _RENDER_CONTRACT_VERSION,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
