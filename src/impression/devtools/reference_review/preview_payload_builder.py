"""Non-UI source loading and tessellation for preview payloads."""

from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Any, Callable, Iterator

import numpy as np

from impression.mesh import Mesh, Polyline
from impression.modeling import (
    SurfaceBody,
    SurfaceConsumerCollection,
    preview_tessellation_request,
    tessellate_surface_body,
)
from impression.modeling.drawing2d import Path2D
from impression.modeling.group import MeshGroup
from impression.modeling.path3d import Path3D
from impression.modeling.topology import Region, Section

from .preview_payload import (
    PreviewPayload,
    PreviewPayloadDiagnostic,
    PreviewPayloadFileMetadata,
    PreviewPayloadRequest,
)

PREVIEW_PAYLOAD_FORMAT = "impression.reference-review.preview-payload.v1"
_PREVIEW_SOURCE_IMPORT_LOCK = Lock()
_RENDERABLE_ARTIFACT_SUFFIXES = frozenset({".stl", ".impress"})


@dataclass(frozen=True)
class LoadedPreviewDataset:
    """Preview-ready datasets produced outside the UI thread."""

    request: PreviewPayloadRequest
    datasets: tuple[Mesh | Polyline, ...]
    source_type: str

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("loaded preview dataset must not be empty")
        if not self.source_type:
            raise ValueError("source_type must not be empty")
        object.__setattr__(self, "datasets", tuple(self.datasets))

    @property
    def dataset_count(self) -> int:
        return len(self.datasets)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "owner": self.request.owner,
            "request_id": self.request.request_id,
            "fixture_id": self.request.fixture_id,
            "generation": self.request.generation,
            "source_type": self.source_type,
            "dataset_count": self.dataset_count,
            "dataset_types": [item.__class__.__name__ for item in self.datasets],
        }


@dataclass(frozen=True)
class PreviewDatasetBuildResult:
    """Result of loading and tessellating a preview request."""

    request: PreviewPayloadRequest
    dataset: LoadedPreviewDataset | None = None
    diagnostic: PreviewPayloadDiagnostic | None = None

    def __post_init__(self) -> None:
        if self.dataset is not None and self.diagnostic is not None:
            raise ValueError("preview dataset result cannot contain both dataset and diagnostic")
        if self.dataset is None and self.diagnostic is None:
            raise ValueError("preview dataset result requires dataset or diagnostic")

    @property
    def ok(self) -> bool:
        return self.dataset is not None


@dataclass(frozen=True)
class ImpressPreviewBuildResult:
    """Preview dataset result for a file-backed `.impress` artifact."""

    generation: int
    artifact_path: Path
    datasets: tuple[object, ...] = ()
    diagnostic: str | None = None


def load_impress_preview_datasets(artifact_path: Path) -> tuple[object, ...]:
    """Load and tessellate a `.impress` artifact without importing UI modules."""

    from impression.io import load_impress

    loaded = load_impress(artifact_path)
    request = preview_tessellation_request(require_watertight=False)
    return tuple(tessellate_surface_body(body, request).mesh for body in loaded.bodies)


def load_artifact_preview_dataset(request: PreviewPayloadRequest) -> LoadedPreviewDataset | None:
    """Load the selected renderable artifact when a fixture has STL/.impress output."""

    artifact_path = _first_renderable_artifact_path(request)
    if artifact_path is None:
        return None
    if artifact_path.suffix.lower() == ".impress":
        datasets = load_impress_preview_datasets(artifact_path)
    elif artifact_path.suffix.lower() == ".stl":
        datasets = (_load_stl_preview_mesh(artifact_path),)
    else:
        return None
    if not datasets:
        raise ValueError("artifact produced no preview datasets")
    return LoadedPreviewDataset(
        request=request,
        datasets=tuple(datasets),
        source_type=f"artifact:{artifact_path.suffix.lower()}",
    )


def build_impress_preview_result(generation: int, artifact_path: Path) -> ImpressPreviewBuildResult:
    """Build preview datasets for `.impress` artifacts in a non-UI worker."""

    try:
        datasets = load_impress_preview_datasets(artifact_path)
    except Exception as exc:
        return ImpressPreviewBuildResult(generation, artifact_path, diagnostic=exc.__class__.__name__)
    return ImpressPreviewBuildResult(generation, artifact_path, datasets=datasets)


def load_preview_source(
    request: PreviewPayloadRequest,
    *,
    import_roots: tuple[Path, ...] = (),
) -> object:
    """Load a source fixture entrypoint and invoke it with request parameters."""

    entrypoint = request.entrypoint
    kwargs = {parameter.name: parameter.value for parameter in request.parameters}
    with _PREVIEW_SOURCE_IMPORT_LOCK:
        with _temporary_import_roots(_source_import_roots(request, import_roots)):
            _evict_preview_source_modules(request)
            if "." in entrypoint and not request.source_path.is_file():
                module_name, function_name = entrypoint.rsplit(".", 1)
                module = importlib.import_module(module_name)
                builder = getattr(module, function_name, None)
            else:
                module = _load_module_from_path(request.source_path, request)
                builder = _resolve_entrypoint(module, entrypoint)
            if builder is None or not callable(builder):
                raise ValueError(f"entrypoint {entrypoint!r} is not callable")
            return builder(**kwargs)


def tessellate_preview_source(
    request: PreviewPayloadRequest,
    source: object,
) -> LoadedPreviewDataset:
    """Convert a loaded source object into preview-ready internal datasets."""

    datasets: list[Mesh | Polyline] = []
    tessellation_request = preview_tessellation_request(require_watertight=False)

    def loop_to_polyline(loop_points: np.ndarray) -> Polyline:
        pts = np.asarray(loop_points, dtype=float).reshape(-1, 2)
        pts3 = np.column_stack([pts, np.zeros((pts.shape[0], 1), dtype=float)])
        return Polyline(pts3, closed=True, color=None)

    def region_to_polylines(region: Region) -> list[Polyline]:
        normalized = region.normalized()
        polylines = [loop_to_polyline(normalized.outer.points)]
        polylines.extend(loop_to_polyline(hole.points) for hole in normalized.holes)
        return polylines

    def visit(item: object) -> None:
        if item is None:
            return
        if isinstance(item, MeshGroup) or (hasattr(item, "to_meshes") and callable(getattr(item, "to_meshes"))):
            for mesh in item.to_meshes():
                visit(mesh)
            return
        if isinstance(item, Mesh):
            datasets.append(item)
            return
        if isinstance(item, Polyline):
            datasets.append(item)
            return
        if isinstance(item, SurfaceBody):
            datasets.append(tessellate_surface_body(item, tessellation_request).mesh)
            return
        if isinstance(item, SurfaceConsumerCollection):
            for record in item.items:
                visit(record.body)
            return
        if isinstance(item, Path2D):
            datasets.append(item.to_polyline())
            return
        if hasattr(item, "to_polylines") and callable(getattr(item, "to_polylines")):
            for polyline in item.to_polylines():
                visit(polyline)
            return
        if isinstance(item, Region):
            datasets.extend(region_to_polylines(item))
            return
        if isinstance(item, Section):
            for region in item.normalized().regions:
                datasets.extend(region_to_polylines(region))
            return
        if isinstance(item, Path3D):
            datasets.append(item.to_polyline())
            return
        if isinstance(item, (list, tuple, set)):
            for value in item:
                visit(value)
            return
        raise ValueError(f"unsupported preview source type: {item.__class__.__name__}")

    visit(source)
    if not datasets:
        raise ValueError("source produced no preview datasets")
    return LoadedPreviewDataset(
        request=request,
        datasets=tuple(datasets),
        source_type=source.__class__.__name__,
    )


def build_preview_dataset(
    request: PreviewPayloadRequest,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> PreviewDatasetBuildResult:
    """Load and tessellate a preview request, returning sanitized diagnostics."""

    try:
        dataset = load_artifact_preview_dataset(request)
        if dataset is None:
            source = load_preview_source(request, import_roots=(() if cwd is None else (cwd,)))
            dataset = tessellate_preview_source(request, source)
    except Exception as exc:
        return PreviewDatasetBuildResult(
            request=request,
            diagnostic=PreviewPayloadDiagnostic.from_exception(
                request,
                exc,
                code="preview-source-load-failed",
                cwd=cwd,
                home=home,
            ),
        )
    return PreviewDatasetBuildResult(request=request, dataset=dataset)


def serialize_preview_dataset(dataset: LoadedPreviewDataset) -> dict[str, Any]:
    """Convert loaded preview datasets into a JSON-compatible payload."""

    return {
        "format": PREVIEW_PAYLOAD_FORMAT,
        "request": dataset.request.to_json_dict(),
        "source_type": dataset.source_type,
        "datasets": [_serialize_dataset(item) for item in dataset.datasets],
    }


def write_preview_payload_file(
    dataset: LoadedPreviewDataset,
    *,
    payload_dir: Path | None = None,
) -> PreviewPayload:
    """Write a preview payload JSON file and return its result record."""

    directory = Path(payload_dir) if payload_dir is not None else Path(
        tempfile.mkdtemp(prefix="impression-reference-review-preview-")
    )
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / _payload_file_name(dataset.request)
    payload = serialize_preview_dataset(dataset)
    payload_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    metadata = PreviewPayloadFileMetadata(
        path=payload_path,
        payload_format=PREVIEW_PAYLOAD_FORMAT,
        byte_count=payload_path.stat().st_size,
        dataset_count=dataset.dataset_count,
    )
    return PreviewPayload.success(
        dataset.request,
        payload_path=payload_path,
        payload_kind=PREVIEW_PAYLOAD_FORMAT,
        file_metadata=metadata,
    )


def build_serialized_preview_payload(
    request: PreviewPayloadRequest,
    *,
    payload_dir: Path | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    dataset_builder: Callable[..., PreviewDatasetBuildResult] = build_preview_dataset,
    payload_writer: Callable[..., PreviewPayload] = write_preview_payload_file,
) -> PreviewPayload:
    """Load, tessellate, and serialize a preview request."""

    result = dataset_builder(request, cwd=cwd, home=home)
    if result.diagnostic is not None:
        return PreviewPayload.failure(
            request,
            result.diagnostic,
            payload_kind=PREVIEW_PAYLOAD_FORMAT,
        )
    assert result.dataset is not None
    try:
        return payload_writer(result.dataset, payload_dir=payload_dir)
    except Exception as exc:
        diagnostic = PreviewPayloadDiagnostic.from_exception(
            request,
            exc,
            code="preview-payload-serialization-failed",
            cwd=cwd,
            home=home,
        )
        return PreviewPayload.failure(
            request,
            diagnostic,
            payload_kind=PREVIEW_PAYLOAD_FORMAT,
        )


def _load_module_from_path(path: Path, request: PreviewPayloadRequest) -> ModuleType:
    module_path = Path(path)
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    module_name = (
        "impression_reference_review_"
        + request.fixture_id.replace("/", "_").replace("-", "_")
        + f"_{request.request_id}_{request.generation}"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import source module {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _first_renderable_artifact_path(request: PreviewPayloadRequest) -> Path | None:
    for path in request.artifact_paths:
        artifact_path = Path(path)
        if artifact_path.suffix.lower() not in _RENDERABLE_ARTIFACT_SUFFIXES:
            continue
        if artifact_path.is_file():
            return artifact_path
    return None


def _load_stl_preview_mesh(artifact_path: Path) -> Mesh:
    data = artifact_path.read_bytes()
    vertices, faces = _read_binary_stl(data)
    if faces.size == 0:
        vertices, faces = _read_ascii_stl(data)
    if faces.size == 0:
        raise ValueError("STL artifact has no faces")
    return Mesh(
        vertices=vertices,
        faces=faces,
        metadata={"source_artifact": artifact_path.as_posix()},
    )


def _read_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 84:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if triangle_count < 1 or expected_size != len(data):
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    vertices = np.empty((triangle_count * 3, 3), dtype=float)
    faces = np.empty((triangle_count, 3), dtype=int)
    offset = 84
    for index in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        vertices[index * 3 : index * 3 + 3] = np.asarray(values[3:12], dtype=float).reshape(3, 3)
        faces[index] = (index * 3, index * 3 + 1, index * 3 + 2)
        offset += 50
    return vertices, faces


def _read_ascii_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 4 or parts[0].lower() != "vertex":
            continue
        try:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    usable_count = len(vertices) - (len(vertices) % 3)
    if usable_count < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    vertex_array = np.asarray(vertices[:usable_count], dtype=float)
    faces = np.arange(usable_count, dtype=int).reshape(-1, 3)
    return vertex_array, faces


def _evict_preview_source_modules(request: PreviewPayloadRequest) -> None:
    source_paths = _preview_source_cache_paths(request)
    if not source_paths:
        return
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            module_path = Path(module_file).expanduser().resolve()
        except OSError:
            continue
        if module_path in source_paths or _is_previous_preview_temp_module(module_path):
            sys.modules.pop(module_name, None)


def _preview_source_cache_paths(request: PreviewPayloadRequest) -> set[Path]:
    paths = {Path(request.source_path).expanduser().resolve()}
    metadata_paths = request.metadata.get("source_snapshot_paths", ())
    if isinstance(metadata_paths, (list, tuple)):
        for value in metadata_paths:
            if isinstance(value, str):
                paths.add(Path(value).expanduser().resolve())
    original_subject = request.metadata.get("original_subject_path")
    if isinstance(original_subject, str):
        paths.add(Path(original_subject).expanduser().resolve())
    return paths


def _is_previous_preview_temp_module(path: Path) -> bool:
    return any(part.startswith("impression-bench-preview-source-") for part in path.parts)


def _source_import_roots(
    request: PreviewPayloadRequest,
    explicit_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for root in explicit_roots:
        roots.append(Path(root))
    source_path = Path(request.source_path)
    if source_path.parent != Path("."):
        roots.append(source_path.parent)
    roots.extend(_workspace_roots_for(source_path))
    return _dedupe_existing_roots(tuple(roots))


def _workspace_roots_for(path: Path) -> tuple[Path, ...]:
    try:
        current = path.resolve()
    except OSError:
        current = path
    if current.is_file():
        current = current.parent
    roots: list[Path] = []
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            roots.append(parent)
            break
    return tuple(roots)


def _dedupe_existing_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if not resolved.exists():
            continue
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return tuple(result)


@contextmanager
def _temporary_import_roots(roots: tuple[Path, ...]) -> Iterator[None]:
    original_path = list(sys.path)
    try:
        for root in reversed(roots):
            root_text = root.as_posix()
            if root_text in sys.path:
                sys.path.remove(root_text)
            sys.path.insert(0, root_text)
        importlib.invalidate_caches()
        yield
    finally:
        sys.path[:] = original_path
        importlib.invalidate_caches()


def _resolve_entrypoint(module: ModuleType, entrypoint: str) -> object:
    current: object = module
    for part in entrypoint.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _serialize_dataset(dataset: Mesh | Polyline) -> dict[str, Any]:
    if isinstance(dataset, Mesh):
        return {
            "kind": "mesh",
            "vertices": dataset.vertices.tolist(),
            "faces": dataset.faces.tolist(),
            "color": None if dataset.color is None else list(dataset.color),
            "face_colors": None
            if dataset.face_colors is None
            else dataset.face_colors.tolist(),
            "metadata": _json_safe_mapping(dataset.metadata),
        }
    return {
        "kind": "polyline",
        "points": dataset.points.tolist(),
        "closed": dataset.closed,
        "color": None if dataset.color is None else list(dataset.color),
    }


def _json_safe_mapping(mapping: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(mapping, sort_keys=True, default=str))


def _payload_file_name(request: PreviewPayloadRequest) -> str:
    fixture = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in request.fixture_id
    ).strip("_")
    if not fixture:
        fixture = "fixture"
    return f"{fixture}-{request.request_id}-{request.generation}.preview-payload.json"
