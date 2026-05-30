from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Literal, Sequence

import numpy as np
from PIL import Image
from PIL import ImageDraw
import pyvista as pv
from skimage import morphology, transform
from fontTools.ttLib import TTFont, TTLibFileIsCollectionError

from impression.io.stl import write_stl
from impression.mesh import Mesh, combine_meshes, mesh_to_pyvista, section_mesh_with_plane
from impression.modeling import SurfaceBody, SurfaceConsumerCollection, TessellationRequest, tessellate_surface_body
from impression.modeling.text import text_profiles

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SilhouetteRelationship = Literal[
    "same_shape_same_orientation",
    "same_shape_rotated",
    "different_shape",
]
ReferenceArtifactKind = Literal["image", "stl"]
ReferenceBaselineTier = Literal["dirty", "clean", "missing"]
SharedCvResultPattern = Literal["positive", "transformed", "different", "unknown"]
CvHarnessStage = Literal["build", "render", "normalize", "interpret", "review"]
CameraProjectionMode = Literal["parallel", "perspective"]
CameraContractViolationKind = Literal[
    "pose_drift",
    "target_drift",
    "up_vector_drift",
    "projection_drift",
    "framing_drift",
    "crop_drift",
]
CanonicalObjectViewName = Literal["front", "side", "top", "isometric"]
AxisName = Literal["x", "y", "z", "-x", "-y", "-z"]
HandednessResultClass = Literal["same_handedness", "mirrored", "orientation_unknown"]
TextCvResultClass = Literal[
    "same_text_same_orientation",
    "same_text_rotated",
    "same_text_mirrored",
    "different_text",
    "unreadable",
]
DiagnosticPanelLabel = Literal["left", "result", "right", "expected", "actual", "diff"]
DiagnosticSnapshotDriftKind = Literal["matches", "changed"]


@dataclass(frozen=True)
class SilhouetteComparison:
    relationship: SilhouetteRelationship
    same_orientation_iou: float
    best_rotation_iou: float
    best_rotation_deg: int


@dataclass(frozen=True)
class ReferenceArtifactState:
    kind: ReferenceArtifactKind
    dirty_path: Path
    clean_path: Path
    selected_path: Path | None
    selected_tier: ReferenceBaselineTier

    @property
    def exists(self) -> bool:
        return self.selected_path is not None


@dataclass(frozen=True)
class ReferenceFixturePairState:
    image: ReferenceArtifactState
    stl: ReferenceArtifactState

    @property
    def is_new_fixture(self) -> bool:
        return not self.image.exists and not self.stl.exists

    @property
    def has_partial_group(self) -> bool:
        return self.image.exists != self.stl.exists


@dataclass(frozen=True)
class ReferenceFixtureContractVersionRecord:
    fixture_id: str
    contract_version: str

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("ReferenceFixtureContractVersionRecord.fixture_id must be non-empty.")
        if not self.contract_version.strip():
            raise ValueError("ReferenceFixtureContractVersionRecord.contract_version must be non-empty.")


@dataclass(frozen=True)
class ReferencePromotionDiagnostic:
    code: Literal["missing-artifact", "dirty-artifact", "partial-fixture", "invalidated-contract"]
    fixture_id: str
    artifact_kind: ReferenceArtifactKind | Literal["fixture"]
    message: str


@dataclass(frozen=True)
class ReferenceArtifactPromotionGateReport:
    fixture_id: str
    promoted: bool
    state: ReferenceFixturePairState
    contract: ReferenceFixtureContractVersionRecord
    diagnostics: tuple[ReferencePromotionDiagnostic, ...]


@dataclass(frozen=True)
class DiagnosticSnapshotKeyPolicy:
    """Stable diagnostic snapshot key policy that excludes volatile details."""

    ignored_keys: tuple[str, ...] = ("traceback", "stack", "stack_trace", "cwd", "tmp_path", "temporary_path")
    path_keys: tuple[str, ...] = ("path", "file", "filename", "implementation_owner")

    def __post_init__(self) -> None:
        _validate_unique_nonempty_strings("ignored_keys", self.ignored_keys)
        _validate_unique_nonempty_strings("path_keys", self.path_keys)


@dataclass(frozen=True)
class DiagnosticSnapshotRecord:
    fixture_id: str
    diagnostic_type: str
    payload: dict[str, object]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostic_type": self.diagnostic_type,
            "fixture_id": self.fixture_id,
            "payload": self.payload,
        }

    def stable_json(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DiagnosticSnapshotComparison:
    fixture_id: str
    drift_kind: DiagnosticSnapshotDriftKind
    expected: DiagnosticSnapshotRecord
    actual: DiagnosticSnapshotRecord
    message: str = ""

    @property
    def matches(self) -> bool:
        return self.drift_kind == "matches"


@dataclass(frozen=True)
class ExpectedDiagnosticKeyRecord:
    key_path: tuple[str, ...]
    expected_value: object | None = None

    def __post_init__(self) -> None:
        _validate_unique_nonempty_strings("key_path", self.key_path)


@dataclass(frozen=True)
class NegativeDiagnosticFixtureRecord:
    fixture_id: str
    domain: str
    expected_keys: tuple[ExpectedDiagnosticKeyRecord, ...]
    expected_snapshot: DiagnosticSnapshotRecord | None = None

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("NegativeDiagnosticFixtureRecord.fixture_id must be non-empty.")
        if not self.domain.strip():
            raise ValueError("NegativeDiagnosticFixtureRecord.domain must be non-empty.")
        if not self.expected_keys:
            raise ValueError("NegativeDiagnosticFixtureRecord.expected_keys must be non-empty.")


@dataclass(frozen=True)
class NegativeDiagnosticDomainCoverageRecord:
    domain: str
    fixture_count: int
    covered: bool


@dataclass(frozen=True)
class NegativeDiagnosticFixtureMatrixDiagnostic:
    code: Literal["missing-domain", "missing-diagnostic-key", "snapshot-drift"]
    fixture_id: str
    domain: str
    message: str


@dataclass(frozen=True)
class NegativeDiagnosticFixtureMatrixReport:
    passed: bool
    fixtures: tuple[NegativeDiagnosticFixtureRecord, ...]
    domain_coverage: tuple[NegativeDiagnosticDomainCoverageRecord, ...]
    diagnostics: tuple[NegativeDiagnosticFixtureMatrixDiagnostic, ...]


@dataclass(frozen=True)
class CvFixtureContract:
    fixture_id: str
    lane: str
    required_artifact_keys: tuple[str, ...]
    known_result_classes: tuple[str, ...]
    positive_result_classes: tuple[str, ...]
    transformed_result_classes: tuple[str, ...] = ()
    different_result_classes: tuple[str, ...] = ()
    unknown_result_classes: tuple[str, ...] = ()
    passing_result_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_unique_nonempty_strings("required_artifact_keys", self.required_artifact_keys)
        _validate_unique_nonempty_strings("known_result_classes", self.known_result_classes)
        _validate_unique_nonempty_strings("positive_result_classes", self.positive_result_classes)
        _validate_unique_nonempty_strings("passing_result_classes", self.passing_result_classes)
        _validate_optional_unique_strings("transformed_result_classes", self.transformed_result_classes)
        _validate_optional_unique_strings("different_result_classes", self.different_result_classes)
        _validate_optional_unique_strings("unknown_result_classes", self.unknown_result_classes)
        if not self.fixture_id.strip():
            raise ValueError("CvFixtureContract requires a non-empty fixture_id.")
        if not self.lane.strip():
            raise ValueError("CvFixtureContract requires a non-empty lane.")
        known = set(self.known_result_classes)
        partitions = {
            "positive_result_classes": set(self.positive_result_classes),
            "transformed_result_classes": set(self.transformed_result_classes),
            "different_result_classes": set(self.different_result_classes),
            "unknown_result_classes": set(self.unknown_result_classes),
        }
        combined = set()
        for field_name, values in partitions.items():
            if not values:
                continue
            if not values.issubset(known):
                unknown_values = sorted(values.difference(known))
                raise ValueError(f"{field_name} contains undeclared result classes: {unknown_values}")
            overlap = combined.intersection(values)
            if overlap:
                raise ValueError(f"CV result classes cannot appear in multiple partitions: {sorted(overlap)}")
            combined.update(values)
        if combined != known:
            missing = sorted(known.difference(combined))
            raise ValueError(f"CV result classes must be partitioned explicitly. Missing: {missing}")
        passing = set(self.passing_result_classes)
        if not passing.issubset(known):
            unknown_values = sorted(passing.difference(known))
            raise ValueError(f"passing_result_classes contains undeclared result classes: {unknown_values}")


@dataclass(frozen=True)
class CvResultAssessment:
    lane_result_class: str
    shared_pattern: SharedCvResultPattern
    passes_contract: bool


@dataclass(frozen=True)
class CvArtifactBundleContract:
    fixture_id: str
    lane: str
    required_keys: tuple[str, ...]
    authoritative_keys: tuple[str, ...] = ()
    review_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_unique_nonempty_strings("required_keys", self.required_keys)
        _validate_optional_unique_strings("authoritative_keys", self.authoritative_keys)
        _validate_optional_unique_strings("review_keys", self.review_keys)
        if not self.fixture_id.strip():
            raise ValueError("CvArtifactBundleContract requires a non-empty fixture_id.")
        if not self.lane.strip():
            raise ValueError("CvArtifactBundleContract requires a non-empty lane.")
        required = set(self.required_keys)
        if self.authoritative_keys and not set(self.authoritative_keys).issubset(required):
            raise ValueError("authoritative_keys must be a subset of required_keys.")
        if self.review_keys and not set(self.review_keys).issubset(required):
            raise ValueError("review_keys must be a subset of required_keys.")


@dataclass(frozen=True)
class CvArtifactBundle:
    contract: CvArtifactBundleContract
    stage: CvHarnessStage
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class CameraFramingContract:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up_vector: tuple[float, float, float]
    projection_mode: CameraProjectionMode
    window_size: tuple[int, int]
    parallel_scale: float | None = None


@dataclass(frozen=True)
class CameraContractViolation:
    kind: CameraContractViolationKind
    expected: object
    observed: object


@dataclass(frozen=True)
class CanonicalObjectViewBundle:
    view_order: tuple[CanonicalObjectViewName, ...]
    silhouettes: dict[CanonicalObjectViewName, Path]
    diagnostic_beauty: dict[CanonicalObjectViewName, Path] | None = None


@dataclass(frozen=True)
class CanonicalObjectViewComparison:
    per_view: dict[CanonicalObjectViewName, SilhouetteComparison]
    mismatched_views: tuple[CanonicalObjectViewName, ...]


@dataclass(frozen=True)
class HandednessSpaceAnchorContract:
    modeling_basis: tuple[AxisName, AxisName, AxisName]
    export_basis: tuple[AxisName, AxisName, AxisName]
    viewer_basis: tuple[AxisName, AxisName, AxisName]
    canonical_view: CanonicalObjectViewName
    camera_contract: CameraFramingContract | None


@dataclass(frozen=True)
class HandednessClassification:
    result_class: HandednessResultClass
    same_iou: float
    mirrored_iou: float
    witness_adequate: bool


@dataclass(frozen=True)
class DiagnosticPanelContract:
    panel_order: tuple[str, ...]
    diagnostic_only: bool = True
    delegated_proof_lane: str | None = None
    shared_scene: bool = False


@dataclass(frozen=True)
class DiagnosticPanelLayout:
    contract: DiagnosticPanelContract
    panel_boxes: dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class TextCvComparison:
    result_class: TextCvResultClass
    same_orientation_iou: float
    best_rotation_iou: float
    best_rotation_deg: int
    best_mirror_iou: float
    best_mirror_deg: int
    fallback_detected: bool
    missing_glyphs: tuple[str, ...]
    in_scope: bool


def render_surface_body_image(
    body: SurfaceBody,
    output_path: Path,
    *,
    request: TessellationRequest | None = None,
    window_size: tuple[int, int] = (960, 720),
    mesh_color: str = "#5b84b1",
    background: str = "white",
) -> Path:
    result = tessellate_surface_body(body, request)
    return render_mesh_image(
        result.mesh,
        output_path,
        window_size=window_size,
        mesh_color=mesh_color,
        background=background,
    )


def render_surface_body_triptych_image(
    left_body: SurfaceBody,
    result_body: SurfaceBody,
    right_body: SurfaceBody,
    output_path: Path,
    *,
    request: TessellationRequest | None = None,
    window_size: tuple[int, int] = (480, 480),
    left_color: str = "#d97706",
    result_color: str = "#2563eb",
    right_color: str = "#059669",
    background: str = "white",
    panel_padding: int = 24,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_specs = (
        ("left", left_body, left_color),
        ("result", result_body, result_color),
        ("right", right_body, right_color),
    )
    panel_paths = []
    for label, body, mesh_color in panel_specs:
        panel_path = output_path.with_name(f"{output_path.stem}-{label}.png")
        render_surface_body_image(
            body,
            panel_path,
            request=request,
            window_size=window_size,
            mesh_color=mesh_color,
            background=background,
        )
        panel_paths.append(panel_path)

    images = [Image.open(path).convert("RGB") for path in panel_paths]
    try:
        total_width = sum(image.width for image in images) + (panel_padding * (len(images) + 1))
        max_height = max(image.height for image in images) + (panel_padding * 2)
        canvas = Image.new("RGB", (total_width, max_height), color=background)
        cursor_x = panel_padding
        for image in images:
            y = panel_padding + ((max_height - (panel_padding * 2) - image.height) // 2)
            canvas.paste(image, (cursor_x, y))
            cursor_x += image.width + panel_padding
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
        for path in panel_paths:
            path.unlink(missing_ok=True)
    return output_path


def write_surface_body_stl(
    body: SurfaceBody,
    output_path: Path,
    *,
    request: TessellationRequest | None = None,
    ascii: bool = True,
) -> Path:
    result = tessellate_surface_body(body, request)
    return write_mesh_stl(result.mesh, output_path, ascii=ascii)


def render_surface_consumer_collection_image(
    collection: SurfaceConsumerCollection,
    output_path: Path,
    *,
    request: TessellationRequest | None = None,
    window_size: tuple[int, int] = (960, 720),
    mesh_color: str = "#5b84b1",
    background: str = "white",
) -> Path:
    mesh = _combined_collection_mesh(collection, request=request)
    return render_mesh_image(
        mesh,
        output_path,
        window_size=window_size,
        mesh_color=mesh_color,
        background=background,
    )


def write_surface_consumer_collection_stl(
    collection: SurfaceConsumerCollection,
    output_path: Path,
    *,
    request: TessellationRequest | None = None,
    ascii: bool = True,
) -> Path:
    mesh = _combined_collection_mesh(collection, request=request)
    return write_mesh_stl(mesh, output_path, ascii=ascii)


def render_mesh_image(
    mesh: Mesh,
    output_path: Path,
    *,
    window_size: tuple[int, int] = (960, 720),
    mesh_color: str = "#5b84b1",
    background: str = "white",
    camera_contract: CameraFramingContract | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pv_mesh = mesh_to_pyvista(mesh)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(pv_mesh, color=mesh_color, smooth_shading=False, show_edges=False)
    if camera_contract is None:
        plotter.camera_position = _camera_position_for_bounds(mesh.bounds)
    else:
        plotter.camera_position = [
            camera_contract.position,
            camera_contract.target,
            camera_contract.up_vector,
        ]
        plotter.camera.parallel_projection = camera_contract.projection_mode == "parallel"
        if camera_contract.parallel_scale is not None:
            plotter.camera.parallel_scale = float(camera_contract.parallel_scale)
    plotter.show(screenshot=str(output_path), auto_close=True, interactive=False)
    return output_path


def write_mesh_stl(mesh: Mesh, output_path: Path, *, ascii: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_stl(mesh, output_path, ascii=ascii)
    return output_path


def render_mesh_image_with_camera_contract(
    mesh: Mesh,
    output_path: Path,
    *,
    camera_contract: CameraFramingContract,
    mesh_color: str = "#5b84b1",
    background: str = "white",
) -> Path:
    return render_mesh_image(
        mesh,
        output_path,
        window_size=camera_contract.window_size,
        mesh_color=mesh_color,
        background=background,
        camera_contract=camera_contract,
    )


def planar_loop_bounds(
    *loop_groups: list[np.ndarray] | tuple[np.ndarray, ...],
    padding_ratio: float = 0.08,
    min_span: float = 1.0,
) -> tuple[float, float, float, float]:
    loops = [np.asarray(loop, dtype=float).reshape(-1, 2) for group in loop_groups for loop in group]
    if not loops:
        raise ValueError("planar_loop_bounds requires at least one loop.")
    stacked = np.vstack(loops)
    xmin = float(stacked[:, 0].min())
    xmax = float(stacked[:, 0].max())
    ymin = float(stacked[:, 1].min())
    ymax = float(stacked[:, 1].max())
    x_span = max(xmax - xmin, float(min_span))
    y_span = max(ymax - ymin, float(min_span))
    x_pad = x_span * float(padding_ratio)
    y_pad = y_span * float(padding_ratio)
    return (xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad)


def render_planar_section_bitmap(
    loops: list[np.ndarray] | tuple[np.ndarray, ...],
    output_path: Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (512, 512),
    stroke_width: int = 4,
    background: int = 0,
    foreground: int = 255,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = _planar_section_mask(
        loops,
        bounds=bounds,
        image_size=image_size,
        stroke_width=stroke_width,
        background=background,
        foreground=foreground,
    )
    Image.fromarray(mask, mode="L").save(output_path)
    return output_path


def render_planar_section_fill_bitmap(
    loops: list[np.ndarray] | tuple[np.ndarray, ...],
    output_path: Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (512, 512),
    background: int = 255,
    foreground: int = 0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bounds is None:
        bounds = planar_loop_bounds(loops)
    mask = _planar_section_fill_mask(loops, bounds=bounds, image_size=image_size)
    image = np.full(mask.shape, int(background), dtype=np.uint8)
    image[mask] = int(foreground)
    Image.fromarray(image, mode="L").save(output_path)
    return output_path


def render_planar_section_diff_image(
    expected_loops: list[np.ndarray] | tuple[np.ndarray, ...],
    actual_loops: list[np.ndarray] | tuple[np.ndarray, ...],
    output_path: Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (512, 512),
    stroke_width: int = 4,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shared_bounds = bounds if bounds is not None else planar_loop_bounds(expected_loops, actual_loops)
    expected = _planar_section_mask(
        expected_loops,
        bounds=shared_bounds,
        image_size=image_size,
        stroke_width=stroke_width,
    )
    actual = _planar_section_mask(
        actual_loops,
        bounds=shared_bounds,
        image_size=image_size,
        stroke_width=stroke_width,
    )
    expected_only = (expected > 0) & (actual == 0)
    actual_only = (actual > 0) & (expected == 0)
    overlap = (expected > 0) & (actual > 0)
    rgb = np.zeros((*expected.shape, 3), dtype=np.uint8)
    rgb[expected_only] = np.asarray([70, 150, 255], dtype=np.uint8)
    rgb[actual_only] = np.asarray([255, 140, 70], dtype=np.uint8)
    rgb[overlap] = np.asarray([255, 255, 255], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return output_path


def dirty_reference_path(reference_root: Path, name: str) -> Path:
    return reference_root / "dirty" / f"{name}.png"


def clean_reference_path(reference_root: Path, name: str) -> Path:
    return reference_root / "clean" / f"{name}.png"


def reference_image_path(reference_root: Path, name: str) -> Path:
    state = reference_artifact_state(reference_root, name, kind="image")
    return state.selected_path if state.selected_path is not None else state.dirty_path


def dirty_reference_stl_path(reference_root: Path, name: str) -> Path:
    return reference_root / "dirty" / f"{name}.stl"


def clean_reference_stl_path(reference_root: Path, name: str) -> Path:
    return reference_root / "clean" / f"{name}.stl"


def reference_stl_path(reference_root: Path, name: str) -> Path:
    state = reference_artifact_state(reference_root, name, kind="stl")
    return state.selected_path if state.selected_path is not None else state.dirty_path


def reference_artifact_state(
    reference_root: Path,
    name: str,
    *,
    kind: ReferenceArtifactKind,
) -> ReferenceArtifactState:
    if kind == "image":
        dirty_path = dirty_reference_path(reference_root, name)
        clean_path = clean_reference_path(reference_root, name)
    else:
        dirty_path = dirty_reference_stl_path(reference_root, name)
        clean_path = clean_reference_stl_path(reference_root, name)

    if clean_path.exists():
        return ReferenceArtifactState(
            kind=kind,
            dirty_path=dirty_path,
            clean_path=clean_path,
            selected_path=clean_path,
            selected_tier="clean",
        )
    if dirty_path.exists():
        return ReferenceArtifactState(
            kind=kind,
            dirty_path=dirty_path,
            clean_path=clean_path,
            selected_path=dirty_path,
            selected_tier="dirty",
        )
    return ReferenceArtifactState(
        kind=kind,
        dirty_path=dirty_path,
        clean_path=clean_path,
        selected_path=None,
        selected_tier="missing",
    )


def reference_artifact_contract_version_path(state: ReferenceArtifactState) -> Path:
    return state.clean_path.with_suffix(state.clean_path.suffix + ".contract")


def write_reference_artifact_contract_version(
    state: ReferenceArtifactState,
    contract: ReferenceFixtureContractVersionRecord,
) -> Path:
    path = reference_artifact_contract_version_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{contract.fixture_id}\n{contract.contract_version}\n", encoding="utf-8")
    return path


def reference_artifact_contract_version_matches(
    state: ReferenceArtifactState,
    contract: ReferenceFixtureContractVersionRecord,
) -> bool:
    if state.selected_tier != "clean":
        return False
    path = reference_artifact_contract_version_path(state)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[:2] == [contract.fixture_id, contract.contract_version]


def reference_fixture_pair_state(
    *,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
) -> ReferenceFixturePairState:
    return ReferenceFixturePairState(
        image=reference_artifact_state(reference_image_root, name, kind="image"),
        stl=reference_artifact_state(reference_stl_root, name, kind="stl"),
    )


def classify_reference_fixture_pair_promotion_gate(
    *,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
    contract_version: str = "v1",
) -> ReferenceArtifactPromotionGateReport:
    """Classify whether a reference fixture pair is promoted clean evidence."""

    state = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    )
    contract = ReferenceFixtureContractVersionRecord(fixture_id=name, contract_version=contract_version)
    diagnostics: list[ReferencePromotionDiagnostic] = []
    if state.has_partial_group:
        diagnostics.append(
            ReferencePromotionDiagnostic(
                code="partial-fixture",
                fixture_id=name,
                artifact_kind="fixture",
                message=f"{name} has a partial reference fixture group.",
            )
        )
    for artifact_kind, artifact_state in (("image", state.image), ("stl", state.stl)):
        if artifact_state.selected_tier == "missing":
            diagnostics.append(
                ReferencePromotionDiagnostic(
                    code="missing-artifact",
                    fixture_id=name,
                    artifact_kind=artifact_kind,
                    message=f"{name} is missing a promoted {artifact_kind} reference artifact.",
                )
            )
        elif artifact_state.selected_tier == "dirty":
            diagnostics.append(
                ReferencePromotionDiagnostic(
                    code="dirty-artifact",
                    fixture_id=name,
                    artifact_kind=artifact_kind,
                    message=f"{name} has only a dirty {artifact_kind} reference artifact.",
                )
            )
        elif not reference_artifact_contract_version_matches(artifact_state, contract):
            diagnostics.append(
                ReferencePromotionDiagnostic(
                    code="invalidated-contract",
                    fixture_id=name,
                    artifact_kind=artifact_kind,
                    message=f"{name} promoted {artifact_kind} reference artifact has no matching contract version.",
                )
            )
    return ReferenceArtifactPromotionGateReport(
        fixture_id=name,
        promoted=not diagnostics,
        state=state,
        contract=contract,
        diagnostics=tuple(diagnostics),
    )


_PATH_FRAGMENT_RE = re.compile(r"(?P<path>(?:~|/)[^\s:]+)")


def _normalize_snapshot_string(value: str) -> str:
    def replace_path(match: re.Match[str]) -> str:
        path = match.group("path")
        name = Path(path).name
        return f"<path:{name}>" if name else "<path>"

    return _PATH_FRAGMENT_RE.sub(replace_path, value)


def _normalize_diagnostic_snapshot_value(value: object, policy: DiagnosticSnapshotKeyPolicy) -> object:
    if hasattr(value, "canonical_payload") and callable(value.canonical_payload):
        value = value.canonical_payload()
    if isinstance(value, BaseException):
        return {"message": _normalize_snapshot_string(str(value)), "type": type(value).__name__}
    if isinstance(value, Path):
        return f"<path:{value.name}>"
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if key in policy.ignored_keys:
                continue
            raw_value = value[raw_key]
            if key in policy.path_keys and isinstance(raw_value, (str, Path)):
                normalized[key] = _normalize_snapshot_string(str(raw_value))
            else:
                normalized[key] = _normalize_diagnostic_snapshot_value(raw_value, policy)
        return normalized
    if isinstance(value, (list, tuple, set)):
        return tuple(_normalize_diagnostic_snapshot_value(item, policy) for item in value)
    if isinstance(value, str):
        return _normalize_snapshot_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def normalize_diagnostic_snapshot(
    diagnostic: object,
    *,
    fixture_id: str,
    policy: DiagnosticSnapshotKeyPolicy | None = None,
) -> DiagnosticSnapshotRecord:
    """Normalize a refusal diagnostic into a stable, portable snapshot."""

    snapshot_policy = DiagnosticSnapshotKeyPolicy() if policy is None else policy
    payload = _normalize_diagnostic_snapshot_value(diagnostic, snapshot_policy)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return DiagnosticSnapshotRecord(
        fixture_id=fixture_id,
        diagnostic_type=type(diagnostic).__name__,
        payload=payload,
    )


def compare_diagnostic_snapshots(
    expected: DiagnosticSnapshotRecord,
    actual: DiagnosticSnapshotRecord,
) -> DiagnosticSnapshotComparison:
    """Compare normalized diagnostic snapshots without incidental formatting drift."""

    if expected.stable_json() == actual.stable_json():
        return DiagnosticSnapshotComparison(
            fixture_id=actual.fixture_id,
            drift_kind="matches",
            expected=expected,
            actual=actual,
        )
    return DiagnosticSnapshotComparison(
        fixture_id=actual.fixture_id,
        drift_kind="changed",
        expected=expected,
        actual=actual,
        message=f"Diagnostic snapshot drift for {actual.fixture_id}.",
    )


def _snapshot_value_at(payload: dict[str, object], key_path: tuple[str, ...]) -> object:
    current: object = payload
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(key_path))
        current = current[key]
    return current


def evaluate_negative_diagnostic_fixture_matrix(
    fixtures: Sequence[NegativeDiagnosticFixtureRecord],
    *,
    required_domains: Sequence[str],
) -> NegativeDiagnosticFixtureMatrixReport:
    """Evaluate domain coverage and expected-key coverage for negative diagnostics."""

    fixture_records = tuple(fixtures)
    domains = tuple(str(domain).strip() for domain in required_domains)
    _validate_unique_nonempty_strings("required_domains", domains)
    diagnostics: list[NegativeDiagnosticFixtureMatrixDiagnostic] = []
    by_domain = {domain: tuple(record for record in fixture_records if record.domain == domain) for domain in domains}
    coverage: list[NegativeDiagnosticDomainCoverageRecord] = []
    for domain in domains:
        domain_fixtures = by_domain[domain]
        coverage.append(
            NegativeDiagnosticDomainCoverageRecord(
                domain=domain,
                fixture_count=len(domain_fixtures),
                covered=bool(domain_fixtures),
            )
        )
        if not domain_fixtures:
            diagnostics.append(
                NegativeDiagnosticFixtureMatrixDiagnostic(
                    code="missing-domain",
                    fixture_id="",
                    domain=domain,
                    message=f"Negative diagnostic fixture matrix has no fixture for domain '{domain}'.",
                )
            )
    for fixture in fixture_records:
        if fixture.expected_snapshot is None:
            continue
        for expected_key in fixture.expected_keys:
            try:
                observed = _snapshot_value_at(fixture.expected_snapshot.payload, expected_key.key_path)
            except KeyError:
                diagnostics.append(
                    NegativeDiagnosticFixtureMatrixDiagnostic(
                        code="missing-diagnostic-key",
                        fixture_id=fixture.fixture_id,
                        domain=fixture.domain,
                        message=(
                            f"Negative diagnostic fixture '{fixture.fixture_id}' is missing "
                            f"expected key '{'.'.join(expected_key.key_path)}'."
                        ),
                    )
                )
                continue
            if expected_key.expected_value is not None and observed != expected_key.expected_value:
                diagnostics.append(
                    NegativeDiagnosticFixtureMatrixDiagnostic(
                        code="missing-diagnostic-key",
                        fixture_id=fixture.fixture_id,
                        domain=fixture.domain,
                        message=(
                            f"Negative diagnostic fixture '{fixture.fixture_id}' expected "
                            f"'{'.'.join(expected_key.key_path)}' to equal {expected_key.expected_value!r}."
                        ),
                    )
                )
    return NegativeDiagnosticFixtureMatrixReport(
        passed=not diagnostics,
        fixtures=fixture_records,
        domain_coverage=tuple(coverage),
        diagnostics=tuple(diagnostics),
    )


def compare_negative_diagnostic_fixture_snapshot(
    fixture: NegativeDiagnosticFixtureRecord,
    actual_snapshot: DiagnosticSnapshotRecord,
) -> DiagnosticSnapshotComparison:
    """Compare a domain-owned negative diagnostic fixture against its expected snapshot."""

    if fixture.expected_snapshot is None:
        raise ValueError("Negative diagnostic fixture has no expected snapshot to compare.")
    return compare_diagnostic_snapshots(fixture.expected_snapshot, actual_snapshot)


def reference_fixture_pair_failures(
    *,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
) -> tuple[str, ...]:
    state = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    )
    failures: list[str] = []
    if not state.image.exists:
        failures.append(f"{name} is missing a reference image baseline.")
    if not state.stl.exists:
        failures.append(f"{name} is missing a reference STL baseline.")
    return tuple(failures)


def required_reference_fixture_pair_failures(
    *,
    reference_image_root: Path,
    reference_stl_root: Path,
    names: Sequence[str],
) -> tuple[str, ...]:
    failures: list[str] = []
    for name in names:
        failures.extend(
            reference_fixture_pair_failures(
                reference_image_root=reference_image_root,
                reference_stl_root=reference_stl_root,
                name=name,
            )
        )
    return tuple(failures)


def invalidate_reference_artifact(
    reference_root: Path,
    name: str,
    *,
    kind: ReferenceArtifactKind,
) -> tuple[Path, ...]:
    state = reference_artifact_state(reference_root, name, kind=kind)
    removed: list[Path] = []
    for path in (state.dirty_path, state.clean_path):
        if path.exists():
            path.unlink()
            removed.append(path)
    return tuple(removed)


def invalidate_reference_fixture_pair(
    *,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
) -> tuple[Path, ...]:
    removed = [
        *invalidate_reference_artifact(reference_image_root, name, kind="image"),
        *invalidate_reference_artifact(reference_stl_root, name, kind="stl"),
    ]
    return tuple(removed)


def reference_image_bundle_paths(reference_root: Path, bundle_name: str, keys: tuple[str, ...]) -> dict[str, Path]:
    return {key: dirty_reference_path(reference_root, f"{bundle_name}_{key}") for key in keys}


def invalidate_reference_image_bundle(
    reference_root: Path,
    bundle_name: str,
    *,
    keys: tuple[str, ...],
) -> tuple[Path, ...]:
    removed: list[Path] = []
    for key in keys:
        removed.extend(
            invalidate_reference_artifact(reference_root, f"{bundle_name}_{key}", kind="image")
        )
    return tuple(removed)


def validate_camera_contract(
    expected: CameraFramingContract,
    observed: CameraFramingContract,
    *,
    position_tolerance: float = 1e-9,
    target_tolerance: float = 1e-9,
    up_tolerance: float = 1e-9,
    framing_tolerance: float = 1e-9,
) -> tuple[CameraContractViolation, ...]:
    violations: list[CameraContractViolation] = []
    if not np.allclose(expected.position, observed.position, atol=position_tolerance):
        violations.append(CameraContractViolation("pose_drift", expected.position, observed.position))
    if not np.allclose(expected.target, observed.target, atol=target_tolerance):
        violations.append(CameraContractViolation("target_drift", expected.target, observed.target))
    if not np.allclose(expected.up_vector, observed.up_vector, atol=up_tolerance):
        violations.append(CameraContractViolation("up_vector_drift", expected.up_vector, observed.up_vector))
    if expected.projection_mode != observed.projection_mode:
        violations.append(
            CameraContractViolation("projection_drift", expected.projection_mode, observed.projection_mode)
        )
    if expected.window_size != observed.window_size:
        violations.append(CameraContractViolation("crop_drift", expected.window_size, observed.window_size))
    expected_scale = 0.0 if expected.parallel_scale is None else float(expected.parallel_scale)
    observed_scale = 0.0 if observed.parallel_scale is None else float(observed.parallel_scale)
    if not np.isclose(expected_scale, observed_scale, atol=framing_tolerance):
        violations.append(
            CameraContractViolation("framing_drift", expected.parallel_scale, observed.parallel_scale)
        )
    return tuple(violations)


def canonical_object_view_camera_contracts(
    bounds: tuple[float, float, float, float, float, float],
    *,
    window_size: tuple[int, int] = (512, 512),
) -> dict[CanonicalObjectViewName, CameraFramingContract]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array(
        [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)],
        dtype=float,
    )
    spans = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    radius = max(float(np.linalg.norm(spans)), 1.0)
    parallel_scale = max(float(spans.max()) * 0.75, 1.0)

    def contract(position: np.ndarray, up_vector: tuple[float, float, float]) -> CameraFramingContract:
        return CameraFramingContract(
            position=(float(position[0]), float(position[1]), float(position[2])),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up_vector=up_vector,
            projection_mode="parallel",
            window_size=window_size,
            parallel_scale=parallel_scale,
        )

    return {
        "front": contract(center + np.array([0.0, -2.0 * radius, 0.0]), (0.0, 0.0, 1.0)),
        "side": contract(center + np.array([2.0 * radius, 0.0, 0.0]), (0.0, 0.0, 1.0)),
        "top": contract(center + np.array([0.0, 0.0, 2.0 * radius]), (0.0, 1.0, 0.0)),
        "isometric": contract(center + np.array([1.5, -1.5, 1.5]) * radius, (0.0, 0.0, 1.0)),
    }


def render_canonical_object_view_bundle(
    mesh: Mesh,
    output_dir: Path,
    *,
    stem: str,
    view_order: tuple[CanonicalObjectViewName, ...] = ("front", "side", "top", "isometric"),
    silhouette_color: str = "black",
    background: str = "white",
    include_diagnostic_beauty: bool = False,
) -> CanonicalObjectViewBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_contracts = canonical_object_view_camera_contracts(mesh.bounds)
    silhouettes: dict[CanonicalObjectViewName, Path] = {}
    beauty: dict[CanonicalObjectViewName, Path] = {}
    for view_name in view_order:
        contract = camera_contracts[view_name]
        silhouette_path = output_dir / f"{stem}_{view_name}_silhouette.png"
        render_mesh_image_with_camera_contract(
            mesh,
            silhouette_path,
            camera_contract=contract,
            mesh_color=silhouette_color,
            background=background,
        )
        silhouettes[view_name] = silhouette_path
        if include_diagnostic_beauty:
            beauty_path = output_dir / f"{stem}_{view_name}_beauty.png"
            render_mesh_image_with_camera_contract(
                mesh,
                beauty_path,
                camera_contract=contract,
                mesh_color="#5b84b1",
                background=background,
            )
            beauty[view_name] = beauty_path
    return CanonicalObjectViewBundle(
        view_order=view_order,
        silhouettes=silhouettes,
        diagnostic_beauty=beauty or None,
    )


def compare_canonical_object_view_bundles(
    expected: CanonicalObjectViewBundle,
    actual: CanonicalObjectViewBundle,
    *,
    iou_threshold: float = 0.9,
) -> CanonicalObjectViewComparison:
    if expected.view_order != actual.view_order:
        raise AssertionError(
            f"Canonical view order mismatch: expected {expected.view_order}, observed {actual.view_order}."
        )
    per_view: dict[CanonicalObjectViewName, SilhouetteComparison] = {}
    mismatched: list[CanonicalObjectViewName] = []
    for view_name in expected.view_order:
        if view_name not in expected.silhouettes or view_name not in actual.silhouettes:
            raise AssertionError(f"Missing canonical silhouette product for view {view_name}.")
        comparison = compare_silhouette_images(
            expected.silhouettes[view_name],
            actual.silhouettes[view_name],
            iou_threshold=iou_threshold,
        )
        per_view[view_name] = comparison
        if comparison.relationship != "same_shape_same_orientation":
            mismatched.append(view_name)
    return CanonicalObjectViewComparison(per_view=per_view, mismatched_views=tuple(mismatched))


def validate_diagnostic_panel_contract(contract: DiagnosticPanelContract) -> tuple[str, ...]:
    issues: list[str] = []
    _validate_unique_nonempty_strings("panel_order", contract.panel_order)
    if contract.delegated_proof_lane is None and not contract.diagnostic_only:
        issues.append("Diagnostic panels must default to diagnostic_only when no proof lane is delegated.")
    if contract.delegated_proof_lane is not None and contract.diagnostic_only:
        issues.append("Delegated proof use must be explicit rather than marked diagnostic_only.")
    if contract.delegated_proof_lane is not None and not contract.shared_scene:
        issues.append("Proof delegation requires shared_scene=True for panel honesty.")
    return tuple(issues)


def triptych_panel_layout(
    image_path: Path,
    *,
    panel_order: tuple[str, str, str] = ("left", "result", "right"),
    panel_padding: int = 24,
    panel_width: int = 480,
) -> DiagnosticPanelLayout:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: dict[str, tuple[int, int, int, int]] = {}
    cursor_x = panel_padding
    for label in panel_order:
        boxes[label] = (cursor_x, panel_padding, cursor_x + panel_width, height - panel_padding)
        cursor_x += panel_width + panel_padding
    return DiagnosticPanelLayout(
        contract=DiagnosticPanelContract(panel_order=panel_order, diagnostic_only=True, shared_scene=False),
        panel_boxes=boxes,
    )


def extract_panel_regions(
    image_path: Path,
    layout: DiagnosticPanelLayout,
) -> dict[str, Image.Image]:
    contract_issues = validate_diagnostic_panel_contract(layout.contract)
    if contract_issues:
        raise AssertionError(f"Invalid diagnostic panel contract: {contract_issues}")
    extracted: dict[str, Image.Image] = {}
    with Image.open(image_path) as image:
        for label in layout.contract.panel_order:
            left, top, right, bottom = layout.panel_boxes[label]
            extracted[label] = image.crop((left, top, right, bottom))
    return extracted


def validate_handedness_space_anchor_contract(
    contract: HandednessSpaceAnchorContract,
) -> tuple[str, ...]:
    issues: list[str] = []
    for field_name, basis in (
        ("modeling_basis", contract.modeling_basis),
        ("export_basis", contract.export_basis),
        ("viewer_basis", contract.viewer_basis),
    ):
        if len(basis) != 3 or len(set(basis)) != 3:
            issues.append(f"{field_name} must declare three distinct axes.")
        if any(not axis.strip() for axis in basis):
            issues.append(f"{field_name} must not contain blank axes.")
    if contract.camera_contract is None:
        issues.append("camera_contract dependency is required for handedness verification.")
    return tuple(issues)


def compare_handedness_space_anchor_contracts(
    expected: HandednessSpaceAnchorContract,
    observed: HandednessSpaceAnchorContract,
) -> tuple[str, ...]:
    issues = list(validate_handedness_space_anchor_contract(observed))
    if expected.modeling_basis != observed.modeling_basis:
        issues.append("modeling_basis drifted from the declared anchor contract.")
    if expected.export_basis != observed.export_basis:
        issues.append("export_basis drifted from the declared anchor contract.")
    if expected.viewer_basis != observed.viewer_basis:
        issues.append("viewer_basis drifted from the declared anchor contract.")
    if expected.canonical_view != observed.canonical_view:
        issues.append("canonical_view drifted from the declared anchor contract.")
    if expected.camera_contract is not None and observed.camera_contract is not None:
        violations = validate_camera_contract(expected.camera_contract, observed.camera_contract)
        if violations:
            issues.append("camera_contract drifted from the declared anchor contract.")
    return tuple(issues)


def classify_handedness_from_silhouettes(
    expected_path: Path,
    actual_path: Path,
    *,
    canvas_size: int = 256,
    margin: int = 16,
    iou_threshold: float = 0.9,
    witness_similarity_threshold: float = 0.95,
) -> HandednessClassification:
    expected_mask = _image_to_binary_mask(expected_path, foreground_threshold=5)
    actual_mask = _image_to_binary_mask(actual_path, foreground_threshold=5)
    normalized_expected = _normalize_silhouette_mask(expected_mask, canvas_size=canvas_size, margin=margin)
    normalized_actual = _normalize_silhouette_mask(actual_mask, canvas_size=canvas_size, margin=margin)
    mirrored_expected = np.fliplr(normalized_expected)
    same_iou = _mask_iou(normalized_expected, normalized_actual)
    mirrored_iou = _mask_iou(mirrored_expected, normalized_actual)
    witness_self_mirror_iou = _mask_iou(normalized_expected, mirrored_expected)
    witness_adequate = witness_self_mirror_iou < witness_similarity_threshold
    if not witness_adequate:
        result_class: HandednessResultClass = "orientation_unknown"
    elif same_iou >= iou_threshold:
        result_class = "same_handedness"
    elif mirrored_iou >= iou_threshold:
        result_class = "mirrored"
    else:
        result_class = "orientation_unknown"
    return HandednessClassification(
        result_class=result_class,
        same_iou=same_iou,
        mirrored_iou=mirrored_iou,
        witness_adequate=witness_adequate,
    )


def ensure_reference_image(
    *,
    rendered_path: Path,
    reference_root: Path,
    name: str,
    update_dirty_reference_images: bool,
    mean_abs_delta_threshold: float = 1.0,
) -> Path:
    state = reference_artifact_state(reference_root, name, kind="image")
    if update_dirty_reference_images:
        state.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, state.dirty_path)
        return state.dirty_path
    if state.selected_path is None:
        state.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, state.dirty_path)
        return state.dirty_path
    mean_abs_delta = image_mean_abs_delta(rendered_path, state.selected_path)
    if mean_abs_delta > mean_abs_delta_threshold:
        raise AssertionError(
            f"Reference image changed for {name}: mean_abs_delta={mean_abs_delta:.3f} "
            f"(threshold={mean_abs_delta_threshold:.3f}) against {state.selected_path}."
        )
    return state.selected_path


def ensure_reference_stl(
    *,
    rendered_path: Path,
    reference_root: Path,
    name: str,
    update_dirty_reference_images: bool,
) -> Path:
    state = reference_artifact_state(reference_root, name, kind="stl")
    if update_dirty_reference_images:
        state.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, state.dirty_path)
        return state.dirty_path
    if state.selected_path is None:
        state.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, state.dirty_path)
        return state.dirty_path
    rendered_text = canonicalize_stl_text(rendered_path.read_text())
    reference_text = canonicalize_stl_text(state.selected_path.read_text())
    if rendered_text != reference_text:
        raise AssertionError(f"Reference STL changed for {name} against {state.selected_path}.")
    return state.selected_path


def ensure_reference_fixture_pair(
    *,
    rendered_image_path: Path,
    rendered_stl_path: Path,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
    update_dirty_reference_images: bool,
    mean_abs_delta_threshold: float = 1.0,
) -> tuple[Path, Path]:
    state = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    )
    if not update_dirty_reference_images and state.has_partial_group:
        missing_kind = "image" if not state.image.exists else "STL"
        raise AssertionError(
            f"Incomplete reference fixture {name}: missing {missing_kind} counterpart. "
            "Model-output reference fixtures must keep image and STL artifacts together."
        )
    image_reference = ensure_reference_image(
        rendered_path=rendered_image_path,
        reference_root=reference_image_root,
        name=name,
        update_dirty_reference_images=update_dirty_reference_images,
        mean_abs_delta_threshold=mean_abs_delta_threshold,
    )
    stl_reference = ensure_reference_stl(
        rendered_path=rendered_stl_path,
        reference_root=reference_stl_root,
        name=name,
        update_dirty_reference_images=update_dirty_reference_images,
    )
    return image_reference, stl_reference


def assess_cv_result(contract: CvFixtureContract, lane_result_class: str) -> CvResultAssessment:
    if lane_result_class not in contract.known_result_classes:
        raise AssertionError(
            f"Result class {lane_result_class!r} is not declared for CV fixture {contract.fixture_id}."
        )
    if lane_result_class in contract.unknown_result_classes:
        pattern: SharedCvResultPattern = "unknown"
    elif lane_result_class in contract.different_result_classes:
        pattern = "different"
    elif lane_result_class in contract.transformed_result_classes:
        pattern = "transformed"
    else:
        pattern = "positive"
    return CvResultAssessment(
        lane_result_class=lane_result_class,
        shared_pattern=pattern,
        passes_contract=lane_result_class in contract.passing_result_classes,
    )


def ensure_complete_cv_artifact_bundle(bundle: CvArtifactBundle) -> CvArtifactBundle:
    missing_keys = [key for key in bundle.contract.required_keys if key not in bundle.artifacts]
    if missing_keys:
        raise AssertionError(
            f"Incomplete CV artifact bundle for {bundle.contract.fixture_id}: missing keys {missing_keys}."
        )
    missing_files = [key for key, path in bundle.artifacts.items() if not Path(path).exists()]
    if missing_files:
        raise AssertionError(
            f"Incomplete CV artifact bundle for {bundle.contract.fixture_id}: missing files for {missing_files}."
        )
    return bundle


def ensure_reference_image_bundle(
    *,
    bundle: CvArtifactBundle,
    reference_root: Path,
    bundle_name: str,
    update_dirty_reference_images: bool,
    mean_abs_delta_threshold: float = 1.0,
) -> dict[str, Path]:
    ensure_complete_cv_artifact_bundle(bundle)
    states = {
        key: reference_artifact_state(reference_root, f"{bundle_name}_{key}", kind="image")
        for key in bundle.contract.required_keys
    }
    existing_keys = {key for key, state in states.items() if state.exists}
    if not update_dirty_reference_images and existing_keys and existing_keys != set(bundle.contract.required_keys):
        missing_keys = sorted(set(bundle.contract.required_keys).difference(existing_keys))
        raise AssertionError(
            f"Incomplete CV artifact bundle for {bundle.contract.fixture_id}: missing existing references for "
            f"{missing_keys}. Bundle products must stay complete together."
        )
    references: dict[str, Path] = {}
    for key in bundle.contract.required_keys:
        references[key] = ensure_reference_image(
            rendered_path=bundle.artifacts[key],
            reference_root=reference_root,
            name=f"{bundle_name}_{key}",
            update_dirty_reference_images=update_dirty_reference_images,
            mean_abs_delta_threshold=mean_abs_delta_threshold,
        )
    return references


def text_cv_expected_loops(
    *,
    content: str,
    font_path: str,
    font_size: float = 1.0,
    justify: str = "left",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
) -> list[np.ndarray]:
    sections = text_profiles(
        content,
        font_size=font_size,
        justify=justify,
        valign=valign,
        letter_spacing=letter_spacing,
        line_height=line_height,
        font_path=font_path,
    )
    loops: list[np.ndarray] = []
    for section in sections:
        for region in section.regions:
            loops.append(np.asarray(region.outer.points, dtype=float))
            for hole in region.holes:
                loops.append(np.asarray(hole.points, dtype=float))
    return loops


def text_cv_actual_loops(
    body: SurfaceBody,
    *,
    slice_z: float,
    normalize_readable_orientation: bool = True,
) -> list[np.ndarray]:
    mesh = tessellate_surface_body(body).mesh
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(slice_z)),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    loops = [polyline.points[:, :2] for polyline in result.polylines if polyline.closed]
    if not normalize_readable_orientation or not loops:
        return loops
    points = np.vstack(loops)
    x_span = float(points[:, 0].max() - points[:, 0].min())
    y_span = float(points[:, 1].max() - points[:, 1].min())
    if y_span <= x_span:
        return loops
    return [np.column_stack((-loop[:, 1], loop[:, 0])) for loop in loops]


def initial_text_cv_scope_support(
    content: str,
    *,
    font_path: str,
    allowed_characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
) -> tuple[bool, tuple[str, ...]]:
    if "\n" in content:
        return False, tuple()
    if any(character not in allowed_characters for character in content):
        return False, tuple()
    try:
        font = TTFont(font_path)
    except TTLibFileIsCollectionError:
        font = TTFont(font_path, fontNumber=0)
    cmap = font.getBestCmap() or {}
    missing = tuple(character for character in content if character != " " and ord(character) not in cmap)
    return True, missing


def compare_text_loop_silhouettes(
    *,
    content: str,
    font_path: str,
    actual_loops: list[np.ndarray] | tuple[np.ndarray, ...],
    expected_loops: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    font_size: float = 1.0,
    justify: str = "left",
    valign: str = "baseline",
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (1024, 512),
    canvas_size: int = 256,
    margin: int = 16,
    iou_threshold: float = 0.9,
) -> TextCvComparison:
    in_scope, missing_glyphs = initial_text_cv_scope_support(content, font_path=font_path)
    if expected_loops is None:
        expected_loops = text_cv_expected_loops(
            content=content,
            font_path=font_path,
            font_size=font_size,
            justify=justify,
            valign=valign,
            letter_spacing=letter_spacing,
            line_height=line_height,
        )
    if not expected_loops or not actual_loops:
        return TextCvComparison(
            result_class="unreadable",
            same_orientation_iou=0.0,
            best_rotation_iou=0.0,
            best_rotation_deg=0,
            best_mirror_iou=0.0,
            best_mirror_deg=0,
            fallback_detected=bool(missing_glyphs),
            missing_glyphs=missing_glyphs,
            in_scope=in_scope,
        )
    if not in_scope:
        return TextCvComparison(
            result_class="unreadable",
            same_orientation_iou=0.0,
            best_rotation_iou=0.0,
            best_rotation_deg=0,
            best_mirror_iou=0.0,
            best_mirror_deg=0,
            fallback_detected=False,
            missing_glyphs=missing_glyphs,
            in_scope=False,
        )
    if missing_glyphs:
        return TextCvComparison(
            result_class="different_text",
            same_orientation_iou=0.0,
            best_rotation_iou=0.0,
            best_rotation_deg=0,
            best_mirror_iou=0.0,
            best_mirror_deg=0,
            fallback_detected=True,
            missing_glyphs=missing_glyphs,
            in_scope=True,
        )
    shared_bounds = bounds if bounds is not None else planar_loop_bounds(expected_loops, actual_loops)
    expected_mask = _planar_section_fill_mask(expected_loops, bounds=shared_bounds, image_size=image_size)
    actual_mask = _planar_section_fill_mask(actual_loops, bounds=shared_bounds, image_size=image_size)
    normalized_expected = _normalize_silhouette_mask(expected_mask, canvas_size=canvas_size, margin=margin)
    normalized_actual = _normalize_silhouette_mask(actual_mask, canvas_size=canvas_size, margin=margin)
    same_orientation_iou = _mask_iou(normalized_expected, normalized_actual)
    if same_orientation_iou >= iou_threshold:
        return TextCvComparison(
            result_class="same_text_same_orientation",
            same_orientation_iou=same_orientation_iou,
            best_rotation_iou=same_orientation_iou,
            best_rotation_deg=0,
            best_mirror_iou=same_orientation_iou,
            best_mirror_deg=0,
            fallback_detected=False,
            missing_glyphs=(),
            in_scope=True,
        )

    best_rotation_deg = 0
    best_rotation_iou = same_orientation_iou
    for rotation_deg in (90, 180, 270):
        rotated_actual = np.rot90(normalized_actual, k=rotation_deg // 90)
        rotation_iou = _mask_iou(normalized_expected, rotated_actual)
        if rotation_iou > best_rotation_iou:
            best_rotation_iou = rotation_iou
            best_rotation_deg = rotation_deg

    mirrored_actual = np.fliplr(normalized_actual)
    best_mirror_deg = 0
    best_mirror_iou = _mask_iou(normalized_expected, mirrored_actual)
    for rotation_deg in (90, 180, 270):
        rotated_mirror = np.rot90(mirrored_actual, k=rotation_deg // 90)
        mirror_iou = _mask_iou(normalized_expected, rotated_mirror)
        if mirror_iou > best_mirror_iou:
            best_mirror_iou = mirror_iou
            best_mirror_deg = rotation_deg

    if best_rotation_iou >= iou_threshold:
        result_class: TextCvResultClass = "same_text_rotated"
    elif best_mirror_iou >= iou_threshold:
        result_class = "same_text_mirrored"
    else:
        result_class = "different_text"
    return TextCvComparison(
        result_class=result_class,
        same_orientation_iou=same_orientation_iou,
        best_rotation_iou=best_rotation_iou,
        best_rotation_deg=best_rotation_deg,
        best_mirror_iou=best_mirror_iou,
        best_mirror_deg=best_mirror_deg,
        fallback_detected=False,
        missing_glyphs=(),
        in_scope=True,
    )


def image_signal_stats(path: Path, *, background_threshold: int = 245) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    non_background = np.any(arr < background_threshold, axis=2)
    occupancy = float(non_background.mean())
    return {
        "width": float(arr.shape[1]),
        "height": float(arr.shape[0]),
        "occupancy": occupancy,
        "mean_luma": float(arr.mean()),
        "std_luma": float(arr.std()),
    }


def image_mean_abs_delta(path_a: Path, path_b: Path) -> float:
    image_a = Image.open(path_a).convert("RGB")
    image_b = Image.open(path_b).convert("RGB")
    if image_a.size != image_b.size:
        raise AssertionError(f"Reference image size mismatch: {image_a.size} != {image_b.size}")
    arr_a = np.asarray(image_a, dtype=np.int16)
    arr_b = np.asarray(image_b, dtype=np.int16)
    return float(np.abs(arr_a - arr_b).mean())


def stl_signal_stats(path: Path) -> dict[str, float]:
    text = path.read_text()
    facet_count = text.count("\n  facet normal ")
    vertex_count = text.count("\n      vertex ")
    return {
        "file_size": float(path.stat().st_size),
        "facet_count": float(facet_count),
        "vertex_count": float(vertex_count),
        "line_count": float(len(text.splitlines())),
    }


def compare_silhouette_images(
    expected_path: Path,
    actual_path: Path,
    *,
    foreground_threshold: int = 5,
    canvas_size: int = 256,
    margin: int = 16,
    iou_threshold: float = 0.9,
) -> SilhouetteComparison:
    expected_mask = _image_to_binary_mask(expected_path, foreground_threshold=foreground_threshold)
    actual_mask = _image_to_binary_mask(actual_path, foreground_threshold=foreground_threshold)
    return compare_silhouette_masks(
        expected_mask,
        actual_mask,
        canvas_size=canvas_size,
        margin=margin,
        iou_threshold=iou_threshold,
    )


def compare_planar_loop_silhouettes(
    expected_loops: list[np.ndarray] | tuple[np.ndarray, ...],
    actual_loops: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (512, 512),
    canvas_size: int = 256,
    margin: int = 16,
    iou_threshold: float = 0.9,
) -> SilhouetteComparison:
    shared_bounds = bounds if bounds is not None else planar_loop_bounds(expected_loops, actual_loops)
    expected_mask = _planar_section_fill_mask(expected_loops, bounds=shared_bounds, image_size=image_size)
    actual_mask = _planar_section_fill_mask(actual_loops, bounds=shared_bounds, image_size=image_size)
    return compare_silhouette_masks(
        expected_mask,
        actual_mask,
        canvas_size=canvas_size,
        margin=margin,
        iou_threshold=iou_threshold,
    )


def compare_silhouette_masks(
    expected_mask: np.ndarray,
    actual_mask: np.ndarray,
    *,
    canvas_size: int = 256,
    margin: int = 16,
    iou_threshold: float = 0.9,
) -> SilhouetteComparison:
    normalized_expected = _normalize_silhouette_mask(expected_mask, canvas_size=canvas_size, margin=margin)
    normalized_actual = _normalize_silhouette_mask(actual_mask, canvas_size=canvas_size, margin=margin)
    same_orientation_iou = _mask_iou(normalized_expected, normalized_actual)
    if same_orientation_iou >= iou_threshold:
        return SilhouetteComparison(
            relationship="same_shape_same_orientation",
            same_orientation_iou=same_orientation_iou,
            best_rotation_iou=same_orientation_iou,
            best_rotation_deg=0,
        )

    best_rotation_deg = 0
    best_rotation_iou = same_orientation_iou
    for rotation_deg in (90, 180, 270):
        rotated_actual = np.rot90(normalized_actual, k=rotation_deg // 90)
        rotation_iou = _mask_iou(normalized_expected, rotated_actual)
        if rotation_iou > best_rotation_iou:
            best_rotation_iou = rotation_iou
            best_rotation_deg = rotation_deg
    relationship: SilhouetteRelationship = (
        "same_shape_rotated" if best_rotation_iou >= iou_threshold else "different_shape"
    )
    return SilhouetteComparison(
        relationship=relationship,
        same_orientation_iou=same_orientation_iou,
        best_rotation_iou=best_rotation_iou,
        best_rotation_deg=best_rotation_deg,
    )


def canonicalize_stl_text(text: str) -> str:
    canonical_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("facet normal "):
            continue
        canonical_lines.append(stripped)
    return "\n".join(canonical_lines)


def _combined_collection_mesh(
    collection: SurfaceConsumerCollection,
    *,
    request: TessellationRequest | None = None,
) -> Mesh:
    meshes = [tessellate_surface_body(record.body, request).mesh for record in collection.items]
    if not meshes:
        return Mesh(np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int))
    return combine_meshes(meshes)


def _camera_position_for_bounds(bounds: tuple[float, float, float, float, float, float]) -> list[tuple[float, float, float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ],
        dtype=float,
    )
    diagonal = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    radius = max(float(diagonal), 1.0)
    eye = center + np.array([1.8, -1.6, 1.4], dtype=float) * radius
    return [
        (float(eye[0]), float(eye[1]), float(eye[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        (0.0, 0.0, 1.0),
    ]


def _image_to_binary_mask(path: Path, *, foreground_threshold: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    mask = np.asarray(image, dtype=np.uint8) > int(foreground_threshold)
    return mask


def _normalize_silhouette_mask(mask: np.ndarray, *, canvas_size: int, margin: int) -> np.ndarray:
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim != 2:
        raise ValueError("Silhouette masks must be two-dimensional.")
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        raise ValueError("Silhouette comparison requires a non-empty foreground mask.")
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    cropped = binary_mask[ymin : ymax + 1, xmin : xmax + 1]
    usable_size = max(int(canvas_size) - (int(margin) * 2), 1)
    scale = usable_size / float(max(cropped.shape))
    target_height = max(1, int(round(cropped.shape[0] * scale)))
    target_width = max(1, int(round(cropped.shape[1] * scale)))
    resized = transform.resize(
        cropped.astype(float),
        (target_height, target_width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) >= 0.5
    dilated = morphology.dilation(resized, morphology.disk(1))
    canvas = np.zeros((int(canvas_size), int(canvas_size)), dtype=bool)
    y_offset = (canvas.shape[0] - dilated.shape[0]) // 2
    x_offset = (canvas.shape[1] - dilated.shape[1]) // 2
    canvas[y_offset : y_offset + dilated.shape[0], x_offset : x_offset + dilated.shape[1]] = dilated
    return canvas


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    intersection = np.logical_and(left_mask, right_mask).sum()
    union = np.logical_or(left_mask, right_mask).sum()
    if union == 0:
        raise ValueError("Silhouette comparison requires non-empty masks.")
    return float(intersection) / float(union)


def _validate_unique_nonempty_strings(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain blank values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates.")


def _validate_optional_unique_strings(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        return
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain blank values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates.")


def _planar_section_mask(
    loops: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] = (512, 512),
    stroke_width: int = 4,
    background: int = 0,
    foreground: int = 255,
) -> np.ndarray:
    loop_arrays = [np.asarray(loop, dtype=float).reshape(-1, 2) for loop in loops]
    if not loop_arrays:
        raise ValueError("_planar_section_mask requires at least one loop.")
    shared_bounds = bounds if bounds is not None else planar_loop_bounds(loop_arrays)
    image = Image.new("L", image_size, color=int(background))
    draw = ImageDraw.Draw(image)
    for loop in loop_arrays:
        points = [_planar_to_pixel(point, shared_bounds, image_size) for point in loop]
        if len(points) >= 2:
            points.append(points[0])
            draw.line(points, fill=int(foreground), width=int(stroke_width))
    return np.asarray(image, dtype=np.uint8)


def _planar_section_fill_mask(
    loops: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    loop_arrays = [np.asarray(loop, dtype=float).reshape(-1, 2) for loop in loops]
    if not loop_arrays:
        raise ValueError("_planar_section_fill_mask requires at least one loop.")
    accumulated = np.zeros(image_size[::-1], dtype=bool)
    for loop in loop_arrays:
        points = [_planar_to_pixel(point, bounds, image_size) for point in loop]
        polygon = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(polygon)
        if len(points) >= 3:
            draw.polygon(points, fill=255)
        accumulated ^= np.asarray(polygon, dtype=np.uint8) > 0
    return accumulated


def _planar_to_pixel(
    point_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    width, height = image_size
    x_span = max(xmax - xmin, 1e-9)
    y_span = max(ymax - ymin, 1e-9)
    usable_w = max(width - 1, 1)
    usable_h = max(height - 1, 1)
    scale = min(usable_w / x_span, usable_h / y_span)
    x_offset = (width - (x_span * scale)) / 2.0
    y_offset = (height - (y_span * scale)) / 2.0
    x = x_offset + (float(point_xy[0]) - xmin) * scale
    y = height - (y_offset + (float(point_xy[1]) - ymin) * scale)
    return (x, y)
