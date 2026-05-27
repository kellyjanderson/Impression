from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, Tuple, cast

import numpy as np

from impression.mesh import Mesh, triangulate_faces

from ._color import set_mesh_color
if TYPE_CHECKING:
    from .surface import SurfaceBody

Backend = Literal["mesh", "surface"]
LegacyPrimitiveMeshAssumptionKind = Literal[
    "surface-native consumer",
    "tessellation-boundary consumer",
    "explicit mesh compatibility consumer",
    "obsolete mesh-primary test",
]

_MESH_REQUIRED_CALLS = (
    "mesh_to_pyvista",
    "combine_meshes",
    "union_meshes",
    "export_stl",
    "save_stl",
    "repair_mesh",
    "analyze_mesh",
)
_MESH_ATTRIBUTE_NAMES = ("vertices", "faces", "n_faces")


@dataclass(frozen=True)
class PrimitiveCSGRouteRecord:
    """Primitive authored route covered by the no-hidden-mesh CSG policy."""

    caller_id: str
    surface_constructor: str
    explicit_mesh_constructor: str
    csg_gate: str = "assert_no_hidden_surface_csg_mesh_fallback"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "csg_gate": self.csg_gate,
            "explicit_mesh_constructor": self.explicit_mesh_constructor,
            "surface_constructor": self.surface_constructor,
        }


@dataclass(frozen=True)
class PrimitivePatchProducerSelectionRecord:
    """Surface-native patch producer chosen for a public primitive call."""

    caller_id: str
    primitive: str
    selected_patch_families: tuple[str, ...]
    surface_constructor: str
    exact_surface_truth: bool = True
    mesh_substitution_allowed: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "exact_surface_truth": self.exact_surface_truth,
            "mesh_substitution_allowed": self.mesh_substitution_allowed,
            "primitive": self.primitive,
            "selected_patch_families": self.selected_patch_families,
            "surface_constructor": self.surface_constructor,
        }


@dataclass(frozen=True)
class LegacyPrimitiveMeshAssumptionClassificationRecord:
    """Classification for a primitive call in a mesh-sensitive context."""

    classification: LegacyPrimitiveMeshAssumptionKind
    rewrite_rule: str
    stale_assumption: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "rewrite_rule": self.rewrite_rule,
            "stale_assumption": self.stale_assumption,
        }


@dataclass(frozen=True)
class LegacyPrimitiveMeshAssumptionFindingRecord:
    """Repository inventory finding for primitive calls that may assume mesh output."""

    path: str
    line_number: int
    primitive_constructor: str
    context: str
    classification: LegacyPrimitiveMeshAssumptionClassificationRecord
    diagnostic: str = ""

    def __post_init__(self) -> None:
        line_number = int(self.line_number)
        if line_number < 1:
            raise ValueError("Legacy primitive mesh assumption findings require a positive line number.")
        object.__setattr__(self, "line_number", line_number)

    @property
    def stale_assumption(self) -> bool:
        return self.classification.stale_assumption

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classification": self.classification.canonical_payload(),
            "context": self.context,
            "diagnostic": self.diagnostic,
            "line_number": self.line_number,
            "path": self.path,
            "primitive_constructor": self.primitive_constructor,
            "stale_assumption": self.stale_assumption,
        }


@dataclass(frozen=True)
class LegacyPrimitiveMeshAssumptionInventoryReport:
    """Bounded inventory report for stale public-primitive mesh assumptions."""

    findings: tuple[LegacyPrimitiveMeshAssumptionFindingRecord, ...]

    @property
    def stale_findings(self) -> tuple[LegacyPrimitiveMeshAssumptionFindingRecord, ...]:
        return tuple(finding for finding in self.findings if finding.stale_assumption)

    @property
    def passed(self) -> bool:
        return not self.stale_findings

    def canonical_payload(self) -> dict[str, object]:
        return {
            "finding_count": len(self.findings),
            "findings": [finding.canonical_payload() for finding in self.findings],
            "passed": self.passed,
            "stale_count": len(self.stale_findings),
        }


@dataclass(frozen=True)
class UnsupportedPrimitiveProducerDiagnostic:
    """Explicit diagnostic for a primitive request that cannot produce surface truth."""

    caller_id: str
    primitive: str
    requested_backend: str
    reason: str

    @property
    def message(self) -> str:
        return (
            f"{self.caller_id} cannot produce surface truth for primitive {self.primitive!r}: "
            f"{self.reason}"
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "primitive": self.primitive,
            "reason": self.reason,
            "requested_backend": self.requested_backend,
        }


PRIMITIVE_CSG_ROUTE_INVENTORY: tuple[PrimitiveCSGRouteRecord, ...] = (
    PrimitiveCSGRouteRecord("primitive.make_box", "make_box", "make_box_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_cylinder", "make_cylinder", "make_cylinder_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_ngon", "make_ngon", "make_ngon_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_polyhedron", "make_polyhedron", "make_polyhedron_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_nhedron", "make_nhedron", "make_nhedron_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_sphere", "make_sphere", "make_sphere_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_torus", "make_torus", "make_torus_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_cone", "make_cone", "make_cone_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_prism", "make_prism", "make_prism_mesh"),
)
PRIMITIVE_PATCH_PRODUCER_SELECTIONS: tuple[PrimitivePatchProducerSelectionRecord, ...] = (
    PrimitivePatchProducerSelectionRecord("primitive.make_box", "box", ("planar",), "make_surface_box"),
    PrimitivePatchProducerSelectionRecord("primitive.make_cylinder", "cylinder", ("planar", "revolution"), "make_surface_cylinder"),
    PrimitivePatchProducerSelectionRecord("primitive.make_ngon", "ngon", ("planar", "ruled"), "make_surface_ngon"),
    PrimitivePatchProducerSelectionRecord("primitive.make_polyhedron", "polyhedron", ("planar",), "make_surface_polyhedron"),
    PrimitivePatchProducerSelectionRecord("primitive.make_nhedron", "nhedron", ("planar",), "make_surface_nhedron"),
    PrimitivePatchProducerSelectionRecord("primitive.make_sphere", "sphere", ("revolution",), "make_surface_sphere"),
    PrimitivePatchProducerSelectionRecord("primitive.make_torus", "torus", ("revolution",), "make_surface_torus"),
    PrimitivePatchProducerSelectionRecord("primitive.make_cone", "cone", ("planar", "revolution"), "make_surface_cone"),
    PrimitivePatchProducerSelectionRecord("primitive.make_prism", "prism", ("planar", "ruled"), "make_surface_prism"),
)
_PRIMITIVE_PATCH_PRODUCER_SELECTION_BY_CALLER = {
    record.caller_id: record for record in PRIMITIVE_PATCH_PRODUCER_SELECTIONS
}


def primitive_csg_route_inventory() -> tuple[PrimitiveCSGRouteRecord, ...]:
    """Return primitive authored routes guarded against hidden mesh fallback."""

    return PRIMITIVE_CSG_ROUTE_INVENTORY


def primitive_patch_producer_selection_inventory() -> tuple[PrimitivePatchProducerSelectionRecord, ...]:
    """Return deterministic surface patch family selections for public primitives."""

    return PRIMITIVE_PATCH_PRODUCER_SELECTIONS


def _primitive_constructor_names() -> tuple[str, ...]:
    return tuple(record.surface_constructor for record in PRIMITIVE_CSG_ROUTE_INVENTORY)


def classify_legacy_primitive_mesh_assumption(
    context: str,
    primitive_constructor: str,
) -> LegacyPrimitiveMeshAssumptionClassificationRecord:
    """Classify one primitive call site against the explicit tessellation-boundary policy."""

    stripped = " ".join(str(context).strip().split())
    primitive = str(primitive_constructor).strip()
    if not stripped:
        raise ValueError("Legacy primitive mesh assumption context must be non-empty.")
    if primitive not in _primitive_constructor_names():
        raise ValueError(f"Unsupported primitive constructor {primitive_constructor!r}.")
    if f"{primitive}_mesh(" in stripped or f"backend=\"mesh\"" in stripped or f"backend='mesh'" in stripped:
        return LegacyPrimitiveMeshAssumptionClassificationRecord(
            classification="explicit mesh compatibility consumer",
            rewrite_rule="keep mesh-specific intent behind make_*_mesh(...) or backend='mesh'",
        )
    if "tessellate_surface_body(" in stripped or "mesh_from_surface_body(" in stripped:
        return LegacyPrimitiveMeshAssumptionClassificationRecord(
            classification="tessellation-boundary consumer",
            rewrite_rule="keep the visible tessellation boundary around the surface primitive",
        )
    mesh_call = any(f"{name}(" in stripped for name in _MESH_REQUIRED_CALLS)
    mesh_attribute = any(f".{name}" in stripped for name in _MESH_ATTRIBUTE_NAMES)
    if mesh_call or mesh_attribute:
        return LegacyPrimitiveMeshAssumptionClassificationRecord(
            classification="obsolete mesh-primary test",
            rewrite_rule="rewrite to tessellate_surface_body(make_*(...)) or use make_*_mesh(...) when testing mesh compatibility",
            stale_assumption=True,
        )
    return LegacyPrimitiveMeshAssumptionClassificationRecord(
        classification="surface-native consumer",
        rewrite_rule="keep the public primitive as a SurfaceBody assertion",
    )


def inventory_legacy_primitive_mesh_assumptions(
    sources: Mapping[str, str],
) -> LegacyPrimitiveMeshAssumptionInventoryReport:
    """Inventory primitive call sites in supplied source text for stale mesh assumptions."""

    findings: list[LegacyPrimitiveMeshAssumptionFindingRecord] = []
    primitive_names = _primitive_constructor_names()
    assignment_by_variable: dict[str, str] = {}
    assignment_pattern = re.compile(
        rf"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<primitive>{'|'.join(primitive_names)})\("
    )
    for path, source in sources.items():
        assignment_by_variable.clear()
        lines = str(source).splitlines()
        for line_number, line in enumerate(lines, start=1):
            assignment = assignment_pattern.search(line)
            if assignment is not None:
                assignment_by_variable[assignment.group("name")] = assignment.group("primitive")
            for route in PRIMITIVE_CSG_ROUTE_INVENTORY:
                if f"{route.explicit_mesh_constructor}(" not in line:
                    continue
                classification = classify_legacy_primitive_mesh_assumption(line, route.surface_constructor)
                findings.append(
                    LegacyPrimitiveMeshAssumptionFindingRecord(
                        path=str(path),
                        line_number=line_number,
                        primitive_constructor=route.surface_constructor,
                        context=line.strip(),
                        classification=classification,
                    )
                )
            for primitive in primitive_names:
                if f"{primitive}(" not in line:
                    continue
                classification = classify_legacy_primitive_mesh_assumption(line, primitive)
                if classification.classification == "surface-native consumer" and assignment is not None:
                    continue
                diagnostic = ""
                if classification.stale_assumption:
                    diagnostic = (
                        f"{path}:{line_number} passes {primitive}(...) into a mesh-sensitive context; "
                        f"{classification.rewrite_rule}."
                    )
                findings.append(
                    LegacyPrimitiveMeshAssumptionFindingRecord(
                        path=str(path),
                        line_number=line_number,
                        primitive_constructor=primitive,
                        context=line.strip(),
                        classification=classification,
                        diagnostic=diagnostic,
                    )
                )
            for variable, primitive in assignment_by_variable.items():
                if not any(f"{variable}.{attribute}" in line for attribute in _MESH_ATTRIBUTE_NAMES):
                    continue
                classification = classify_legacy_primitive_mesh_assumption(line, primitive)
                diagnostic = (
                    f"{path}:{line_number} accesses mesh attributes on {variable} assigned from {primitive}(...); "
                    f"{classification.rewrite_rule}."
                )
                findings.append(
                    LegacyPrimitiveMeshAssumptionFindingRecord(
                        path=str(path),
                        line_number=line_number,
                        primitive_constructor=primitive,
                        context=line.strip(),
                        classification=classification,
                        diagnostic=diagnostic,
                    )
                )
    return LegacyPrimitiveMeshAssumptionInventoryReport(findings=tuple(findings))


def scan_legacy_primitive_mesh_assumptions(
    root: str | Path,
    *,
    suffixes: tuple[str, ...] = (".py", ".md"),
) -> LegacyPrimitiveMeshAssumptionInventoryReport:
    """Scan a repository tree for stale primitive mesh assumptions."""

    root_path = Path(root)
    sources: dict[str, str] = {}
    for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.suffix in suffixes):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        sources[str(path.relative_to(root_path))] = path.read_text(encoding="utf-8")
    return inventory_legacy_primitive_mesh_assumptions(sources)


def select_primitive_patch_producer(caller_id: str) -> PrimitivePatchProducerSelectionRecord:
    """Return the surface-native producer selection for a primitive caller."""

    try:
        return _PRIMITIVE_PATCH_PRODUCER_SELECTION_BY_CALLER[caller_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported primitive surface producer caller_id {caller_id!r}.") from exc


def unsupported_primitive_producer_diagnostic(
    caller_id: str,
    *,
    requested_backend: str,
    reason: str,
) -> UnsupportedPrimitiveProducerDiagnostic:
    """Build an explicit unsupported producer diagnostic without falling back to mesh."""

    primitive = caller_id.rsplit(".", maxsplit=1)[-1].removeprefix("make_")
    return UnsupportedPrimitiveProducerDiagnostic(
        caller_id=caller_id,
        primitive=primitive,
        requested_backend=requested_backend,
        reason=reason,
    )


def _ensure_backend(backend: Backend) -> None:
    if backend not in {"mesh", "surface"}:
        raise ValueError(
            f"Unsupported backend '{backend}'. Only 'mesh' and 'surface' are available right now."
        )


def _surface_metadata(*, color: Sequence[float] | str | None) -> dict[str, object] | None:
    if color is None:
        return None
    return {"consumer": {"color": color}}


def _surface_primitive_result(caller_id: str, result: SurfaceBody) -> SurfaceBody:
    from .csg import assert_no_hidden_surface_csg_mesh_fallback
    from .surface import SurfaceBody

    selection = select_primitive_patch_producer(caller_id)
    checked = assert_no_hidden_surface_csg_mesh_fallback(caller_id, result)
    if not isinstance(checked, SurfaceBody):
        diagnostic = unsupported_primitive_producer_diagnostic(
            caller_id,
            requested_backend="surface",
            reason=f"producer returned {type(checked).__name__}, expected SurfaceBody",
        )
        raise TypeError(diagnostic.message)
    emitted_families = {patch.family for patch in checked.iter_patches()}
    expected_families = set(selection.selected_patch_families)
    if not emitted_families <= expected_families:
        diagnostic = unsupported_primitive_producer_diagnostic(
            caller_id,
            requested_backend="surface",
            reason=(
                f"producer emitted patch families {sorted(emitted_families)} outside "
                f"selected families {sorted(expected_families)}"
            ),
        )
        raise ValueError(diagnostic.message)
    return cast("SurfaceBody", checked)


def make_box(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Axis-aligned box specified by size (dx, dy, dz) and center."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_box

        return _surface_primitive_result(
            "primitive.make_box",
            make_surface_box(size=size, center=center, metadata=_surface_metadata(color=color)),
        )

    from ._legacy_mesh_primitives import box_mesh

    mesh = box_mesh(size, center)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    capping: bool = True,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Right circular cylinder aligned with `direction`."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_cylinder

        return _surface_primitive_result(
            "primitive.make_cylinder",
            make_surface_cylinder(
                radius=radius,
                height=height,
                center=center,
                direction=direction,
                resolution=resolution,
                capping=capping,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(radius, radius, height, resolution, capping=capping)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_ngon(
    sides: int = 6,
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Mesh | SurfaceBody:
    """Regular n-gon prism aligned to `direction`."""

    _ensure_backend(backend)
    sides = int(sides)
    if sides < 3:
        raise ValueError("sides must be >= 3.")
    if side_length is not None:
        inferred = float(side_length) / (2.0 * np.sin(np.pi / sides))
        if radius != 0.5 and not np.isclose(radius, inferred):
            raise ValueError("Specify either radius or side_length, not both.")
        radius = inferred

    if backend == "surface":
        from ._surface_primitives import make_surface_ngon

        return _surface_primitive_result(
            "primitive.make_ngon",
            make_surface_ngon(
                sides=sides,
                radius=radius,
                height=height,
                center=center,
                direction=direction,
                side_length=side_length,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(radius, radius, height, sides, capping=True)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_polyhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Regular polyhedron specified by number of faces (4, 6, 8, 12, 20)."""

    _ensure_backend(backend)
    faces = int(faces)
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if backend == "surface":
        from ._surface_primitives import make_surface_polyhedron

        return _surface_primitive_result(
            "primitive.make_polyhedron",
            make_surface_polyhedron(
                faces=faces,
                radius=radius,
                center=center,
                metadata=_surface_metadata(color=color),
            ),
        )

    vertices, face_list = _regular_polyhedron_data(faces)
    vertices = np.asarray(vertices, dtype=float)
    faces_arr = triangulate_faces(face_list)
    if faces_arr.size:
        faces_arr = _orient_faces_outward(vertices, faces_arr)

    max_norm = np.linalg.norm(vertices, axis=1).max(initial=0.0)
    if max_norm > 0:
        vertices = vertices * (radius / max_norm)
    vertices = vertices + np.asarray(center, dtype=float).reshape(3)

    mesh = Mesh(vertices, faces_arr)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_nhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Compatibility wrapper for make_polyhedron."""

    result = make_polyhedron(
        faces=faces,
        radius=radius,
        center=center,
        backend=backend,
        color=color,
    )
    if backend == "surface":
        return _surface_primitive_result("primitive.make_nhedron", cast("SurfaceBody", result))
    return result


def make_sphere(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_sphere

        return _surface_primitive_result(
            "primitive.make_sphere",
            make_surface_sphere(
                radius=radius,
                center=center,
                theta_resolution=theta_resolution,
                phi_resolution=phi_resolution,
                metadata=_surface_metadata(color=color),
            ),
        )
    from ._legacy_mesh_primitives import sphere_mesh

    mesh = sphere_mesh(radius, theta_resolution, phi_resolution)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Generate a torus (donut) with given major/minor radii."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_torus

        return _surface_primitive_result(
            "primitive.make_torus",
            make_surface_torus(
                major_radius=major_radius,
                minor_radius=minor_radius,
                center=center,
                direction=direction,
                n_theta=n_theta,
                n_phi=n_phi,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import orient_mesh, torus_mesh

    base = torus_mesh(major_radius, minor_radius, n_theta, n_phi)
    aligned = orient_mesh(base, direction)
    aligned.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(aligned, color)
    return aligned


def make_cone(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
    *,
    radius: float | None = None,
) -> Mesh | SurfaceBody:
    """Circular frustum. Set top_diameter=0 for a classic cone."""

    _ensure_backend(backend)
    if radius is not None:
        inferred_bottom = 2.0 * float(radius)
        if bottom_diameter != 1.0 and not np.isclose(bottom_diameter, inferred_bottom):
            raise ValueError("Specify either bottom_diameter or radius, not both.")
        bottom_diameter = inferred_bottom

    bottom_radius = bottom_diameter / 2.0
    top_radius = top_diameter / 2.0
    if bottom_radius <= 0 and top_radius <= 0:
        raise ValueError("At least one of bottom_diameter or top_diameter must be > 0.")
    if backend == "surface":
        from ._surface_primitives import make_surface_cone

        return _surface_primitive_result(
            "primitive.make_cone",
            make_surface_cone(
                bottom_diameter=bottom_diameter,
                top_diameter=top_diameter,
                height=height,
                center=center,
                direction=direction,
                resolution=resolution,
                metadata=_surface_metadata(color=color),
            ),
        )

    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(bottom_radius, top_radius, height, resolution)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_prism(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """
    Rectangular frustum (pyramid/prism). Set top_size=(0,0) for a pyramid, or None to match base.
    """

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_prism

        return _surface_primitive_result(
            "primitive.make_prism",
            make_surface_prism(
                base_size=base_size,
                top_size=top_size,
                height=height,
                center=center,
                direction=direction,
                metadata=_surface_metadata(color=color),
            ),
        )

    if top_size is None:
        top_size = tuple(base_size)
    from ._legacy_mesh_primitives import orient_mesh, rectangular_frustum_mesh

    mesh = rectangular_frustum_mesh(tuple(base_size), tuple(top_size), height)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_box_mesh(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_box`."""

    return cast(Mesh, make_box(size=size, center=center, backend="mesh", color=color))


def make_cylinder_mesh(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    capping: bool = True,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_cylinder`."""

    return cast(
        Mesh,
        make_cylinder(
            radius=radius,
            height=height,
            center=center,
            direction=direction,
            resolution=resolution,
            capping=capping,
            backend="mesh",
            color=color,
        ),
    )


def make_ngon_mesh(
    sides: int = 6,
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_ngon`."""

    return cast(
        Mesh,
        make_ngon(
            sides=sides,
            radius=radius,
            height=height,
            center=center,
            direction=direction,
            backend="mesh",
            color=color,
            side_length=side_length,
        ),
    )


def make_polyhedron_mesh(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_polyhedron`."""

    return cast(Mesh, make_polyhedron(faces=faces, radius=radius, center=center, backend="mesh", color=color))


def make_nhedron_mesh(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_nhedron`."""

    return cast(Mesh, make_nhedron(faces=faces, radius=radius, center=center, backend="mesh", color=color))


def make_sphere_mesh(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_sphere`."""

    return cast(
        Mesh,
        make_sphere(
            radius=radius,
            center=center,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
            backend="mesh",
            color=color,
        ),
    )


def make_torus_mesh(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_torus`."""

    return cast(
        Mesh,
        make_torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            center=center,
            direction=direction,
            n_theta=n_theta,
            n_phi=n_phi,
            backend="mesh",
            color=color,
        ),
    )


def make_cone_mesh(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    color: Sequence[float] | str | None = None,
    *,
    radius: float | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_cone`."""

    return cast(
        Mesh,
        make_cone(
            bottom_diameter=bottom_diameter,
            top_diameter=top_diameter,
            height=height,
            center=center,
            direction=direction,
            resolution=resolution,
            backend="mesh",
            color=color,
            radius=radius,
        ),
    )


def make_prism_mesh(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_prism`."""

    return cast(
        Mesh,
        make_prism(
            base_size=base_size,
            top_size=top_size,
            height=height,
            center=center,
            direction=direction,
            backend="mesh",
            color=color,
        ),
    )


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    oriented = faces.copy()
    centers = vertices[oriented].mean(axis=1)
    v1 = vertices[oriented[:, 1]] - vertices[oriented[:, 0]]
    v2 = vertices[oriented[:, 2]] - vertices[oriented[:, 0]]
    normals = np.cross(v1, v2)
    dots = np.einsum("ij,ij->i", normals, centers)
    flip = dots < 0
    if np.any(flip):
        oriented[flip] = oriented[flip][:, [0, 2, 1]]
    return oriented


def _regular_polyhedron_data(face_count: int) -> tuple[np.ndarray, list[list[int]]]:
    if face_count == 4:
        vertices = np.array(
            [
                (1.0, 1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0),
            ]
        )
        faces = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
        return vertices, faces

    if face_count == 6:
        vertices = np.array(
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ]
        )
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        return vertices, faces

    if face_count == 8:
        vertices = np.array(
            [
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ]
        )
        faces = [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
        return vertices, faces

    if face_count == 20:
        return _icosahedron_data()

    if face_count == 12:
        ico_vertices, ico_faces = _icosahedron_data()
        dodeca_vertices, dodeca_faces = _dodecahedron_from_icosa(ico_vertices, ico_faces)
        return dodeca_vertices, dodeca_faces

    raise ValueError("faces must be one of: 4, 6, 8, 12, 20.")


def _icosahedron_data() -> tuple[np.ndarray, list[list[int]]]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ]
    )
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    return vertices, faces


def _dodecahedron_from_icosa(
    ico_vertices: np.ndarray, ico_faces: list[list[int]]
) -> tuple[np.ndarray, list[list[int]]]:
    ico_vertices = np.asarray(ico_vertices, dtype=float)
    centroids = np.asarray([ico_vertices[face].mean(axis=0) for face in ico_faces], dtype=float)
    norms = np.linalg.norm(centroids, axis=1)
    centroids = centroids / norms[:, None]

    face_map: dict[int, list[int]] = {idx: [] for idx in range(len(ico_vertices))}
    for face_idx, face in enumerate(ico_faces):
        for vert_idx in face:
            face_map[vert_idx].append(face_idx)

    faces: list[list[int]] = []
    for vert_idx, face_indices in face_map.items():
        normal = ico_vertices[vert_idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal = normal / normal_norm
        axis = np.array([0.0, 0.0, 1.0])
        if abs(normal[2]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        basis_u = np.cross(axis, normal)
        basis_norm = np.linalg.norm(basis_u)
        if basis_norm == 0:
            basis_u = np.array([1.0, 0.0, 0.0])
        else:
            basis_u = basis_u / basis_norm
        basis_v = np.cross(normal, basis_u)

        angles: list[tuple[float, int]] = []
        for face_idx in face_indices:
            vec = centroids[face_idx]
            vec = vec - normal * np.dot(vec, normal)
            x = float(np.dot(vec, basis_u))
            y = float(np.dot(vec, basis_v))
            angles.append((np.arctan2(y, x), face_idx))
        ordered = [idx for _, idx in sorted(angles)]
        faces.append(ordered)

    return centroids, faces
