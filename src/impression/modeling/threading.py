from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType


def _load_extension() -> ModuleType:
    package_name = "impression_threading"
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise

    sibling_src = Path(__file__).resolve().parents[4] / "impression-threading" / "src"
    if sibling_src.exists():
        sys.path.insert(0, str(sibling_src))
        return importlib.import_module(package_name)

    raise ModuleNotFoundError(
        "impression.modeling.threading now lives in the sibling "
        "impression-threading project. Install it with "
        "`python -m pip install -e ../impression-threading`."
    )


_extension = _load_extension()
__all__ = list(getattr(_extension, "__all__", ()))
globals().update({name: getattr(_extension, name) for name in __all__})


@dataclass(frozen=True)
class ThreadAssemblyLoweringDiagnostic:
    assembly_type: str
    operation: str
    reason: str
    missing_dependency: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "assembly_type": self.assembly_type,
            "operation": self.operation,
            "reason": self.reason,
            "missing_dependency": self.missing_dependency,
        }


class ThreadAssemblyLoweringError(ValueError):
    def __init__(self, diagnostic: ThreadAssemblyLoweringDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic.reason)


@dataclass(frozen=True)
class ThreadMeshCompatibilityResult:
    mesh: object
    estimate: object
    builder: str
    boundary: str = "explicit-mesh-compatibility"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "builder": self.builder,
            "mesh_vertices": self.mesh.n_vertices,
            "mesh_faces": self.mesh.n_faces,
            "predicted_vertices": self.estimate.predicted_vertices,
            "predicted_faces": self.estimate.predicted_faces,
        }


def thread_feature_csg_dependencies():
    from .csg import SurfaceCSGFeatureDependencyRecord

    return (
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_external_thread",
            module="impression.modeling.threading",
            operation=None,
            surface_builder="make_external_thread",
            explicit_mesh_route="make_thread_mesh_compatibility_result",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_internal_thread",
            module="impression.modeling.threading",
            operation=None,
            surface_builder="make_internal_thread",
            explicit_mesh_route="make_thread_mesh_compatibility_result",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_threaded_rod",
            module="impression.modeling.threading",
            operation=None,
            surface_builder="make_threaded_rod",
            explicit_mesh_route="make_thread_mesh_compatibility_result",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_tapped_hole_cutter",
            module="impression.modeling.threading",
            operation=None,
            surface_builder="make_tapped_hole_cutter",
            explicit_mesh_route="make_thread_mesh_compatibility_result",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_hex_nut",
            module="impression.modeling.threading",
            operation="difference",
            surface_builder="make_hex_nut",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_round_nut",
            module="impression.modeling.threading",
            operation="difference",
            surface_builder="make_round_nut",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.make_runout_relief",
            module="impression.modeling.threading",
            operation=None,
            surface_builder="make_runout_relief",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="threading.lower_thread_surface_assembly",
            module="impression.modeling.threading",
            operation="difference",
            surface_builder="lower_thread_surface_assembly",
            explicit_mesh_route="make_thread_mesh_compatibility_result",
        ),
    )


def _thread_surface_result(caller_id: str, result):
    from .csg import assert_no_hidden_surface_csg_mesh_fallback

    return assert_no_hidden_surface_csg_mesh_fallback(caller_id, result)


def make_external_thread(spec, *args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_external_thread, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_external_thread",
        _extension.make_external_thread(spec, *args, backend=backend, **kwargs),
    )


def make_internal_thread(spec, *args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_internal_thread, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_internal_thread",
        _extension.make_internal_thread(spec, *args, backend=backend, **kwargs),
    )


def make_threaded_rod(spec, *args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_threaded_rod, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_threaded_rod",
        _extension.make_threaded_rod(spec, *args, backend=backend, **kwargs),
    )


def make_tapped_hole_cutter(spec, *args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_tapped_hole_cutter, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_tapped_hole_cutter",
        _extension.make_tapped_hole_cutter(spec, *args, backend=backend, **kwargs),
    )


def make_hex_nut(spec, *args, backend="mesh", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_hex_nut, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_hex_nut",
        _extension.make_hex_nut(spec, *args, backend=backend, **kwargs),
    )


def make_round_nut(spec, *args, backend="mesh", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_round_nut, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_round_nut",
        _extension.make_round_nut(spec, *args, backend=backend, **kwargs),
    )


def make_runout_relief(spec, *args, backend="mesh", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_runout_relief, spec, *args, backend=backend, **kwargs)
    return _thread_surface_result(
        "threading.make_runout_relief",
        _extension.make_runout_relief(spec, *args, backend=backend, **kwargs),
    )


def _call_with_legacy_mesh_primitives(fn, *args, **kwargs):
    from .primitives import make_cylinder_mesh, make_ngon_mesh

    globals_dict = getattr(fn, "__globals__", {})
    original_cylinder = globals_dict.get("make_cylinder")
    original_ngon = globals_dict.get("make_ngon")
    globals_dict["make_cylinder"] = make_cylinder_mesh
    globals_dict["make_ngon"] = make_ngon_mesh
    try:
        return fn(*args, **kwargs)
    finally:
        if original_cylinder is not None:
            globals_dict["make_cylinder"] = original_cylinder
        if original_ngon is not None:
            globals_dict["make_ngon"] = original_ngon


def lower_thread_surface_assembly(assembly):
    if not isinstance(assembly, _extension.ThreadSurfaceAssembly):
        raise TypeError("lower_thread_surface_assembly requires a ThreadSurfaceAssembly.")
    lowered = tuple(lower_thread_operand_to_surface_body(operand) for operand in assembly.operands)
    if assembly.operation == "difference":
        from .csg import surface_csg_feature_gate

        gate = surface_csg_feature_gate(
            "threading.lower_thread_surface_assembly",
            "difference",
            lowered,
        )
        raise ThreadAssemblyLoweringError(
            ThreadAssemblyLoweringDiagnostic(
                assembly_type=assembly.assembly_type,
                operation=assembly.operation,
                reason=(
                    gate.reason
                    if not gate.supported
                    else "Thread surface assembly requires surface boolean difference lowering."
                ),
                missing_dependency=gate.boundary if not gate.supported else "surface-boolean-difference",
            )
        )
    if assembly.operation not in {"standalone", "union"}:
        raise ThreadAssemblyLoweringError(
            ThreadAssemblyLoweringDiagnostic(
                assembly_type=assembly.assembly_type,
                operation=assembly.operation,
                reason=f"Unsupported thread surface assembly operation {assembly.operation!r}.",
            )
        )

    from .surface import make_surface_body

    if assembly.operation == "union" and len(lowered) > 1:
        from .csg import surface_csg_feature_gate

        gate = surface_csg_feature_gate("threading.lower_thread_surface_assembly", "union", lowered)
        if not gate.supported:
            raise ThreadAssemblyLoweringError(
                ThreadAssemblyLoweringDiagnostic(
                    assembly_type=assembly.assembly_type,
                    operation=assembly.operation,
                    reason=gate.reason,
                    missing_dependency=gate.boundary,
                )
            )
    shells = tuple(shell for body in lowered for shell in body.shells)
    return _thread_surface_result(
        "threading.lower_thread_surface_assembly",
        make_surface_body(
            shells,
            metadata={
                "kernel": {
                    "producer": "threading",
                    "assembly_type": assembly.assembly_type,
                    "operation": assembly.operation,
                    "source_identity": assembly.stable_identity,
                }
            },
        ),
    )


def make_thread_mesh_compatibility_result(
    spec,
    *,
    builder="external",
    quality=None,
    color=None,
    overshoot=0.5,
):
    quality = _extension.MeshQuality() if quality is None else quality
    if builder == "external":
        estimate_spec = replace(spec, kind="external")
        mesh = make_external_thread(spec, quality=quality, color=color, backend="mesh")
    elif builder == "internal":
        estimate_spec = replace(spec, kind="internal")
        mesh = make_internal_thread(spec, quality=quality, color=color, backend="mesh")
    elif builder == "threaded_rod":
        estimate_spec = replace(spec, kind="external")
        mesh = make_threaded_rod(spec, quality=quality, color=color, backend="mesh")
    elif builder == "tapped_hole_cutter":
        estimate_spec = replace(
            spec,
            kind="internal",
            length=spec.length + 2.0 * float(overshoot),
            axis_origin=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] - float(overshoot)),
        )
        mesh = make_tapped_hole_cutter(
            spec,
            quality=quality,
            color=color,
            overshoot=overshoot,
            backend="mesh",
        )
    else:
        raise _extension.InvalidThreadSpec(
            "builder must be 'external', 'internal', 'threaded_rod', or 'tapped_hole_cutter'."
        )
    estimate = _extension.estimate_mesh_cost(estimate_spec, quality)
    result = ThreadMeshCompatibilityResult(mesh=mesh, estimate=estimate, builder=builder)
    mesh.metadata.update({"thread_mesh_compatibility": result.canonical_payload()})
    return result


def lower_thread_operand_to_surface_body(operand):
    if not isinstance(operand, _extension.ThreadSurfaceOperand):
        raise TypeError("lower_thread_operand_to_surface_body requires a ThreadSurfaceOperand.")
    if operand.kind == "thread":
        return _lower_thread_payload_to_surface_body(operand.payload, role=operand.role)
    if operand.kind == "primitive":
        return _lower_thread_primitive_payload_to_surface_body(operand.payload, role=operand.role)
    raise ThreadAssemblyLoweringError(
        ThreadAssemblyLoweringDiagnostic(
            assembly_type="operand",
            operation="lower",
            reason=f"Unsupported thread surface operand kind {operand.kind!r}.",
        )
    )


def _lower_thread_payload_to_surface_body(payload: dict[str, object], *, role: str):
    from ._surface_primitives import make_surface_cylinder

    radius = float(payload["major_diameter"]) / 2.0
    height = float(payload["length"])
    axis_origin = tuple(float(value) for value in payload["axis_origin"])
    axis_direction = tuple(float(value) for value in payload["axis_direction"])
    center = (
        axis_origin[0] + axis_direction[0] * height / 2.0,
        axis_origin[1] + axis_direction[1] * height / 2.0,
        axis_origin[2] + axis_direction[2] * height / 2.0,
    )
    return make_surface_cylinder(
        radius=radius,
        height=height,
        center=center,
        direction=axis_direction,
        metadata={"kernel": {"producer": "threading", "role": role, "thread_payload": dict(payload)}},
    )


def _lower_thread_primitive_payload_to_surface_body(payload: dict[str, object], *, role: str):
    from ._surface_primitives import make_surface_cylinder, make_surface_ngon

    family = payload.get("family")
    metadata = {"kernel": {"producer": "threading", "role": role, "primitive_payload": dict(payload)}}
    if family == "cylinder":
        return make_surface_cylinder(
            radius=float(payload["radius"]),
            height=float(payload["height"]),
            center=payload.get("center", (0.0, 0.0, 0.0)),
            direction=payload.get("direction", (0.0, 0.0, 1.0)),
            metadata=metadata,
        )
    if family == "ngon_prism":
        return make_surface_ngon(
            sides=int(payload["sides"]),
            radius=float(payload["radius"]),
            height=float(payload["height"]),
            center=payload.get("center", (0.0, 0.0, 0.0)),
            metadata=metadata,
        )
    raise ThreadAssemblyLoweringError(
        ThreadAssemblyLoweringDiagnostic(
            assembly_type="operand",
            operation="lower",
            reason=f"Unsupported thread primitive operand family {family!r}.",
        )
    )


globals().update(
    {
        "make_external_thread": make_external_thread,
        "make_internal_thread": make_internal_thread,
        "make_threaded_rod": make_threaded_rod,
        "make_tapped_hole_cutter": make_tapped_hole_cutter,
        "make_hex_nut": make_hex_nut,
        "make_round_nut": make_round_nut,
        "make_runout_relief": make_runout_relief,
        "ThreadAssemblyLoweringDiagnostic": ThreadAssemblyLoweringDiagnostic,
        "ThreadAssemblyLoweringError": ThreadAssemblyLoweringError,
        "ThreadMeshCompatibilityResult": ThreadMeshCompatibilityResult,
        "lower_thread_operand_to_surface_body": lower_thread_operand_to_surface_body,
        "lower_thread_surface_assembly": lower_thread_surface_assembly,
        "make_thread_mesh_compatibility_result": make_thread_mesh_compatibility_result,
    }
)

__all__.extend(
    [
        "ThreadAssemblyLoweringDiagnostic",
        "ThreadAssemblyLoweringError",
        "ThreadMeshCompatibilityResult",
        "lower_thread_operand_to_surface_body",
        "lower_thread_surface_assembly",
        "make_thread_mesh_compatibility_result",
    ]
)
