from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
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


def make_external_thread(spec, *args, backend="surface", **kwargs):
    return _extension.make_external_thread(spec, *args, backend=backend, **kwargs)


def make_internal_thread(spec, *args, backend="surface", **kwargs):
    return _extension.make_internal_thread(spec, *args, backend=backend, **kwargs)


def make_threaded_rod(spec, *args, backend="surface", **kwargs):
    return _extension.make_threaded_rod(spec, *args, backend=backend, **kwargs)


def make_tapped_hole_cutter(spec, *args, backend="surface", **kwargs):
    return _extension.make_tapped_hole_cutter(spec, *args, backend=backend, **kwargs)


def lower_thread_surface_assembly(assembly):
    if not isinstance(assembly, _extension.ThreadSurfaceAssembly):
        raise TypeError("lower_thread_surface_assembly requires a ThreadSurfaceAssembly.")
    if assembly.operation == "difference":
        raise ThreadAssemblyLoweringError(
            ThreadAssemblyLoweringDiagnostic(
                assembly_type=assembly.assembly_type,
                operation=assembly.operation,
                reason="Thread surface assembly requires surface boolean difference lowering.",
                missing_dependency="surface-boolean-difference",
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

    lowered = tuple(lower_thread_operand_to_surface_body(operand) for operand in assembly.operands)
    shells = tuple(shell for body in lowered for shell in body.shells)
    return make_surface_body(
        shells,
        metadata={
            "kernel": {
                "producer": "threading",
                "assembly_type": assembly.assembly_type,
                "operation": assembly.operation,
                "source_identity": assembly.stable_identity,
            }
        },
    )


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
        "ThreadAssemblyLoweringDiagnostic": ThreadAssemblyLoweringDiagnostic,
        "ThreadAssemblyLoweringError": ThreadAssemblyLoweringError,
        "lower_thread_operand_to_surface_body": lower_thread_operand_to_surface_body,
        "lower_thread_surface_assembly": lower_thread_surface_assembly,
    }
)

__all__.extend(
    [
        "ThreadAssemblyLoweringDiagnostic",
        "ThreadAssemblyLoweringError",
        "lower_thread_operand_to_surface_body",
        "lower_thread_surface_assembly",
    ]
)
