from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
import pytest

from impression.io import ImpressFormatError, make_impress_document_payload
from impression.mesh import Mesh
from impression.modeling import (
    HeightmapMeshCompatibilityResult,
    HingeSurfaceAssembly,
    SurfaceBody,
    SurfaceBooleanResult,
    TextMeshCompatibilityResult,
    ThreadMeshCompatibilityResult,
    ThreadSurfaceRepresentation,
    as_section,
    boolean_union,
    heightmap,
    heightmap_mesh_compatibility_result,
    loft_execute_plan,
    loft_plan_sections,
    lookup_standard_thread,
    make_box,
    make_external_thread,
    make_line,
    make_sphere,
    make_text,
    make_text_mesh_result,
    make_thread_mesh_compatibility_result,
    make_traditional_hinge_pair,
)
from impression.modeling.drawing2d import make_rect
from impression.modeling.loft import Station


@dataclass(frozen=True)
class UnsupportedOperationDiagnostic:
    """Test-side assertion record for explicit unsupported surface outcomes."""

    subsystem: str
    boundary: str
    reason: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "reason": self.reason,
            "subsystem": self.subsystem,
        }


@dataclass(frozen=True)
class NoFallbackFixture:
    """One bounded fixture in the no-hidden-mesh-fallback matrix."""

    subsystem: str
    call: Callable[[], object]
    expected_type: type
    explicit_mesh_boundary: str | None = None


def _simple_loft_body() -> SurfaceBody:
    start = make_rect(size=(1.0, 1.0))
    end = make_rect(size=(0.8, 1.2))
    stations = [
        Station(
            t=0.0,
            section=as_section(start),
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
        Station(
            t=1.0,
            section=as_section(end),
            origin=[0.0, 0.0, 1.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
    ]
    return loft_execute_plan(loft_plan_sections(stations, samples=12), cap_ends=False)


def _assert_no_hidden_mesh_fallback(fixture: NoFallbackFixture) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = fixture.call()
    assert isinstance(result, fixture.expected_type), fixture.subsystem
    if fixture.explicit_mesh_boundary is None:
        assert not isinstance(result, Mesh), fixture.subsystem
    else:
        payload = result.canonical_payload()
        assert payload["boundary"] == fixture.explicit_mesh_boundary
    return result


def test_authored_surface_api_matrix_has_no_hidden_mesh_fallbacks() -> None:
    image = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    spec = lookup_standard_thread("metric", "M6x1", length=6.0)
    fixtures = (
        NoFallbackFixture("primitive", lambda: make_box(size=(1.0, 1.0, 1.0)), SurfaceBody),
        NoFallbackFixture("loft", _simple_loft_body, SurfaceBody),
        NoFallbackFixture("text", lambda: make_text("", backend="surface"), SurfaceBody),
        NoFallbackFixture("drafting", lambda: make_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), SurfaceBody),
        NoFallbackFixture("heightmap-surface", lambda: heightmap(image, backend="surface"), SurfaceBody),
        NoFallbackFixture("threading", lambda: make_external_thread(spec, backend="surface"), ThreadSurfaceRepresentation),
        NoFallbackFixture("hinge", lambda: make_traditional_hinge_pair(width=24.0, knuckle_count=5), HingeSurfaceAssembly),
    )

    for fixture in fixtures:
        _assert_no_hidden_mesh_fallback(fixture)


def test_explicit_mesh_compatibility_matrix_names_mesh_boundaries() -> None:
    image = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    spec = lookup_standard_thread("metric", "M6x1", length=6.0)
    fixtures = (
        NoFallbackFixture(
            "text-mesh-compatibility",
            lambda: make_text_mesh_result(""),
            TextMeshCompatibilityResult,
            "explicit-mesh-compatibility",
        ),
        NoFallbackFixture(
            "heightmap-mesh-compatibility",
            lambda: heightmap_mesh_compatibility_result(image),
            HeightmapMeshCompatibilityResult,
            "explicit-mesh-compatibility",
        ),
        NoFallbackFixture(
            "thread-mesh-compatibility",
            lambda: make_thread_mesh_compatibility_result(spec, builder="external"),
            ThreadMeshCompatibilityResult,
            "explicit-mesh-compatibility",
        ),
    )

    for fixture in fixtures:
        _assert_no_hidden_mesh_fallback(fixture)


def test_surface_csg_unsupported_result_is_diagnostic_not_mesh_fallback() -> None:
    result = boolean_union(
        [make_box(size=(1.0, 1.0, 1.0)), make_sphere(radius=0.5, center=(2.0, 0.0, 0.0))],
        backend="surface",
    )

    diagnostic = UnsupportedOperationDiagnostic(
        subsystem="csg",
        boundary="surface-boolean",
        reason=result.failure_reason or "",
    )
    assert isinstance(result, SurfaceBooleanResult)
    assert result.status == "unsupported"
    assert result.body is None
    assert "operand-family-eligibility" in diagnostic.reason
    assert diagnostic.canonical_payload()["boundary"] == "surface-boolean"


def test_impress_serializer_rejects_mesh_inputs_instead_of_wrapping_surface_truth() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    with pytest.raises(ImpressFormatError, match="SurfaceBody"):
        make_impress_document_payload([mesh])
