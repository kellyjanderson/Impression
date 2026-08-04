from __future__ import annotations

import numpy as np
import pytest

import impression.modeling.csg as csg_module
from impression.modeling import (
    Loft,
    Loop,
    Region,
    Section,
    Station,
    SurfaceBooleanResult,
    boolean_union,
    make_box,
)


def _rectangle(*, center: tuple[float, float], size: tuple[float, float]) -> Loop:
    cx, cy = center
    width, height = size
    return Loop(
        np.asarray(
            (
                (cx - width / 2.0, cy - height / 2.0),
                (cx + width / 2.0, cy - height / 2.0),
                (cx + width / 2.0, cy + height / 2.0),
                (cx - width / 2.0, cy + height / 2.0),
            ),
            dtype=float,
        )
    )


def _loft_between(section: Section, stations: tuple[Station, Station]):
    return Loft(
        progression=(0.0, 1.0),
        stations=stations,
        topology=(section, section),
        samples=64,
        cap_ends=True,
    )


def _floor(*, z_min: float = 0.0):
    section = Section(regions=(Region(outer=_rectangle(center=(0.0, 0.0), size=(32.0, 32.0))),))
    stations = tuple(
        Station(
            t=float(index),
            section=section,
            origin=(0.0, 0.0, z),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        )
        for index, z in enumerate((z_min, z_min + 1.6))
    )
    return _loft_between(section, stations)


def _wall(
    *,
    normal_axis: str,
    normal_position: float,
    openings: tuple[tuple[float, float, float, float], ...] = (),
):
    section = Section(
        regions=(
            Region(
                outer=_rectangle(center=(0.0, 15.2), size=(32.0, 30.4)),
                holes=tuple(
                    _rectangle(center=(u_center, z_center), size=(width, height))
                    for u_center, z_center, width, height in openings
                ),
            ),
        )
    )
    if normal_axis == "x":
        origins = ((normal_position - 0.8, 0.0, 0.0), (normal_position + 0.8, 0.0, 0.0))
        u = (0.0, 1.0, 0.0)
        n = (1.0, 0.0, 0.0)
    else:
        origins = ((0.0, normal_position - 0.8, 0.0), (0.0, normal_position + 0.8, 0.0))
        u = (-1.0, 0.0, 0.0)
        n = (0.0, 1.0, 0.0)
    stations = tuple(
        Station(
            t=float(index),
            section=section,
            origin=origin,
            u=u,
            v=(0.0, 0.0, 1.0),
            n=n,
        )
        for index, origin in enumerate(origins)
    )
    return _loft_between(section, stations)


def _audio_cube_shell_operands():
    snap_openings = ((-7.0, 27.2, 3.4, 1.8), (7.0, 27.2, 3.4, 1.8))
    acoustic_opening = (0.0, 9.512, 3.0, 3.0)
    usb_opening = (-7.86, 11.29, 5.4, 10.14)
    return (
        _floor(),
        _wall(normal_axis="x", normal_position=-15.2),
        _wall(normal_axis="x", normal_position=15.2, openings=(usb_opening,)),
        _wall(normal_axis="y", normal_position=15.2, openings=(*snap_openings, acoustic_opening)),
        _wall(normal_axis="y", normal_position=-15.2, openings=snap_openings),
    )


@pytest.mark.parametrize("reverse", (False, True), ids=("floor-first", "wall-first"))
def test_minimal_coplanar_loft_union_returns_typed_refusal_without_partial_body(reverse: bool) -> None:
    operands = (_floor(), _wall(normal_axis="x", normal_position=-15.2))
    if reverse:
        operands = tuple(reversed(operands))

    result = boolean_union(operands)

    assert isinstance(result, SurfaceBooleanResult)
    assert result.status == "invalid"
    assert result.body is None
    assert result.classification is None
    assert "Coplanar loft-body union" in str(result.failure_reason)
    assert "invalid-result classification" in str(result.failure_reason)
    assert "no partial result" in str(result.failure_reason)


@pytest.mark.parametrize("reverse", (False, True), ids=("floor-first", "floor-last"))
def test_audio_cube_shell_union_returns_same_typed_classification_in_both_orders(reverse: bool) -> None:
    operands = _audio_cube_shell_operands()
    if reverse:
        operands = tuple(reversed(operands))

    result = boolean_union(operands)

    assert isinstance(result, SurfaceBooleanResult)
    assert result.status == "unsupported"
    assert result.body is None
    assert "Coplanar loft-body union" in str(result.failure_reason)
    assert "coplanar-unsupported classification" in str(result.failure_reason)
    assert "no partial result" in str(result.failure_reason)


def test_public_union_validator_rejects_a_success_result_that_drops_operand_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operands = (_floor(z_min=0.0), _floor(z_min=5.0))

    def incomplete_result(_operation: str, prepared_operands) -> SurfaceBooleanResult:
        return SurfaceBooleanResult(
            operation="union",
            operands=prepared_operands,
            status="succeeded",
            body=prepared_operands.bodies[0],
            classification="closed",
        )

    monkeypatch.setattr(csg_module, "surface_boolean_result", incomplete_result)

    result = boolean_union(operands)

    assert isinstance(result, SurfaceBooleanResult)
    assert result.status == "invalid"
    assert result.body is None
    assert "operand-witness bounds were not retained" in str(result.failure_reason)


def test_existing_supported_box_union_remains_successful() -> None:
    outer = make_box(size=(2.0, 2.0, 2.0))
    inner = make_box(size=(1.0, 1.0, 1.0))

    result = boolean_union((outer, inner))

    assert isinstance(result, SurfaceBooleanResult)
    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.bounds_estimate() == pytest.approx(outer.bounds_estimate())
