from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from impression.modeling import Loft, LoftPlannerOptions, Loop, Region, Section, Station, SurfaceBody, TopologyPath
from impression.modeling.loft import LoftPlanningBlockedError, loft_plan_sections


def _region() -> Region:
    return Region(Loop(np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))))


def _stations(
    source_ids: tuple[tuple[str, ...], ...],
    target_ids: tuple[tuple[str, ...], ...],
    *,
    target_order: tuple[int, ...] | None = None,
) -> tuple[Station, Station, Section, Section]:
    count = len(source_ids)
    order = tuple(range(count)) if target_order is None else target_order
    source_section = Section(tuple(_region() for _ in range(count)))
    target_section = Section(tuple(_region() for _ in order))
    source = Station(
        t=0.0,
        section=source_section,
        origin=(0.0, 0.0, 0.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
        successor_ids=source_ids,
    )
    target = Station(
        t=1.0,
        section=target_section,
        origin=(0.0, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
        predecessor_ids=tuple(target_ids[index] for index in order),
    )
    return source, target, source_section, target_section


@pytest.mark.parametrize("count", [1, 64, 65, 80])
def test_identity_matched_regions_bypass_branch_enumeration(count: int) -> None:
    identities = tuple((f"guide-{index}",) for index in range(count))
    source, target, _source_section, _target_section = _stations(identities, identities)

    plan = loft_plan_sections((source, target), samples=3, ambiguity_max_branches=1)

    assert len(plan.transitions[0].region_pairs) == count
    assert plan.metadata["identity_resolved_pair_count"] == count
    assert plan.metadata["identity_residual_region_count"] == 0
    assert plan.metadata["ambiguity_resolved_intervals_count"] == 0


def test_public_loft_routes_65_identified_regions_before_geometry_search() -> None:
    count = 65
    identities = tuple((f"guide-{index}",) for index in range(count))
    source, target, source_section, target_section = _stations(identities, identities)

    body = Loft(
        (0.0, 1.0),
        (source, target),
        (source_section, target_section),
        samples=3,
        ambiguity_max_branches=1,
    )

    assert isinstance(body, SurfaceBody)
    assert body.kernel_metadata()["identity_resolved_pair_count"] == count


def test_identity_pairing_is_source_ordered_when_target_is_shuffled() -> None:
    identities = tuple((f"guide-{index}",) for index in range(8))
    order = (5, 1, 7, 0, 3, 6, 2, 4)
    source, target, _source_section, _target_section = _stations(
        identities,
        identities,
        target_order=order,
    )

    plan = loft_plan_sections((source, target), samples=3, ambiguity_max_branches=1)

    inverse = {source_index: target_index for target_index, source_index in enumerate(order)}
    assert tuple(pair.curr_region_ref.index for pair in plan.transitions[0].region_pairs) == tuple(
        inverse[index] for index in range(8)
    )


def test_duplicate_and_contradictory_region_ids_are_named_invalid_inputs() -> None:
    duplicate = (("same",), ("same",))
    target_ids = (("first",), ("second",))
    source, target, _source_section, _target_section = _stations(duplicate, target_ids)
    with pytest.raises(ValueError, match="invalid_region_identity duplicate source id 'same'"):
        loft_plan_sections((source, target), samples=3)

    source, target, _source_section, _target_section = _stations(
        (("left",), ("right",)),
        (("left",), ("other",)),
    )
    with pytest.raises(ValueError, match="invalid_region_identity contradictory exact id sets"):
        loft_plan_sections((source, target), samples=3)


def test_mixed_identity_fixture_enumerates_only_anonymous_residue() -> None:
    source_ids = (("a",), ("b",), ("c",), (), ())
    target_ids = (("a",), ("b",), ("c",), (), ())
    source, target, _source_section, _target_section = _stations(source_ids, target_ids)

    plan = loft_plan_sections((source, target), samples=3, ambiguity_max_branches=2)

    assert plan.metadata["identity_resolved_pair_count"] == 3
    assert plan.metadata["identity_residual_region_count"] == 2


def test_anonymous_ambiguity_still_obeys_branch_limit() -> None:
    source, target, _source_section, _target_section = _stations(((), (), ()), ((), (), ()))

    with pytest.raises(LoftPlanningBlockedError, match="candidate_enumeration_limit"):
        loft_plan_sections((source, target), samples=3, ambiguity_max_branches=2)


def test_loft_planner_options_are_immutable_and_have_a_canonical_diagnostic_payload() -> None:
    options = LoftPlannerOptions(
        split_merge_mode="resolve",
        split_merge_steps=6,
        split_merge_bias=0.4,
        ambiguity_mode="auto",
        ambiguity_cost_profile="balanced",
        ambiguity_max_branches=17,
        fairness_mode="global",
        fairness_weight=0.35,
        skeleton_mode="auto",
        fairness_iterations=9,
    )

    with pytest.raises(FrozenInstanceError):
        options.ambiguity_max_branches = 64  # type: ignore[misc]

    assert options.canonical_payload()["ambiguity_max_branches"] == 17
    assert options.canonical_payload()["fairness_mode"] == "global"


def _named_rectangle(identity: str, center_x: float) -> TopologyPath:
    return TopologyPath.from_points(
        (
            (center_x - 0.5, -0.5),
            (center_x + 0.5, -0.5),
            (center_x + 0.5, 0.5),
            (center_x - 0.5, 0.5),
        ),
        id=identity,
    )


def _section_with_named_holes(paths: tuple[TopologyPath, ...]) -> Section:
    outer = TopologyPath.from_points(((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0)), id="outer")
    return Section(
        regions=(
            Region(
                outer=outer.to_section_loop(),
                holes=tuple(path.to_section_loop() for path in paths),
            ),
        ),
        metadata={"topology_paths": (outer, *paths)},
    )


def _hole_stations(source: Section, target: Section) -> tuple[Station, Station]:
    frame = {
        "u": (1.0, 0.0, 0.0),
        "v": (0.0, 1.0, 0.0),
        "n": (0.0, 0.0, 1.0),
    }
    return (
        Station(t=0.0, section=source, origin=(0.0, 0.0, 0.0), **frame),
        Station(t=1.0, section=target, origin=(0.0, 0.0, 2.0), **frame),
    )


def test_named_holes_that_exchange_positions_pair_by_identity() -> None:
    source = _section_with_named_holes((_named_rectangle("hole-a", -2.0), _named_rectangle("hole-b", 2.0)))
    target = _section_with_named_holes((_named_rectangle("hole-a", 2.0), _named_rectangle("hole-b", -2.0)))

    plan = loft_plan_sections(_hole_stations(source, target), samples=16, fairness_mode="off")
    hole_pairs = plan.transitions[0].region_pairs[0].loop_pairs[1:]

    assert tuple(pair.prev_loop_ref.identity for pair in hole_pairs) == ("hole-a", "hole-b")
    assert tuple(pair.curr_loop_ref.identity for pair in hole_pairs) == ("hole-a", "hole-b")
    assert tuple(float(pair.curr_loop[:, 0].mean()) for pair in hole_pairs) == pytest.approx((2.0, -2.0))
    assert plan.metadata["hole_identity_resolved_pair_count"] == 2

    body = Loft(
        (0.0, 1.0),
        _hole_stations(source, target),
        (source, target),
        samples=16,
        fairness_mode="off",
        cap_ends=True,
    )
    assert body.kernel_metadata()["hole_identity_resolved_pair_count"] == 2


def test_duplicate_and_missing_named_hole_identities_fail_before_geometry() -> None:
    source = _section_with_named_holes((_named_rectangle("hole-a", -2.0), _named_rectangle("hole-a", 2.0)))
    target = _section_with_named_holes((_named_rectangle("hole-a", -2.0), _named_rectangle("hole-b", 2.0)))
    with pytest.raises(ValueError, match="invalid_hole_identity duplicate source id 'hole-a'"):
        loft_plan_sections(_hole_stations(source, target), samples=16)

    source = _section_with_named_holes((_named_rectangle("hole-a", -2.0), _named_rectangle("hole-b", 2.0)))
    target = _section_with_named_holes((_named_rectangle("hole-a", -2.0), _named_rectangle("hole-c", 2.0)))
    with pytest.raises(ValueError, match="invalid_hole_identity contradictory exact id sets"):
        loft_plan_sections(_hole_stations(source, target), samples=16)


def test_named_pairing_leaves_anonymous_residue_to_geometric_fallback() -> None:
    source_named = _named_rectangle("named", -2.0)
    target_named = _named_rectangle("named", 2.0)
    anonymous_source = _named_rectangle("unused-source", 0.0).to_section_loop()
    anonymous_target = _named_rectangle("unused-target", 0.0).to_section_loop()
    outer = TopologyPath.from_points(((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0)), id="outer")
    source = Section(
        regions=(Region(outer=outer.to_section_loop(), holes=(source_named.to_section_loop(), anonymous_source)),),
        metadata={"topology_paths": (outer, source_named)},
    )
    target = Section(
        regions=(Region(outer=outer.to_section_loop(), holes=(target_named.to_section_loop(), anonymous_target)),),
        metadata={"topology_paths": (outer, target_named)},
    )

    plan = loft_plan_sections(_hole_stations(source, target), samples=16, fairness_mode="off")
    hole_pairs = plan.transitions[0].region_pairs[0].loop_pairs[1:]

    assert tuple(pair.prev_loop_ref.identity for pair in hole_pairs) == ("named", None)
    assert tuple(float(pair.curr_loop[:, 0].mean()) for pair in hole_pairs) == pytest.approx((2.0, 0.0))
