from __future__ import annotations

import pytest
import numpy as np

from impression.modeling import MeshQuality, Section, Station, as_section
from impression.modeling.drawing2d import make_rect
from impression.modeling.loft import (
    LoftAmbiguityCandidate,
    LoftAmbiguityIntervalReport,
    LoftAmbiguityReport,
    LoftPlan,
    PlannedLoopPair,
    PlannedLoopRef,
    PlannedRegionPair,
    PlannedRegionRef,
    PlannedStation,
    PlannedTransition,
    _ambiguity_failure_candidate_count,
    _ambiguity_failure_stage,
    _derive_disambiguation_seed,
    _enumerate_hole_ambiguity_candidates,
    _enumerate_region_ambiguity_candidates,
    _predicted_actions_for_assignment,
    _resolve_interactive_region_assignment,
    _resolve_probabilistic_region_assignment,
    _validate_loft_plan,
    loft_plan_ambiguities,
)


def _candidate(
    candidate_id: str,
    assignment: tuple[int, ...],
    *,
    score: tuple[float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 0.0),
    local_fairness: float = 0.0,
) -> LoftAmbiguityCandidate:
    return LoftAmbiguityCandidate(
        interval=(0, 1),
        candidate_id=candidate_id,
        topology_case="many_to_many_expand",
        assignment=assignment,
        predicted_actions=("split_match", "split_birth"),
        score=score,
        local_fairness=local_fairness,
    )


def _square(x0: float, y0: float, size: float = 1.0) -> np.ndarray:
    return np.asarray(
        [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
        ],
        dtype=float,
    )


def _base_loft_plan() -> LoftPlan:
    sampled_loop = np.asarray(
        [(float(index), float(index % 2)) for index in range(8)],
        dtype=float,
    )
    station = PlannedStation(
        station_index=0,
        t=0.0,
        origin=np.array([0.0, 0.0, 0.0], dtype=float),
        u=np.array([1.0, 0.0, 0.0], dtype=float),
        v=np.array([0.0, 1.0, 0.0], dtype=float),
        n=np.array([0.0, 0.0, 1.0], dtype=float),
        regions=((sampled_loop,),),
    )
    station_b = PlannedStation(
        station_index=1,
        t=1.0,
        origin=np.array([0.0, 0.0, 1.0], dtype=float),
        u=np.array([1.0, 0.0, 0.0], dtype=float),
        v=np.array([0.0, 1.0, 0.0], dtype=float),
        n=np.array([0.0, 0.0, 1.0], dtype=float),
        regions=((sampled_loop,),),
    )
    transition = PlannedTransition(
        interval=(0, 1),
        region_pairs=(
            PlannedRegionPair(
                prev_region_ref=PlannedRegionRef(kind="actual", index=0),
                curr_region_ref=PlannedRegionRef(kind="actual", index=0),
                loop_pairs=(
                    PlannedLoopPair(
                        prev_loop_ref=PlannedLoopRef(kind="actual", index=0),
                        curr_loop_ref=PlannedLoopRef(kind="actual", index=0),
                        prev_loop=sampled_loop,
                        curr_loop=sampled_loop,
                        role="stable",
                    ),
                ),
                closures=(),
                action="stable",
                branch_id="b0",
            ),
        ),
        branch_order=("b0",),
        topology_case="one_to_one",
        prev_region_count=1,
        curr_region_count=1,
    )
    metadata = {
        "plan_schema_version": 1,
        "planner": "test",
        "split_merge_mode": "resolve",
        "split_merge_steps": 4,
        "split_merge_bias": 0.5,
        "ambiguity_mode": "auto",
        "ambiguity_selection_policy": "required",
        "ambiguity_cost_profile": "balanced",
        "ambiguity_max_branches": 8,
        "disambiguation_mode": "deterministic",
        "disambiguation_seed": 1,
        "probabilistic_trials": 8,
        "probabilistic_temperature": 0.25,
        "probabilistic_min_confidence": 0.5,
        "probabilistic_fallback": "deterministic",
        "ambiguity_resolved_intervals_count": 0,
        "ambiguity_failed_intervals_count": 0,
        "ambiguity_class_counts": {
            "permutation": 0,
            "containment": 0,
            "symmetry": 0,
            "closure": 0,
        },
        "probabilistic_selected_confidence": 1.0,
        "probabilistic_candidate_count": 0,
        "probabilistic_selected_candidate_ids": {},
        "region_topology_case_counts": {},
        "region_action_counts": {},
        "fairness_mode": "local",
        "fairness_weight": 0.2,
        "fairness_iterations": 4,
        "skeleton_mode": "auto",
        "fairness_objective_pre": {
            "curvature_continuity": 0.0,
            "branch_crossing": 0.0,
            "branch_acceleration": 0.0,
            "synthetic_harshness": 0.0,
            "closure_stress": 0.0,
        },
        "fairness_objective_post": {
            "curvature_continuity": 0.0,
            "branch_crossing": 0.0,
            "branch_acceleration": 0.0,
            "synthetic_harshness": 0.0,
            "closure_stress": 0.0,
        },
        "fairness_diagnostics": {
            "branch_crossing_count": 0.0,
            "continuity_score": 0.0,
            "closure_distortion_score": 0.0,
        },
        "fairness_optimization_convergence_status": "not_run",
    }
    return LoftPlan(samples=8, stations=(station, station_b), transitions=(transition,), metadata=metadata)


def test_loft_ambiguity_helpers_parse_failure_messages() -> None:
    explicit = (
        "unsupported_topology_ambiguity (tie_break_stage=residual_tie_break; "
        "candidate_count_after_pruning=3; detail=demo)"
    )
    assert _ambiguity_failure_stage(explicit) == "residual_tie_break"
    assert _ambiguity_failure_candidate_count(explicit) == "3"
    assert _ambiguity_failure_stage("candidate count exceeds ambiguity_max_branches") == "candidate_enumeration_limit"
    assert _ambiguity_failure_stage("residual indeterminate ambiguity") == "residual_tie_break"
    assert _ambiguity_failure_stage("ambiguity detected in region pair") == "ambiguity_gate"
    assert _ambiguity_failure_stage("something else") == "unknown"
    assert _ambiguity_failure_candidate_count("no explicit candidate count") == "unknown"


def test_loft_disambiguation_seed_is_deterministic_and_input_sensitive() -> None:
    shape = make_rect(size=(1.0, 1.0))
    station_a = Station(
        t=0.0,
        section=as_section(shape),
        origin=[0.0, 0.0, 0.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        n=[0.0, 0.0, 1.0],
    )
    station_b = Station(
        t=1.0,
        section=as_section(shape),
        origin=[0.0, 0.0, 1.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        n=[0.0, 0.0, 1.0],
    )
    station_c = Station(
        t=1.0,
        section=as_section(shape),
        origin=[0.2, 0.0, 1.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        n=[0.0, 0.0, 1.0],
    )

    seed_a = _derive_disambiguation_seed([station_a, station_b], 32)
    seed_b = _derive_disambiguation_seed([station_a, station_b], 32)
    seed_c = _derive_disambiguation_seed([station_a, station_c], 32)

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_loft_interactive_assignment_helper_supports_required_and_best_effort() -> None:
    candidates = (
        _candidate("c0", (0, 1)),
        _candidate("c1", (1, 0)),
    )

    assert _resolve_interactive_region_assignment(
        interval=(0, 1),
        candidates=candidates,
        ambiguity_selection={(0, 1): "c1"},
        ambiguity_selection_policy="required",
    ) == (1, 0)

    assert _resolve_interactive_region_assignment(
        interval=(0, 1),
        candidates=candidates,
        ambiguity_selection=None,
        ambiguity_selection_policy="best_effort",
    ) is None

    with pytest.raises(ValueError, match="missing selection"):
        _resolve_interactive_region_assignment(
            interval=(0, 1),
            candidates=candidates,
            ambiguity_selection=None,
            ambiguity_selection_policy="required",
        )

    with pytest.raises(ValueError, match="unknown candidate_id"):
        _resolve_interactive_region_assignment(
            interval=(0, 1),
            candidates=candidates,
            ambiguity_selection={(0, 1): "missing"},
            ambiguity_selection_policy="required",
        )


def test_loft_probabilistic_assignment_helper_covers_empty_single_fallback_and_failure() -> None:
    with pytest.raises(ValueError, match="no candidates available"):
        _resolve_probabilistic_region_assignment(
            interval=(0, 1),
            candidates=(),
            seed=17,
            probabilistic_trials=8,
            probabilistic_temperature=0.25,
            probabilistic_min_confidence=0.5,
            probabilistic_fallback="deterministic",
        )

    single = (_candidate("only", (0, 1)),)
    assignment, selected, confidence = _resolve_probabilistic_region_assignment(
        interval=(0, 1),
        candidates=single,
        seed=17,
        probabilistic_trials=8,
        probabilistic_temperature=0.25,
        probabilistic_min_confidence=0.5,
        probabilistic_fallback="deterministic",
    )
    assert assignment == (0, 1)
    assert selected.candidate_id == "only"
    assert confidence == 1.0

    candidates = (
        _candidate("best", (0, 1), score=(0.1, 0.0, 0.0, 0.0, 0.0)),
        _candidate("alt", (1, 0), score=(0.2, 0.0, 0.0, 0.0, 0.0)),
    )
    assignment, selected, confidence = _resolve_probabilistic_region_assignment(
        interval=(0, 1),
        candidates=candidates,
        seed=17,
        probabilistic_trials=4,
        probabilistic_temperature=10.0,
        probabilistic_min_confidence=1.1,
        probabilistic_fallback="deterministic",
    )
    assert assignment == (0, 1)
    assert selected.candidate_id == "best"
    assert confidence < 1.1

    with pytest.raises(ValueError, match="selected_confidence="):
        _resolve_probabilistic_region_assignment(
            interval=(0, 1),
            candidates=candidates,
            seed=17,
            probabilistic_trials=4,
            probabilistic_temperature=10.0,
            probabilistic_min_confidence=1.1,
            probabilistic_fallback="error",
        )


def test_loft_ambiguity_report_records_are_constructible() -> None:
    candidate = _candidate("demo", (0, 1))
    interval = LoftAmbiguityIntervalReport(
        interval=(0, 1),
        topology_case="many_to_many_expand",
        ambiguity_class="permutation",
        candidates=(candidate,),
        relationship_group="many_to_many_regions",
    )
    report = LoftAmbiguityReport(schema_version=1, ambiguity_mode="interactive", intervals=(interval,))

    assert report.schema_version == 1
    assert report.intervals[0].candidates[0].assignment == (0, 1)


def test_loft_candidate_enumerators_and_prediction_helpers_cover_remaining_paths() -> None:
    assert _predicted_actions_for_assignment(topology_case="one_to_one", source_count=2, target_count=2) == (
        "stable",
        "stable",
    )
    assert _predicted_actions_for_assignment(topology_case="many_to_many_expand", source_count=2, target_count=3) == (
        "split_match",
        "split_match",
        "split_birth",
    )
    assert _predicted_actions_for_assignment(topology_case="many_to_many_collapse", source_count=3, target_count=2) == (
        "merge_match",
        "merge_match",
        "merge_death",
    )
    assert _predicted_actions_for_assignment(topology_case="unknown", source_count=2, target_count=2) == ()

    assert (
        _enumerate_region_ambiguity_candidates(
            prev_regions=[[ _square(0.0, 0.0) ]],
            curr_regions=[[ _square(0.0, 0.0) ]],
            interval=(0, 1),
            topology_case="one_to_one",
            ambiguity_class="none",
            ambiguity_cost_profile="balanced",
            ambiguity_max_branches=8,
        )
        == ()
    )
    assert (
        _enumerate_region_ambiguity_candidates(
            prev_regions=[],
            curr_regions=[[ _square(0.0, 0.0) ]],
            interval=(0, 1),
            topology_case="many_to_many_expand",
            ambiguity_class="permutation",
            ambiguity_cost_profile="balanced",
            ambiguity_max_branches=8,
        )
        == ()
    )

    region_candidates = _enumerate_region_ambiguity_candidates(
        prev_regions=[[_square(0.0, 0.0)], [_square(2.0, 0.0)]],
        curr_regions=[[_square(0.0, 0.0)], [_square(2.0, 0.0)], [_square(4.0, 0.0)]],
        interval=(0, 1),
        topology_case="many_to_many_expand",
        ambiguity_class="permutation",
        ambiguity_cost_profile="balanced",
        ambiguity_max_branches=8,
    )
    assert region_candidates
    assert all(candidate.candidate_id.startswith("many_to_many_expand:") for candidate in region_candidates)
    assert all(candidate.predicted_actions[-1] == "split_birth" for candidate in region_candidates)
    assert (
        _enumerate_region_ambiguity_candidates(
            prev_regions=[[_square(0.0, 0.0)], [_square(2.0, 0.0)]],
            curr_regions=[[_square(0.0, 0.0)], [_square(2.0, 0.0)], [_square(4.0, 0.0)]],
            interval=(0, 1),
            topology_case="many_to_many_expand",
            ambiguity_class="permutation",
            ambiguity_cost_profile="balanced",
            ambiguity_max_branches=0,
        )
        == ()
    )

    assert _enumerate_hole_ambiguity_candidates(
        prev_holes=[_square(0.0, 0.0)],
        curr_holes=[_square(0.0, 0.0), _square(2.0, 0.0)],
        interval=(0, 1),
        region_index=0,
        ambiguity_cost_profile="balanced",
        ambiguity_max_branches=8,
    ) == ()

    hole_candidates = _enumerate_hole_ambiguity_candidates(
        prev_holes=[_square(0.0, 0.0), _square(2.0, 0.0)],
        curr_holes=[_square(0.0, 0.0), _square(2.0, 0.0), _square(4.0, 0.0)],
        interval=(0, 1),
        region_index=2,
        ambiguity_cost_profile="balanced",
        ambiguity_max_branches=8,
    )
    assert hole_candidates
    assert all(candidate.relationship_group == "hole_many_to_many:r2" for candidate in hole_candidates)
    assert all(candidate.candidate_id.startswith("hole:r2:many_to_many_expand:") for candidate in hole_candidates)
    assert (
        _enumerate_hole_ambiguity_candidates(
            prev_holes=[_square(0.0, 0.0), _square(2.0, 0.0)],
            curr_holes=[_square(0.0, 0.0), _square(2.0, 0.0), _square(4.0, 0.0)],
            interval=(0, 1),
            region_index=2,
            ambiguity_cost_profile="balanced",
            ambiguity_max_branches=0,
        )
        == ()
    )


def test_loft_plan_ambiguities_covers_quality_and_missing_section_validation() -> None:
    shape = make_rect(size=(1.0, 1.0))
    stations = [
        Station(t=0.0, section=as_section(shape), origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=as_section(shape), origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]
    report = loft_plan_ambiguities(stations, samples=16, quality=MeshQuality(lod="preview"))
    assert isinstance(report, LoftAmbiguityReport)

    with pytest.raises(ValueError, match="samples must be >= 3"):
        loft_plan_ambiguities(stations, samples=2)

    broken = [
        Station(t=0.0, section=None, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        stations[1],
    ]
    with pytest.raises(ValueError, match="missing section data"):
        loft_plan_ambiguities(broken, samples=16)


def test_loft_plan_validation_and_transition_properties_cover_guardrails() -> None:
    plan = _base_loft_plan()
    _validate_loft_plan(plan)
    plan.require_executable()
    assert plan.transitions[0].many_to_many_candidate_set is None
    assert plan.transitions[0].many_to_many_decomposition_order == ()
    assert plan.transitions[0].many_to_many_decomposability is None

    blocked = LoftPlan(
        samples=plan.samples,
        stations=plan.stations,
        transitions=plan.transitions,
        metadata={**plan.metadata, "ambiguity_failed_intervals_count": 1},
    )
    with pytest.raises(ValueError, match="not executable"):
        blocked.require_executable()

    missing_schema = LoftPlan(
        samples=plan.samples,
        stations=plan.stations,
        transitions=plan.transitions,
        metadata={key: value for key, value in plan.metadata.items() if key != "plan_schema_version"},
    )
    with pytest.raises(ValueError, match="missing plan_schema_version"):
        _validate_loft_plan(missing_schema)

    bad_candidate_count = LoftPlan(
        samples=plan.samples,
        stations=plan.stations,
        transitions=plan.transitions,
        metadata={**plan.metadata, "probabilistic_candidate_count": -1},
    )
    with pytest.raises(ValueError, match="probabilistic_candidate_count must be >= 0"):
        _validate_loft_plan(bad_candidate_count)

    bad_class_counts = LoftPlan(
        samples=plan.samples,
        stations=plan.stations,
        transitions=plan.transitions,
        metadata={**plan.metadata, "ambiguity_class_counts": {"permutation": 0}},
    )
    with pytest.raises(ValueError, match="missing key 'containment'"):
        _validate_loft_plan(bad_class_counts)

    invalid_pair = PlannedRegionPair(
        prev_region_ref=PlannedRegionRef(kind="actual", index=0),
        curr_region_ref=PlannedRegionRef(kind="actual", index=0),
        loop_pairs=plan.transitions[0].region_pairs[0].loop_pairs,
        closures=(),
        action="mystery",
        branch_id="b0",
    )
    with pytest.raises(ValueError, match="Unsupported planned region-pair action"):
        _ = invalid_pair.operator_family


def test_station_requires_topology_for_directional_correspondence() -> None:
    with pytest.raises(ValueError, match="Directional correspondence requires section topology"):
        Station(
            t=0.0,
            section=None,
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
            predecessor_ids=(frozenset({"a"}),),
        )
