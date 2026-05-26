from impression.modeling.loft import (
    InferenceCandidateScore,
    RailResolutionResult,
    accept_or_refuse_inferred_correspondence,
    resolve_authored_rails,
    score_correspondence_candidates,
)
from impression.modeling.topology import TopologyPath


def _path(points: tuple[tuple[str, tuple[float, float], str], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates, correspondence_id in points:
        builder.point(name, coordinates, correspond=correspondence_id)
    return builder.build()


def _candidate(
    cost: float,
    *,
    reversed: bool = False,
    protected_anchor_count: int = 2,
    protected_anchor_agreement: float = 1.0,
) -> InferenceCandidateScore:
    return InferenceCandidateScore(
        shift=0,
        reversed=reversed,
        cost=cost,
        cost_terms={"protected_anchor_count": float(protected_anchor_count)},
        protected_anchor_agreement=protected_anchor_agreement,
    )


def test_unique_high_confidence_cyclic_shift_is_accepted() -> None:
    source = _path(
        (
            ("a", (0.0, 0.0), "rail-a"),
            ("b", (1.0, 0.0), "b"),
            ("c", (1.0, 1.0), "rail-c"),
            ("d", (0.0, 1.0), "d"),
        )
    )
    target = _path(
        (
            ("d2", (0.0, 1.0), "d2"),
            ("a2", (0.0, 0.0), "rail-a"),
            ("b2", (1.0, 0.0), "b2"),
            ("c2", (1.0, 1.0), "rail-c"),
        )
    )
    rail_result = resolve_authored_rails(source, target)

    candidates = score_correspondence_candidates(source, target, rail_result)
    result = accept_or_refuse_inferred_correspondence(candidates)

    assert result.accepted is True
    assert result.candidate.shift == 1
    assert result.candidate.reversed is False


def test_equal_cost_symmetric_candidates_are_refused() -> None:
    result = accept_or_refuse_inferred_correspondence((_candidate(0.05), _candidate(0.05)))

    assert result.accepted is False
    assert result.diagnostics[0].reason == "ambiguous_phase"


def test_second_best_separation_below_threshold_is_refused() -> None:
    result = accept_or_refuse_inferred_correspondence((_candidate(0.05), _candidate(0.12)))

    assert result.accepted is False
    assert result.diagnostics[0].reason == "ambiguous_phase"


def test_reversal_requires_explicit_permission() -> None:
    refused = accept_or_refuse_inferred_correspondence((_candidate(0.01, reversed=True),))
    accepted = accept_or_refuse_inferred_correspondence(
        (_candidate(0.01, reversed=True),),
        topology_semantics={"allow_reversal": True},
    )

    assert refused.accepted is False
    assert refused.diagnostics[0].reason == "reversal_conflicts_with_authored_direction"
    assert accepted.accepted is True


def test_protected_anchor_crossing_and_missing_anchor_cases_refuse() -> None:
    crossing = accept_or_refuse_inferred_correspondence(
        (_candidate(0.01, protected_anchor_agreement=0.5),)
    )
    missing = accept_or_refuse_inferred_correspondence(
        (_candidate(0.01, protected_anchor_count=0),)
    )

    assert crossing.diagnostics[0].reason == "crossing_protected_order"
    assert missing.diagnostics[0].reason == "missing_stable_anchors"


def test_scoring_returns_no_candidates_for_incompatible_point_counts() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = TopologyPath.named_rect(1.0, 1.0)

    assert score_correspondence_candidates(source, target, RailResolutionResult(matches=(), source_by_match={})) == ()
