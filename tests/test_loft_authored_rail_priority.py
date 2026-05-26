import pytest

from impression.modeling.loft import RailSource, resolve_authored_rails, validate_rail_priority
from impression.modeling.topology import TopologyLandmark, TopologyPath, TopologySegment


def _triangle(*, correspond: tuple[str, str, str] | None = None) -> TopologyPath:
    ids = correspond or ("a", "b", "c")
    return (
        TopologyPath.closed()
        .point("a", (0.0, 0.0), correspond=ids[0])
        .point("b", (1.0, 0.0), correspond=ids[1])
        .point("c", (0.0, 1.0), correspond=ids[2])
        .build()
    )


def test_explicit_ids_beat_names_and_generated_rails() -> None:
    source = _triangle(correspond=("shared", "source-b", "source-c"))
    target = (
        TopologyPath.closed()
        .point("renamed", (0.0, 0.0), correspond="shared")
        .point("b", (1.0, 0.0), correspond="target-b")
        .point("c", (0.0, 1.0), correspond="target-c")
        .build()
    )

    result = resolve_authored_rails(source, target)

    assert ("a", "renamed") in result.matches
    assert result.source_by_match[("a", "renamed")] == RailSource.EXPLICIT_ID


def test_landmark_names_beat_authored_order() -> None:
    source = TopologyPath(
        closed=True,
        points=_triangle().points,
        landmarks=(TopologyLandmark(name="nose", point_ordinal=1),),
    )
    target = TopologyPath(
        closed=True,
        points=_triangle(correspond=("x", "y", "z")).points,
        landmarks=(TopologyLandmark(name="nose", point_ordinal=2),),
    )

    result = resolve_authored_rails(source, target)

    assert ("nose", "nose") in result.matches
    assert result.source_by_match[("nose", "nose")] == RailSource.LANDMARK_NAME


def test_segment_names_resolve_before_authored_order() -> None:
    source = TopologyPath(
        closed=True,
        points=_triangle().points,
        segments=(TopologySegment(name="crown", start_ref="a", end_ref="b"),),
    )
    target = TopologyPath(
        closed=True,
        points=_triangle(correspond=("x", "y", "z")).points,
        segments=(TopologySegment(name="crown", start_ref="x", end_ref="y"),),
    )

    result = resolve_authored_rails(source, target)

    assert ("crown", "crown") in result.matches
    assert result.source_by_match[("crown", "crown")] == RailSource.SEGMENT_NAME


def test_generated_rails_apply_after_authored_rails() -> None:
    source = TopologyPath.named_rect(2.0, 1.0)
    target = TopologyPath.named_rect(3.0, 1.5)

    result = resolve_authored_rails(source, target)

    assert ("bottom-left", "bottom-left") in result.matches
    assert result.source_by_match[("bottom-left", "bottom-left")] == RailSource.GENERATED_RAIL


def test_conflicting_explicit_rails_produce_invalid_input_diagnostics() -> None:
    source = _triangle(correspond=("shared", "shared", "c"))
    target = _triangle(correspond=("shared", "b", "c"))

    result = resolve_authored_rails(source, target)

    assert result.conflicts
    assert result.conflicts[0].priority_tier == RailSource.EXPLICIT_ID
    with pytest.raises(ValueError, match="Invalid authored rail priority"):
        validate_rail_priority(result)
