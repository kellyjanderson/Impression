import pytest

from impression.modeling.topology import TopologyPath


def test_topology_path_closed_builder_preserves_authored_order() -> None:
    path = (
        TopologyPath.closed(anchor="bottom-left")
        .point("bottom-left", (0.0, 0.0))
        .point("bottom-right", (2.0, 0.0))
        .point("top-left", (0.0, 1.0))
        .build()
    )

    assert path.closed is True
    assert path.anchor_id == "bottom-left"
    assert [point.id for point in path.points] == ["bottom-left", "bottom-right", "top-left"]


def test_builder_point_can_declare_anchor_inline() -> None:
    path = (
        TopologyPath.closed()
        .point("bottom-left", (0.0, 0.0), anchor=True)
        .point("bottom-right", (2.0, 0.0))
        .point("top-left", (0.0, 1.0))
        .build()
    )

    assert path.anchor_id == "bottom-left"
    assert path.points[0].provenance["source"] == "builder.point"


def test_builder_point_correspond_alias_normalizes_to_correspondence_id() -> None:
    path = (
        TopologyPath.closed()
        .point("a", (0.0, 0.0), correspond="shared-a")
        .point("b", (1.0, 0.0))
        .point("c", (0.0, 1.0))
        .build()
    )

    assert path.points[0].correspondence_id == "shared-a"
    assert path.points[1].correspondence_id == "b"


def test_builder_segment_attaches_landmarks_to_path() -> None:
    path = (
        TopologyPath.closed()
        .point("a", (0.0, 0.0))
        .point("b", (1.0, 0.0))
        .point("c", (0.0, 1.0))
        .segment("crown", correspond="crown")
        .build()
    )

    assert path.segments[0].id == "crown"
    assert path.segments[0].correspondence_id == "crown"


def test_builder_segment_accepts_authored_point_arrays() -> None:
    path = (
        TopologyPath.closed()
        .segment("outer", points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], correspond="outer")
        .build()
    )

    assert path.segments[0].id == "outer"
    assert path.segments[0].source_kind == "polyline"
    assert path.segments[0].correspondence_id == "outer"
    assert len(path.segments[0].points) == 3


def test_builder_segment_accepts_explicit_id_for_curve_reference() -> None:
    curve = object()

    path = (
        TopologyPath.closed()
        .point("a", (0.0, 0.0))
        .point("b", (1.0, 0.0))
        .point("c", (0.0, 1.0))
        .segment("crown", id="crown-curve", curve=curve)
        .build()
    )

    assert path.segments[0].id == "crown-curve"
    assert path.segments[0].curve is curve
    assert path.segments[0].source_kind == "curve"


def test_builder_rejects_duplicate_point_ids_at_point_boundary() -> None:
    builder = TopologyPath.closed().point("a", (0.0, 0.0))

    with pytest.raises(ValueError, match="point id 'a' duplicates"):
        builder.point("a", (1.0, 0.0))


def test_builder_rejects_invalid_point_fields_at_point_boundary() -> None:
    with pytest.raises(ValueError, match="point name"):
        TopologyPath.closed().point("", (0.0, 0.0))

    with pytest.raises(ValueError, match="point correspond"):
        TopologyPath.closed().point("a", (0.0, 0.0), correspond="")

    with pytest.raises(ValueError, match="finite coordinates"):
        TopologyPath.closed().point("a", (float("nan"), 0.0))


def test_builder_rejects_invalid_segment_fields_at_segment_boundary() -> None:
    builder = TopologyPath.closed()

    with pytest.raises(ValueError, match="segment name"):
        builder.segment("", points=[(0.0, 0.0), (1.0, 0.0)])

    with pytest.raises(ValueError, match="both points and curve"):
        builder.segment("bad", points=[(0.0, 0.0), (1.0, 0.0)], curve=object())

    with pytest.raises(ValueError, match="at least two points"):
        builder.segment("bad", points=[(0.0, 0.0)])

    builder.segment("edge", points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    with pytest.raises(ValueError, match="segment id 'edge' duplicates"):
        builder.segment("edge")


def test_builder_requires_valid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        (
            TopologyPath.closed(direction="sideways")
            .point("a", (0.0, 0.0))
            .point("b", (1.0, 0.0))
            .point("c", (0.0, 1.0))
            .build()
        )
