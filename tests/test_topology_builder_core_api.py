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


def test_builder_rejects_duplicate_names_on_build() -> None:
    builder = (
        TopologyPath.closed()
        .point("a", (0.0, 0.0))
        .point("a", (1.0, 0.0))
        .point("c", (0.0, 1.0))
    )

    with pytest.raises(ValueError, match="duplicate point ids"):
        builder.build()


def test_builder_requires_valid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        (
            TopologyPath.closed(direction="sideways")
            .point("a", (0.0, 0.0))
            .point("b", (1.0, 0.0))
            .point("c", (0.0, 1.0))
            .build()
        )
