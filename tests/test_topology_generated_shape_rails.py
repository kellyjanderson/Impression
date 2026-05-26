import pytest

from impression.modeling.topology import GeneratedRailProvenance, TopologyPath


def _generated_roles(path: TopologyPath) -> dict[str, str]:
    return {point.name: point.provenance["generated_rail"].generated_role for point in path.points}


def test_named_rect_generates_corner_and_midpoint_rails() -> None:
    path = TopologyPath.named_rect(4.0, 2.0)

    assert path.anchor_id == "bottom-left"
    assert [point.name for point in path.points] == [
        "bottom-left",
        "bottom-mid",
        "bottom-right",
        "right-mid",
        "top-right",
        "top-mid",
        "top-left",
        "left-mid",
    ]
    roles = _generated_roles(path)
    assert roles["bottom-left"] == "corner"
    assert roles["bottom-mid"] == "side_midpoint"
    assert path.points[0].correspondence_id == "bottom-left"
    assert isinstance(path.points[0].provenance["generated_rail"], GeneratedRailProvenance)


def test_named_circle_generates_start_and_quadrant_rails() -> None:
    path = TopologyPath.named_circle(2.0)

    assert path.anchor_id == "start"
    assert [point.name for point in path.points] == ["start", "north", "west", "south"]
    assert _generated_roles(path) == {
        "start": "start",
        "north": "quadrant",
        "west": "quadrant",
        "south": "quadrant",
    }


def test_named_rounded_rect_generates_segments_arcs_and_tangent_landmarks() -> None:
    path = TopologyPath.named_rounded_rect(4.0, 2.0, 0.25)

    assert path.anchor_id == "bottom-left"
    assert [segment.name for segment in path.segments] == [
        "bottom",
        "bottom-right-arc",
        "right",
        "top-right-arc",
        "top",
        "top-left-arc",
        "left",
        "bottom-left-arc",
    ]
    assert {segment.source_kind for segment in path.segments} == {"straight", "corner_arc"}
    assert len(path.landmarks) == 16
    assert {landmark.role for landmark in path.landmarks} == {"tangent_transition"}


def test_name_prefix_namespaces_generated_ids_and_anchor() -> None:
    path = TopologyPath.named_rect(4.0, 2.0, name_prefix="outer shell", anchor="outer-shell-top-left")

    assert path.anchor_id == "outer-shell-top-left"
    assert path.points[0].id == "outer-shell-bottom-left"
    assert path.points[0].name == "outer-shell-bottom-left"
    assert path.points[0].correspondence_id == "outer-shell-bottom-left"
    provenance = path.points[0].provenance["generated_rail"]
    assert provenance.name_prefix == "outer shell"
    assert provenance.source_parameter == "bottom-left"


def test_generated_shape_rails_validate_dimensions_and_anchor() -> None:
    with pytest.raises(ValueError, match="positive"):
        TopologyPath.named_rect(0.0, 2.0)

    with pytest.raises(ValueError, match="positive"):
        TopologyPath.named_circle(0.0)

    with pytest.raises(ValueError, match="fit"):
        TopologyPath.named_rounded_rect(1.0, 1.0, 0.75)

    with pytest.raises(ValueError, match="valid anchors"):
        TopologyPath.named_rect(1.0, 1.0, anchor="missing")
