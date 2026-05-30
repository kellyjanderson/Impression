import numpy as np
import pytest

from impression.modeling.topology import (
    Loop,
    TopologyPath,
    TopologyPathSamplingPolicy,
)


def test_topology_path_from_points_preserves_authored_anchor_and_direction() -> None:
    path = TopologyPath.from_points(
        [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)],
        closed=True,
    )

    assert path.closed is True
    assert path.anchor_id == "point-0"
    assert path.anchor_policy == "authored"
    assert path.direction == "forward"
    assert [point.id for point in path.points] == ["point-0", "point-1", "point-2"]


def test_topology_path_named_anchor_uses_authored_name() -> None:
    path = TopologyPath.from_points(
        [
            ("bottom-left", (0.0, 0.0)),
            ("bottom-right", (1.0, 0.0)),
            ("top-left", (0.0, 1.0)),
        ],
        anchor="bottom-left",
    )

    assert path.anchor_id == "bottom-left"
    assert path.points[0].correspondence_id == "bottom-left"
    assert path.points[0].protection_policy == "protected"


def test_topology_sampling_policy_validates_auto_and_positive_counts() -> None:
    assert TopologyPathSamplingPolicy().sample_count == "auto"
    assert TopologyPathSamplingPolicy(sample_count=8).sample_count == 8

    with pytest.raises(ValueError, match="sample_count"):
        TopologyPathSamplingPolicy(sample_count=0)

    with pytest.raises(ValueError, match="min_span_samples"):
        TopologyPathSamplingPolicy(min_span_samples=0)


def test_topology_path_rejects_invalid_path_level_state() -> None:
    with pytest.raises(ValueError, match="requires at least one point or segment"):
        TopologyPath(id="empty", points=())

    with pytest.raises(ValueError, match="direction"):
        TopologyPath.from_points(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            direction="sideways",
        )

    with pytest.raises(ValueError, match="finite coordinates"):
        TopologyPath.from_points([(0.0, 0.0), (1.0, 0.0), (np.nan, 1.0)])


def test_topology_path_to_section_loop_keeps_authored_order() -> None:
    path = TopologyPath.from_points(
        [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)],
        closed=True,
    )

    loop = path.to_section_loop()

    assert isinstance(loop, Loop)
    np.testing.assert_allclose(loop.points, np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]))
