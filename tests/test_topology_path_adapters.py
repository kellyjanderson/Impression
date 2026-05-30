import numpy as np
import pytest

from impression.modeling.bspline import BSpline2D
from impression.modeling.drawing2d import Path2D
from impression.modeling.topology import TopologyLandmark, TopologyPath


def test_from_points_accepts_named_tuple_points_and_anchor() -> None:
    path = TopologyPath.from_points(
        [
            ("bottom-left", (0.0, 0.0)),
            ("bottom-right", (2.0, 0.0)),
            ("top-left", (0.0, 1.0)),
        ],
        anchor="bottom-left",
    )

    assert path.metadata["source"] == "from_points"
    assert [point.id for point in path.points] == ["bottom-left", "bottom-right", "top-left"]
    assert path.anchor_id == "bottom-left"


def test_from_path2d_preserves_sampled_order_and_source_metadata() -> None:
    source = Path2D.from_points([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)], closed=True)

    path = TopologyPath.from_path2d(source)

    assert path.metadata["source"] == "from_path2d"
    assert path.closed is True
    np.testing.assert_allclose(
        np.vstack([point.coordinates for point in path.points]),
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
    )


def test_from_bspline_attaches_landmarks_to_curve_segment_parameters() -> None:
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        degree=2,
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    landmark = TopologyLandmark(name="peak", parameter=0.5, correspondence_id="peak")

    path = TopologyPath.from_bspline(curve, landmarks=(landmark,), id="crown")

    assert path.metadata["source"] == "from_bspline"
    assert path.segments[0].id == "crown-bspline"
    assert path.landmarks[0].segment_id == "crown-bspline"
    assert path.landmarks[0].parameter == pytest.approx(0.5)
    assert path.landmarks[0].correspondence_id == "peak"


def test_from_path2d_rejects_invalid_anchor() -> None:
    source = Path2D.from_points([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)], closed=True)

    with pytest.raises(ValueError, match="anchor_id"):
        TopologyPath.from_path2d(source, anchor="missing")


def test_from_bspline_to_section_loop_samples_curve_when_needed() -> None:
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        degree=2,
        knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )

    loop = TopologyPath.from_bspline(curve, closed=True).to_section_loop()

    assert loop.points.shape[1] == 2
    assert loop.points.shape[0] > 3
