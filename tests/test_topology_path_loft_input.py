from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import Loft, SurfaceBody, TopologyLandmark, TopologyPath
from impression.modeling.topology import as_section


def _diagonal_path(*, reverse: bool = False) -> TopologyPath:
    points = (
        ("diagonal-negative-corner", (-16.0, -16.0)),
        ("negative-y-positive-corner", (16.0, -16.0)),
        ("diagonal-positive-corner", (16.0, 16.0)),
    )
    if reverse:
        points = tuple(reversed(points))
    builder = TopologyPath.closed(
        anchor="diagonal-negative-corner",
        direction="reverse" if reverse else "forward",
        id="diagonal-half",
    )
    for name, coordinates in points:
        builder = builder.point(name, coordinates, correspond=name, role="corner")
    return builder.build()


@pytest.mark.parametrize("reverse", [False, True])
def test_as_section_preserves_topology_path_identity_without_mutation(reverse: bool) -> None:
    path = _diagonal_path(reverse=reverse)
    before = tuple(point.coordinates.copy() for point in path.points)

    section = as_section(path)

    assert section.metadata["topology_paths"] == (path,)
    retained = section.metadata["topology_paths"][0]
    assert retained.id == "diagonal-half"
    assert retained.anchor_id == "diagonal-negative-corner"
    assert retained.direction == ("reverse" if reverse else "forward")
    assert [point.id for point in retained.points] == [point.id for point in path.points]
    assert [point.correspondence_id for point in retained.points] == [
        point.correspondence_id for point in path.points
    ]
    assert all(point.protection_policy == "protected" for point in retained.points)
    assert all(np.array_equal(point.coordinates, original) for point, original in zip(path.points, before, strict=True))


def test_public_loft_accepts_topology_path_and_retains_planner_identity() -> None:
    path = _diagonal_path()

    body = Loft(
        progression=(0.0, 1.0),
        stations=((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
        topology=(path, path),
        samples=12,
        cap_ends=True,
        fairness_mode="off",
        fairness_weight=0.0,
    )

    assert isinstance(body, SurfaceBody)
    retained = body.metadata["source_topology_paths"]
    assert retained == ((path,), (path,))
    assert retained[0][0].points[0].correspondence_id == "diagonal-negative-corner"
    assert retained[0][0].points[0].protection_policy == "protected"


def test_topology_path_landmarks_roles_and_anchor_survive_normalization() -> None:
    landmark = TopologyLandmark(
        name="seam-landmark",
        point_ordinal=1,
        role="seam",
        correspondence_id="shared-seam",
    )
    path = TopologyPath.from_points(
        (("lower-left", (-1.0, -1.0)), ("lower-right", (1.0, -1.0)), ("top", (0.0, 1.0))),
        anchor="lower-right",
        landmarks=(landmark,),
        id="landmarked-triangle",
    )

    retained = as_section(path).metadata["topology_paths"][0]

    assert retained.anchor_id == "lower-right"
    assert retained.landmarks == (landmark,)
    assert retained.landmarks[0].role == "seam"
    assert retained.landmarks[0].protection_policy == "protected"


def test_open_topology_path_is_refused_before_loft_planning() -> None:
    path = TopologyPath.from_points(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        closed=False,
    )

    with pytest.raises(ValueError, match="TopologyPath must be closed"):
        Loft(
            progression=(0.0, 1.0),
            stations=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            topology=(path, path),
            samples=8,
        )


def test_duplicate_topology_identity_has_stable_diagnostic() -> None:
    with pytest.raises(ValueError, match="duplicates an existing topology point"):
        TopologyPath.closed().point("first", (0.0, 0.0), id="same").point(
            "second", (1.0, 0.0), id="same"
        )
