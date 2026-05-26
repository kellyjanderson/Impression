import pytest

from impression.modeling.topology import (
    TopologyLandmark,
    TopologyPath,
    TopologyPoint,
    TopologySegment,
    derive_stable_id,
)


def test_derive_stable_id_normalizes_user_facing_names() -> None:
    assert derive_stable_id("Bottom Left") == "bottom-left"
    assert derive_stable_id("  Crown/Peak!  ") == "crown-peak"


def test_topology_point_derives_id_and_protection_policy_from_name_and_role() -> None:
    point = TopologyPoint(
        id=None,
        name="Bottom Left",
        coordinates=(0.0, 0.0),
        ordinal=0,
        role="corner",
        correspondence_id="corner-a",
    )

    assert point.id == "bottom-left"
    assert point.protection_policy == "protected"
    assert point.provenance["id_source"] == "derived_from_name"


def test_topology_identity_validation_rejects_duplicate_point_ids() -> None:
    p0 = TopologyPoint(id="a", coordinates=(0.0, 0.0), ordinal=0)
    p1 = TopologyPoint(id="a", coordinates=(1.0, 0.0), ordinal=1)
    p2 = TopologyPoint(id="c", coordinates=(0.0, 1.0), ordinal=2)

    with pytest.raises(ValueError, match="duplicate point ids"):
        TopologyPath(points=(p0, p1, p2))


def test_topology_landmark_rejects_missing_segment_reference() -> None:
    segment = TopologySegment(id="edge-a", name="edge-a")
    landmark = TopologyLandmark(
        name="missing-segment-landmark",
        segment_id="edge-b",
        parameter=0.5,
    )

    with pytest.raises(ValueError, match="segment_id"):
        TopologyPath(
            points=(
                TopologyPoint(id="a", coordinates=(0.0, 0.0), ordinal=0),
                TopologyPoint(id="b", coordinates=(1.0, 0.0), ordinal=1),
                TopologyPoint(id="c", coordinates=(0.0, 1.0), ordinal=2),
            ),
            segments=(segment,),
            landmarks=(landmark,),
        )


def test_topology_landmark_rejects_point_ordinal_outside_path_points() -> None:
    landmark = TopologyLandmark(name="bad-point", point_ordinal=3)

    with pytest.raises(ValueError, match="point_ordinal"):
        TopologyPath.from_points(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            landmarks=(landmark,),
        )


def test_topology_identity_validation_rejects_conflicting_protection_policy() -> None:
    point = TopologyPoint(
        id="a",
        coordinates=(0.0, 0.0),
        ordinal=0,
        correspondence_id="shared",
        protection_policy="protected",
    )
    landmark = TopologyLandmark(
        name="sample-shared",
        point_ordinal=0,
        correspondence_id="shared",
        protection_policy="sample",
    )

    with pytest.raises(ValueError, match="Conflicting protection policies"):
        TopologyPath(
            points=(
                point,
                TopologyPoint(id="b", coordinates=(1.0, 0.0), ordinal=1),
                TopologyPoint(id="c", coordinates=(0.0, 1.0), ordinal=2),
            ),
            landmarks=(landmark,),
        )
