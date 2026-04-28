from __future__ import annotations

from impression.modeling import (
    Line3D,
    Path3D,
    PathBackedProgression,
    ProgressionStationAttachment,
    ProgressionProvenanceRecord,
    ProgressionTransportPolicy,
    Station,
    as_section,
)
from impression.modeling.drawing2d import make_rect


def test_progression_objects_reference_an_underlying_path_explicitly() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    progression = PathBackedProgression(path=path)

    assert progression.path is path
    assert progression.parameter_domain == (0.0, 1.0)


def test_explicit_vs_inferred_provenance_remains_inspectable() -> None:
    explicit = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
        provenance=ProgressionProvenanceRecord(kind="explicit", source="authored_path"),
    )
    inferred = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.5, 0.2, 1.0)]),
        provenance=ProgressionProvenanceRecord(
            kind="inferred",
            source="dense_station_inference",
        ),
    )

    assert explicit.provenance.kind == "explicit"
    assert inferred.provenance.kind == "inferred"
    assert inferred.provenance.source == "dense_station_inference"


def test_progression_remains_distinct_from_the_raw_path_primitive() -> None:
    path = Path3D(segments=[Line3D((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))])
    progression = PathBackedProgression(path=path)

    assert isinstance(path, Path3D)
    assert isinstance(progression, PathBackedProgression)
    assert progression != path


def test_provenance_remains_durable_and_replayable() -> None:
    provenance = ProgressionProvenanceRecord(
        kind="inferred",
        source="shared_trajectory_fit",
    )

    first = provenance
    second = ProgressionProvenanceRecord(
        kind="inferred",
        source="shared_trajectory_fit",
    )

    assert first == second


def test_progression_identity_is_stable_enough_for_later_diagnostics() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.1, 1.0)])
    first = PathBackedProgression(
        path=path,
        domain_start=2.0,
        domain_end=4.0,
        provenance=ProgressionProvenanceRecord(kind="explicit", source="authored_path"),
    )
    second = PathBackedProgression(
        path=path,
        domain_start=2.0,
        domain_end=4.0,
        provenance=ProgressionProvenanceRecord(kind="explicit", source="authored_path"),
    )

    assert first.identity == second.identity


def test_stations_attach_to_progression_explicitly_rather_than_via_loose_scalar_arrays() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    )
    station = Station(
        t=0.25,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, 0.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )

    attachment = ProgressionStationAttachment.from_station(
        progression=progression,
        station=station,
        station_index=0,
    )

    assert attachment.progression_identity == progression.identity
    assert attachment.progression_value == 0.25


def test_attachment_ordering_and_identity_remain_inspectable() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
    )
    stations = [
        Station(
            t=0.0,
            section=as_section(make_rect(size=(1.0, 1.0))),
            origin=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=1.0,
            section=as_section(make_rect(size=(0.8, 1.2))),
            origin=(0.0, 0.0, 1.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]

    attachments = progression.attach_stations(stations)

    assert [attachment.station_index for attachment in attachments] == [0, 1]
    assert attachments[0].identity != attachments[1].identity


def test_topology_owned_station_truth_remains_intact_after_attachment() -> None:
    section = as_section(make_rect(size=(1.0, 0.6)))
    station = Station(
        t=0.5,
        section=section,
        origin=(0.0, 0.0, 0.5),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    )

    attachment = ProgressionStationAttachment.from_station(
        progression=progression,
        station=station,
        station_index=0,
    )

    assert attachment.topology_state is station.topology_state
    assert attachment.directional_correspondence == station.directional_correspondence


def test_attachment_ordering_remains_deterministic() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
    )
    stations = [
        Station(
            t=0.25,
            section=as_section(make_rect(size=(1.0, 1.0))),
            origin=(0.0, 0.0, 0.5),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=0.75,
            section=as_section(make_rect(size=(0.8, 0.8))),
            origin=(0.0, 0.0, 1.5),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]

    first = progression.attach_stations(stations)
    second = progression.attach_stations(stations)

    assert tuple(attachment.identity for attachment in first) == tuple(
        attachment.identity for attachment in second
    )


def test_station_attachment_remains_durable_enough_for_replay() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    )
    station = Station(
        t=0.5,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, 0.5),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )

    first = ProgressionStationAttachment.from_station(
        progression=progression,
        station=station,
        station_index=0,
    )
    second = ProgressionStationAttachment.from_station(
        progression=progression,
        station=station,
        station_index=0,
    )

    assert first.identity == second.identity


def test_progression_records_expose_explicit_transport_semantics() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
        transport_policy=ProgressionTransportPolicy(kind="parallel_transport"),
    )

    assert progression.transport_policy.kind == "parallel_transport"


def test_loft_consumes_transport_semantics_through_an_inspectable_contract() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.0, 1.0)]),
    )
    contract = progression.loft_transport_contract

    assert contract.progression_identity == progression.identity
    assert contract.transport_policy == progression.transport_policy


def test_transport_semantics_remain_deterministic_for_identical_inputs() -> None:
    kwargs = dict(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.0, 1.0)]),
        transport_policy=ProgressionTransportPolicy(kind="parallel_transport"),
    )

    first = PathBackedProgression(**kwargs)
    second = PathBackedProgression(**kwargs)

    assert first.loft_transport_contract == second.loft_transport_contract


def test_transport_policy_remains_separate_from_twist_and_scale_semantic_slots() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
    )

    assert progression.transport_policy.kind == "parallel_transport"
    assert not hasattr(progression.transport_policy, "twist")
    assert not hasattr(progression.transport_policy, "scale")


def test_transport_ownership_is_durable_enough_for_replay_and_diagnostics() -> None:
    progression = PathBackedProgression(
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.1, 1.0)]),
        transport_policy=ProgressionTransportPolicy(kind="parallel_transport"),
    )

    first = progression.loft_transport_contract
    second = progression.loft_transport_contract

    assert first == second
