from __future__ import annotations

from impression.modeling import (
    ExplicitSharedGuidanceAttachmentRecord,
    FitConfigurationRecord,
    KnotCountPolicyRecord,
    KnotPlacementPolicyRecord,
    ParameterizationPolicyRecord,
    Path3D,
    PathBackedProgression,
    Station,
    as_section,
    generate_shared_whole_loft_trajectory_candidates,
    prepare_dense_loft_fit_descriptors,
)
from impression.modeling.drawing2d import make_circle, make_rect


def _dense_fixture() -> list[Station]:
    return [
        Station(
            t=0.0,
            section=as_section(make_rect(size=(1.0, 1.0))),
            origin=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=0.5,
            section=as_section(make_circle(radius=0.6)),
            origin=(0.3, 0.1, 0.5),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=1.0,
            section=as_section(make_rect(size=(0.8, 1.2))),
            origin=(1.0, 0.0, 1.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]


def _fit_config() -> FitConfigurationRecord:
    return FitConfigurationRecord(
        parameterization_policy=ParameterizationPolicyRecord(method="uniform"),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=3),
        knot_placement_policy=KnotPlacementPolicyRecord(placement_method="uniform_internal"),
    )


def _progression() -> PathBackedProgression:
    return PathBackedProgression(path=Path3D.from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]))


def test_explicit_shared_guidance_attaches_through_a_durable_attachment_record() -> None:
    candidate = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )[0]

    record = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-0",
        progression=_progression(),
        candidate=candidate,
    )

    assert record.trajectory_candidate_id == candidate.candidate_id
    assert record.progression_identity == _progression().identity


def test_attachment_identity_remains_inspectable() -> None:
    candidate = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )[0]

    record = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-1",
        progression=_progression(),
        candidate=candidate,
    )

    assert record.identity[0] == "explicit_shared_guidance_attachment"
    assert record.identity[1] == "guidance-1"


def test_attachment_record_shape_is_stable_and_replayable() -> None:
    candidate = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )[0]
    first = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-2",
        progression=_progression(),
        candidate=candidate,
    )
    second = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-2",
        progression=_progression(),
        candidate=candidate,
    )

    assert first.replay_payload == second.replay_payload


def test_deterministic_attachment_identity_is_preserved() -> None:
    candidate = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )[0]
    first = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-3",
        progression=_progression(),
        candidate=candidate,
    )
    second = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-3",
        progression=_progression(),
        candidate=candidate,
    )

    assert first.identity == second.identity


def test_attachment_metadata_remains_durable_for_later_diagnostics() -> None:
    candidate = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )[0]
    record = ExplicitSharedGuidanceAttachmentRecord.from_candidate(
        guidance_id="guidance-4",
        progression=_progression(),
        candidate=candidate,
        metadata_entries=(("source", "authored_guidance"), ("note", "hourglass")), 
    )

    assert record.metadata_entries == (("source", "authored_guidance"), ("note", "hourglass"))
