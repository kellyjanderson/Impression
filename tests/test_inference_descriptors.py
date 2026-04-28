from __future__ import annotations

from impression.modeling import (
    Station,
    assemble_span_local_curve_intent_evidence,
    as_section,
    build_curve_intent_descriptor_families,
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
            origin=(0.1, 0.0, 0.5),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=1.0,
            section=as_section(make_rect(size=(0.8, 1.2))),
            origin=(0.2, 0.0, 1.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]


def test_dense_loft_fixtures_emit_descriptor_records_in_deterministic_order() -> None:
    band = prepare_dense_loft_fit_descriptors(_dense_fixture())

    assert band.station_indices == (0, 1, 2)
    assert band.progression_values == (0.0, 0.5, 1.0)


def test_descriptor_normalization_preserves_station_ordering_and_correspondence_meaning() -> None:
    band = prepare_dense_loft_fit_descriptors(_dense_fixture())

    assert [descriptor.directional_correspondence_count for descriptor in band.descriptors] == [1, 1, 1]
    assert [descriptor.region_count for descriptor in band.descriptors] == [1, 1, 1]


def test_prepared_descriptor_bands_are_replayable_for_identical_inputs() -> None:
    first = prepare_dense_loft_fit_descriptors(_dense_fixture())
    second = prepare_dense_loft_fit_descriptors(_dense_fixture())

    assert first == second


def test_ordering_and_continuity_survive_descriptor_preparation() -> None:
    band = prepare_dense_loft_fit_descriptors(_dense_fixture())

    assert tuple(sorted(band.progression_values)) == band.progression_values
    assert band.descriptors[0].origin[2] < band.descriptors[1].origin[2] < band.descriptors[2].origin[2]


def test_prepared_evidence_is_consumable_by_later_candidate_fit_branches() -> None:
    band = prepare_dense_loft_fit_descriptors(_dense_fixture())

    assert all(descriptor.identity[0] == "dense_loft_station_descriptor" for descriptor in band.descriptors)


def test_section_loop_and_correspondence_track_descriptor_records_emit_in_deterministic_order() -> None:
    families = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    assert [descriptor.station_index for descriptor in families.section_descriptors] == [0, 1, 2]
    assert [descriptor.station_index for descriptor in families.loop_descriptors] == [0, 1, 2]
    assert [descriptor.station_index for descriptor in families.correspondence_track_descriptors] == [0, 1, 2]


def test_family_level_descriptor_fields_remain_inspectable() -> None:
    families = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    assert families.section_descriptors[1].progression_value == 0.5
    assert families.loop_descriptors[1].loop_count == 1
    assert families.correspondence_track_descriptors[1].correspondence_track_count == 1


def test_descriptor_families_are_durable_enough_for_replay_and_comparison() -> None:
    first = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )
    second = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    assert first == second


def test_family_boundaries_remain_stable_for_identical_inputs() -> None:
    families = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    assert len(families.section_descriptors) == 3
    assert len(families.loop_descriptors) == 3
    assert len(families.correspondence_track_descriptors) == 3


def test_descriptor_families_remain_reusable_by_later_evidence_assembly() -> None:
    families = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    assert tuple(descriptor.station_index for descriptor in families.section_descriptors) == (0, 1, 2)


def test_span_local_evidence_records_remain_inspectable() -> None:
    evidence = assemble_span_local_curve_intent_evidence(
        build_curve_intent_descriptor_families(
            prepare_dense_loft_fit_descriptors(_dense_fixture())
        )
    )

    assert evidence[0].span_start_station_index == 0
    assert evidence[0].span_end_station_index == 1


def test_identical_descriptor_inputs_produce_identical_evidence_ordering() -> None:
    families = build_curve_intent_descriptor_families(
        prepare_dense_loft_fit_descriptors(_dense_fixture())
    )

    first = assemble_span_local_curve_intent_evidence(families)
    second = assemble_span_local_curve_intent_evidence(families)

    assert first == second


def test_span_local_evidence_shape_is_stable_for_identical_inputs() -> None:
    evidence = assemble_span_local_curve_intent_evidence(
        build_curve_intent_descriptor_families(
            prepare_dense_loft_fit_descriptors(_dense_fixture())
        )
    )

    assert len(evidence) == 2
    assert evidence[0].section_region_counts == (1, 1)


def test_normalization_and_ordering_remain_explicit() -> None:
    evidence = assemble_span_local_curve_intent_evidence(
        build_curve_intent_descriptor_families(
            prepare_dense_loft_fit_descriptors(_dense_fixture())
        )
    )

    assert tuple(item.span_start_station_index for item in evidence) == (0, 1)
    assert tuple(item.span_end_station_index for item in evidence) == (1, 2)


def test_later_candidate_classification_branches_can_consume_the_same_evidence_shape() -> None:
    evidence = assemble_span_local_curve_intent_evidence(
        build_curve_intent_descriptor_families(
            prepare_dense_loft_fit_descriptors(_dense_fixture())
        )
    )

    assert all(isinstance(item.loop_counts, tuple) for item in evidence)
    assert all(isinstance(item.correspondence_track_counts, tuple) for item in evidence)
