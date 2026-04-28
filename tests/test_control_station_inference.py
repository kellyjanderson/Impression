from __future__ import annotations

from impression.modeling import (
    HiddenControlStationProvenanceRecord,
    HiddenControlStationRecord,
    Path3D,
    PathBackedProgression,
    ProgressionStationAttachment,
    ReducedProgressionBundle,
    RetainedStationRecord,
    Station,
    as_section,
)
from impression.modeling.drawing2d import make_rect


def _progression() -> PathBackedProgression:
    return PathBackedProgression(path=Path3D.from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]))


def _station(progress: float = 0.5) -> Station:
    return Station(
        t=progress,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, progress),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )


def test_reduced_progression_bundles_emit_the_required_replay_payload_structure() -> None:
    bundle = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-0",
        progression=_progression(),
        retained_progression_values=(0.0, 0.5, 1.0),
        hidden_control_station_ids=("control-0",),
    )

    assert bundle.replay_payload[0] == "reduced_progression_bundle"
    assert bundle.replay_payload[3] == _progression().identity


def test_bundle_local_provenance_fields_remain_inspectable() -> None:
    bundle = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-1",
        progression=_progression(),
        retained_progression_values=(0.0, 1.0),
        hidden_control_station_ids=("control-1",),
        provenance_source="dense_station_inference",
    )

    assert bundle.provenance_source == "dense_station_inference"


def test_reduced_progression_output_is_replayable() -> None:
    bundle = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-2",
        progression=_progression(),
        retained_progression_values=(0.0, 0.5, 1.0),
        hidden_control_station_ids=("control-2", "control-3"),
    )

    assert bundle.replay_payload[-2] == ("control-2", "control-3")


def test_bundle_shape_is_stable_for_identical_inputs() -> None:
    first = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-3",
        progression=_progression(),
        retained_progression_values=(0.0, 0.5, 1.0),
        hidden_control_station_ids=("control-4",),
    )
    second = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-3",
        progression=_progression(),
        retained_progression_values=(0.0, 0.5, 1.0),
        hidden_control_station_ids=("control-4",),
    )

    assert first.replay_payload == second.replay_payload


def test_bundle_local_provenance_remains_explicit() -> None:
    bundle = ReducedProgressionBundle.from_progression(
        bundle_id="reduction-4",
        progression=_progression(),
        retained_progression_values=(0.0, 1.0),
        hidden_control_station_ids=("control-5",),
        provenance_source="shared_trajectory_fit",
    )

    assert bundle.provenance_source == "shared_trajectory_fit"


def test_reduced_progression_results_retain_topology_and_hidden_control_station_classifications() -> None:
    attachment = ProgressionStationAttachment.from_station(
        progression=_progression(),
        station=_station(0.25),
        station_index=0,
    )
    hidden = HiddenControlStationRecord.from_station(
        station_id="control-0",
        station=_station(0.75),
        provenance=HiddenControlStationProvenanceRecord(),
    )

    topology = RetainedStationRecord.from_attachment(
        station_id="topo-0",
        attachment=attachment,
    )
    control = RetainedStationRecord.from_hidden_control(
        record=hidden,
        progression_value=0.75,
    )

    assert topology.kind == "topology"
    assert control.kind == "hidden_control"


def test_supporting_diagnostic_references_remain_inspectable_from_retained_station_records() -> None:
    attachment = ProgressionStationAttachment.from_station(
        progression=_progression(),
        station=_station(0.5),
        station_index=0,
    )
    record = RetainedStationRecord.from_attachment(
        station_id="topo-1",
        attachment=attachment,
        diagnostic_references=("diag-a", "diag-b"),
    )

    assert record.diagnostic_references == ("diag-a", "diag-b")


def test_retained_station_class_recording_is_explicit() -> None:
    record = RetainedStationRecord(
        station_id="topo-2",
        kind="topology",
        progression_value=1.0,
    )

    assert record.kind == "topology"


def test_diagnostic_association_does_not_blur_topology_and_control_station_classes() -> None:
    topology = RetainedStationRecord(
        station_id="shared-id",
        kind="topology",
        progression_value=0.0,
        diagnostic_references=("diag-topo",),
    )
    control = RetainedStationRecord(
        station_id="shared-id",
        kind="hidden_control",
        progression_value=0.5,
        diagnostic_references=("diag-control",),
    )

    assert topology.kind != control.kind
    assert topology.diagnostic_references != control.diagnostic_references


def test_retained_structure_remains_inspectable_after_reduction() -> None:
    record = RetainedStationRecord(
        station_id="topo-3",
        kind="topology",
        progression_value=0.25,
        diagnostic_references=("diag-structure",),
    )

    assert record.identity[0] == "retained_station_record"
    assert record.progression_value == 0.25
