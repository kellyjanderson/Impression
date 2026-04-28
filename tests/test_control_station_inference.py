from __future__ import annotations

from impression.modeling import (
    Path3D,
    PathBackedProgression,
    ReducedProgressionBundle,
)


def _progression() -> PathBackedProgression:
    return PathBackedProgression(path=Path3D.from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]))


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
