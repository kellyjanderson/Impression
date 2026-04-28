from __future__ import annotations

from impression.modeling import (
    Line3D,
    Path3D,
    PathBackedProgression,
    ProgressionProvenanceRecord,
)


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
