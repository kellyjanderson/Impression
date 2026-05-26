from types import SimpleNamespace

import numpy as np
import pytest

from impression.modeling.loft import (
    emit_mesh_faces_from_sample_correspondence,
    resolve_authored_rails,
    resolve_point_birth_death_events,
    resample_loop_correspondence,
    validate_mesh_executor_correspondence_input,
)
from impression.modeling.topology import TopologyPath


def _path(points: tuple[tuple[str, tuple[float, float], str], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates, correspondence_id in points:
        builder.point(name, coordinates, correspond=correspondence_id)
    return builder.build()


def test_mesh_face_indices_follow_sample_correspondence_records() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    resampled = resample_loop_correspondence({"source": source, "target": target}, sample_count="auto")

    mesh = emit_mesh_faces_from_sample_correspondence(resampled)

    assert mesh.n_vertices == 6
    assert mesh.n_faces == 6
    assert tuple(mesh.faces[0]) == (0, 1, 4)
    assert mesh.metadata["sample_records"][0].track_id == "a->a"


def test_mesh_emits_faces_for_birth_support_samples() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path(
        (("a", (0.0, 0.0), "a"), ("born", (0.5, 0.0), "born"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c"))
    )
    rail_result = resolve_authored_rails(source, target)
    lifecycle = resolve_point_birth_death_events({"source": source, "target": target}, rail_result, None)
    resampled = resample_loop_correspondence(
        {"source": source, "target": target, "rail_result": rail_result, "lifecycle_resolution": lifecycle},
        sample_count="auto",
    )

    mesh = emit_mesh_faces_from_sample_correspondence(resampled)

    assert mesh.n_faces == len(resampled.sample_records) * 2
    assert any(record.lifecycle_event_id for record in mesh.metadata["sample_records"])


def test_mesh_emits_faces_for_death_support_samples() -> None:
    source = _path(
        (("a", (0.0, 0.0), "a"), ("dying", (0.5, 0.0), "dying"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c"))
    )
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    rail_result = resolve_authored_rails(source, target)
    lifecycle = resolve_point_birth_death_events({"source": source, "target": target}, rail_result, None)
    resampled = resample_loop_correspondence(
        {"source": source, "target": target, "rail_result": rail_result, "lifecycle_resolution": lifecycle},
        sample_count="auto",
    )

    mesh = emit_mesh_faces_from_sample_correspondence(resampled)

    assert mesh.n_faces == len(resampled.sample_records) * 2
    assert any(record.source_point_ref == "dying" for record in mesh.metadata["sample_records"])


def test_mesh_executor_refuses_missing_sample_records() -> None:
    broken = SimpleNamespace(
        source_samples=np.zeros((3, 2)),
        target_samples=np.zeros((3, 2)),
        sample_records=(),
        diagnostics={},
    )

    with pytest.raises(ValueError, match="missing_sample_correspondence"):
        validate_mesh_executor_correspondence_input(broken)


def test_mesh_executor_refuses_mismatched_record_indices_before_partial_output() -> None:
    broken = SimpleNamespace(
        source_samples=np.zeros((3, 2)),
        target_samples=np.zeros((3, 2)),
        sample_records=(SimpleNamespace(index=1, track_id="bad"),) * 3,
        diagnostics={},
    )

    with pytest.raises(ValueError, match="sample_record_index_mismatch"):
        emit_mesh_faces_from_sample_correspondence(broken)
