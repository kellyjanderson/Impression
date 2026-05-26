from types import SimpleNamespace

import numpy as np
import pytest

from impression.modeling.loft import (
    SampleCorrespondenceRecord,
    emit_surface_patches_from_sample_correspondence,
    resolve_authored_rails,
    resolve_point_birth_death_events,
    resample_loop_correspondence,
    validate_surface_executor_correspondence_input,
)
from impression.modeling.topology import TopologyPath


def _path(points: tuple[tuple[str, tuple[float, float], str], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates, correspondence_id in points:
        builder.point(name, coordinates, correspond=correspondence_id)
    return builder.build()


def test_surface_ruled_patch_boundaries_follow_sample_correspondence() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a", (0.0, 0.0), "a"), ("b", (2.0, 0.0), "b"), ("c", (0.0, 2.0), "c")))
    resampled = resample_loop_correspondence({"source": source, "target": target}, sample_count="auto")

    body = emit_surface_patches_from_sample_correspondence(resampled)
    patch = body.shells[0].patches[0]

    assert body.patch_count == 1
    assert patch.start_curve.shape == (3, 3)
    assert patch.end_curve.shape == (3, 3)
    assert tuple(patch.end_curve[1]) == (2.0, 0.0, 1.0)
    assert patch.metadata["sample_records"][1].track_id == "b->b"


def test_surface_preserves_protected_landmarks_as_boundary_samples() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    resampled = resample_loop_correspondence({"source": source, "target": target}, sample_count=5)

    body = emit_surface_patches_from_sample_correspondence(resampled)

    assert body.shells[0].patches[0].metadata["protected_indices"] == (0, 1, 2)


def test_surface_preserves_birth_support_in_output_metadata() -> None:
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

    body = emit_surface_patches_from_sample_correspondence(resampled)

    assert any(record.lifecycle_event_id for record in body.metadata["sample_records"])


def test_surface_executor_refuses_missing_records_without_mesh_fallback() -> None:
    broken = SimpleNamespace(
        source_samples=np.zeros((3, 2)),
        target_samples=np.zeros((3, 2)),
        sample_records=(),
        diagnostics={},
    )

    with pytest.raises(ValueError, match="missing_sample_correspondence"):
        emit_surface_patches_from_sample_correspondence(broken)


def test_surface_executor_refuses_semantically_invalid_lifecycle_sample() -> None:
    broken = SimpleNamespace(
        source_samples=np.zeros((1, 2)),
        target_samples=np.zeros((1, 2)),
        sample_records=(SampleCorrespondenceRecord(index=0, lifecycle_event_id="birth-a", track_id=None),),
        diagnostics={},
    )

    with pytest.raises(ValueError, match="missing_lifecycle_support"):
        validate_surface_executor_correspondence_input(broken)
