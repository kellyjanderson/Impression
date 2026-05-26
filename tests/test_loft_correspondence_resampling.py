import pytest

from impression.modeling.loft import (
    PointLifecycleResolution,
    ResampledLoopCorrespondence,
    SampleCorrespondenceRecord,
    resolve_authored_rails,
    resolve_point_birth_death_events,
    resample_loop_correspondence,
    validate_sample_correspondence,
)
from impression.modeling.topology import TopologyPath


def _path(points: tuple[tuple[str, tuple[float, float], str], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates, correspondence_id in points:
        builder.point(name, coordinates, correspond=correspondence_id)
    return builder.build()


def test_protected_rail_matches_survive_as_exact_sample_records() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a2", (0.0, 0.0), "a"), ("b2", (2.0, 0.0), "b"), ("c2", (0.0, 2.0), "c")))
    rail_result = resolve_authored_rails(source, target)

    resampled = resample_loop_correspondence(
        {"source": source, "target": target, "rail_result": rail_result},
        sample_count="auto",
    )

    assert isinstance(resampled, ResampledLoopCorrespondence)
    assert len(resampled.source_samples) == len(resampled.target_samples) == len(resampled.sample_records)
    assert resampled.sample_records[0].source_point_ref == "a"
    assert resampled.sample_records[0].target_point_ref == "a2"
    assert tuple(resampled.source_samples[0]) == (0.0, 0.0)
    assert tuple(resampled.target_samples[1]) == (2.0, 0.0)
    assert set(resampled.protected_indices) == {0, 1, 2}


def test_synthetic_birth_support_produces_paired_sample() -> None:
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

    lifecycle_records = [record for record in resampled.sample_records if record.lifecycle_event_id]
    assert len(lifecycle_records) == 1
    assert lifecycle_records[0].target_point_ref == "born"
    assert lifecycle_records[0].protected is True


def test_explicit_sample_count_cap_refuses_when_too_low() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))

    with pytest.raises(ValueError, match="sample_count_too_low"):
        resample_loop_correspondence({"source": source, "target": target}, sample_count=2)


def test_sample_records_align_with_arrays_and_unprotected_fill_samples() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))

    resampled = resample_loop_correspondence({"source": source, "target": target}, sample_count=5)

    validate_sample_correspondence(resampled)
    assert [record.index for record in resampled.sample_records] == list(range(5))
    assert sum(1 for record in resampled.sample_records if not record.protected) == 2


def test_missing_lifecycle_support_refuses_before_executor_input() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path(
        (("a", (0.0, 0.0), "a"), ("born", (0.5, 0.0), "born"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c"))
    )
    rail_result = resolve_authored_rails(source, target)
    lifecycle = resolve_point_birth_death_events({"source": source, "target": target}, rail_result, None)
    lifecycle_without_support = PointLifecycleResolution(events=lifecycle.events)

    with pytest.raises(ValueError, match="missing_lifecycle_support"):
        resample_loop_correspondence(
            {"source": source, "target": target, "rail_result": rail_result, "lifecycle_resolution": lifecycle_without_support},
            sample_count="auto",
        )


def test_validate_sample_correspondence_rejects_record_misalignment() -> None:
    with pytest.raises(ValueError, match="indices"):
        ResampledLoopCorrespondence(
            source_samples=[(0.0, 0.0)],
            target_samples=[(0.0, 0.0)],
            sample_records=(SampleCorrespondenceRecord(index=1),),
        )
