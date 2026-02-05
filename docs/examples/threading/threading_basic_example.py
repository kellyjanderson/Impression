"""Basic external/internal thread generation with fit presets."""

from impression.modeling import (
    MeshQuality,
    apply_fit,
    group,
    lookup_standard_thread,
    make_external_thread,
    make_internal_thread,
    translate,
)


def build():
    base = lookup_standard_thread("metric", "M8x1.25", length=14.0)
    male = apply_fit(base, "fdm_default", kind="external")
    female = apply_fit(base, "fdm_default", kind="internal")

    male_mesh = make_external_thread(male, quality=MeshQuality(segments_per_turn=24))
    female_cutter = make_internal_thread(female, quality=MeshQuality(segments_per_turn=24))

    translate(male_mesh, (-8.0, 0.0, 0.0))
    translate(female_cutter, (8.0, 0.0, 0.0))

    return group([male_mesh, female_cutter])
