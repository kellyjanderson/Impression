"""Hex nut generation with internal thread cutter subtraction."""

from impression.modeling import MeshQuality, lookup_standard_thread, make_hex_nut, make_threaded_rod, translate, group


def build():
    spec = lookup_standard_thread("metric", "M6x1", length=8.0)
    rod = make_threaded_rod(spec, quality=MeshQuality(segments_per_turn=20))
    nut = make_hex_nut(spec, thickness=5.0, across_flats=10.0, quality=MeshQuality(segments_per_turn=20))

    translate(rod, (-8.0, 0.0, 0.0))
    translate(nut, (8.0, 0.0, 0.0))
    return group([rod, nut])
