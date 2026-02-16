from __future__ import annotations

from impression.modeling import Profile2D, loft, loft_endcaps, make_circle, make_rect, translate


def _profile(size, hole_radius):
    outer = make_rect(size=size).outer
    hole = make_circle(radius=hole_radius).outer
    return Profile2D(outer=outer, holes=[hole])


def build():
    profiles = [
        _profile((10.0, 6.0), 1.6),
        _profile((12.0, 8.0), 2.1),
    ]

    legacy = loft(
        profiles,
        cap_ends=True,
        start_cap="dome",
        end_cap="dome",
        cap_steps=8,
        start_cap_length=2.0,
        end_cap_length=2.0,
        cap_scale_dims="both",
    )

    experimental = loft_endcaps(
        profiles,
        endcap_mode="ROUND",
        endcap_amount=2.0,
        endcap_steps=12,
        endcap_placement="BOTH",
    )

    translate(legacy, (-18.0, 0.0, 0.0))
    translate(experimental, (18.0, 0.0, 0.0))
    return [legacy, experimental]

