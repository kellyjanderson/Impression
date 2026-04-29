from __future__ import annotations

import sys
from pathlib import Path

from impression.modeling import make_text, translate

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import DEFAULT_LABEL_DIRECTION
from loft_bottle_nozzle_family_example import build as bottle_scene
from loft_ergonomic_handle_example import build as handle_scene
from loft_hvac_round_to_rect_example import build as hvac_scene
from loft_topology_vent_transition_example import build as vent_scene
from loft_wearable_enclosure_example import build as wearable_scene


def build():
    hero = []

    hvac = hvac_scene()
    for mesh in hvac:
        translate(mesh, (-32.0, 16.0, 0.0))
    hero.extend(hvac)

    handle = handle_scene()
    for mesh in handle:
        translate(mesh, (10.0, 16.0, 0.0))
    hero.extend(handle)

    bottle = bottle_scene()
    for mesh in bottle:
        translate(mesh, (-30.0, -20.0, 0.0))
    hero.extend(bottle)

    wearable = wearable_scene()
    for mesh in wearable:
        translate(mesh, (12.0, -20.0, 0.0))
    hero.extend(wearable)

    vent = vent_scene()
    for mesh in vent:
        translate(mesh, (48.0, -2.0, 0.0))
    hero.extend(vent)

    title = make_text(
        "LOFT: MECHANICAL + ORGANIC",
        depth=1.3,
        center=(12.0, 36.0, 24.0),
        direction=DEFAULT_LABEL_DIRECTION,
        font_size=11.0,
        justify="center",
        valign="middle",
        color="#ff00ff",
    )
    title.face_colors = None
    title.color = (1.0, 0.0, 1.0, 1.0)
    subtitle = make_text(
        "real transitions, topology events, production intent",
        depth=0.9,
        center=(12.0, 28.0, 19.0),
        direction=DEFAULT_LABEL_DIRECTION,
        font_size=5.8,
        justify="center",
        valign="middle",
        color="#d1d5db",
    )
    subtitle.face_colors = None
    subtitle.color = (0.82, 0.84, 0.86, 1.0)
    hero.extend([title, subtitle])
    return hero
