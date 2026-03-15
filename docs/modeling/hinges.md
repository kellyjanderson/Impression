# Modeling - Hinges

Imported from `impression.modeling`:

```python
from impression.modeling import (
    make_traditional_hinge_leaf,
    make_traditional_hinge_pair,
    make_living_hinge,
    make_bistable_hinge,
)
```

## Traditional Hinges

`make_traditional_hinge_leaf(...)` builds one hinge leaf with alternating knuckles and a pin bore.

`make_traditional_hinge_pair(...)` builds a two-leaf assembly and returns a `MeshGroup`:

- separate leaf meshes for assembly previews
- optional center pin (`include_pin=True`)
- `opened_angle_deg` to preview open/closed states

Example: `docs/examples/hinges/traditional_hinge_example.py`

## Living Hinges

`make_living_hinge(...)` creates a slit-pattern flex panel intended for print-in-place bending.

Key parameters:

- `hinge_band_width`
- `slit_width`
- `slit_pitch`
- `edge_margin`
- `bridge`

Example: `docs/examples/hinges/living_hinge_example.py`

## Bistable Hinges

`make_bistable_hinge(...)` creates a flexure-style over-center blank with anchors, shuttle, and struts.

Key parameters:

- `anchor_width`
- `shuttle_width`
- `shuttle_height`
- `ligament_width`
- `preload_offset`

Example: `docs/examples/hinges/bistable_hinge_example.py`

## Overview Example

Preview all hinge families in one scene:

```bash
impression preview docs/examples/hinges/hinges_overview_example.py
```
