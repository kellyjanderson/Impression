# Tutorial - Serious Modeling Workflow

This tutorial builds a more substantial part: a cable clamp with a curved bridge,
mounting holes, and a relief slot. It combines 2D profiles, extrusions, CSG,
and a light use of loft. The goal is to show how to think in steps and keep
the model watertight.

## Overview

We will build the clamp in layers:

1. Make a base plate from a 2D profile and linear extrude.
2. Cut mounting holes using CSG difference.
3. Build the curved bridge with a loft between two profiles.
4. Add a relief slot and union everything together.

## Step 1 - Base Plate Profile

```python
from impression.modeling import make_rect, make_circle
from impression.modeling.drawing2d import Profile2D
from impression.modeling import linear_extrude


def make_base():
    outer = make_rect(size=(40, 20)).outer
    hole_left = make_circle(radius=3, center=(-14, 0)).outer
    hole_right = make_circle(radius=3, center=(14, 0)).outer

    profile = Profile2D(outer=outer, holes=[hole_left, hole_right])
    return linear_extrude(profile, height=4)
```

Notes:

- The profile is a closed path with holes.
- `linear_extrude()` triangulates the cap and builds the side faces.

## Step 2 - Relief Slot

```python
from impression.modeling import make_rect, boolean_difference, linear_extrude


def make_relief(base_mesh):
    slot_profile = make_rect(size=(22, 4))
    slot_mesh = linear_extrude(slot_profile, height=6).translate((0, 0, 1.5))
    return boolean_difference(base_mesh, [slot_mesh])
```

## Step 3 - Bridge Profiles

We loft between two rounded rectangles to form a smooth arch.

```python
from impression.modeling import make_rect, loft


def make_bridge():
    lower = make_rect(size=(26, 10))
    upper = make_rect(size=(18, 6))

    profiles = [lower, upper]

    # Loft along a simple vertical path
    path = [(0, 0, 6), (0, 0, 16)]
    return loft(profiles, path=path, cap_ends=False)
```

Constraints to keep in mind:

- Loft requires profiles with the same hole count.
- The path is resampled to match the number of profiles.

## Step 4 - Assemble

```python
from impression.modeling import boolean_union


def build():
    base = make_base()
    base = make_relief(base)
    bridge = make_bridge()
    return boolean_union([base, bridge])
```

## Full Script (Save as `examples/serious_clamp.py`)

```python
from impression.modeling import (
    make_rect,
    make_circle,
    boolean_difference,
    boolean_union,
    linear_extrude,
    loft,
)
from impression.modeling.drawing2d import Profile2D


def make_base():
    outer = make_rect(size=(40, 20)).outer
    hole_left = make_circle(radius=3, center=(-14, 0)).outer
    hole_right = make_circle(radius=3, center=(14, 0)).outer
    profile = Profile2D(outer=outer, holes=[hole_left, hole_right])
    return linear_extrude(profile, height=4)


def make_relief(base_mesh):
    slot_profile = make_rect(size=(22, 4))
    slot_mesh = linear_extrude(slot_profile, height=6).translate((0, 0, 1.5))
    return boolean_difference(base_mesh, [slot_mesh])


def make_bridge():
    lower = make_rect(size=(26, 10))
    upper = make_rect(size=(18, 6))
    path = [(0, 0, 6), (0, 0, 16)]
    return loft([lower, upper], path=path, cap_ends=False)


def build():
    base = make_base()
    base = make_relief(base)
    bridge = make_bridge()
    return boolean_union([base, bridge])
```

## Preview

```bash
impression preview examples/serious_clamp.py
```

## Export

```bash
impression export examples/serious_clamp.py --output dist/serious_clamp.stl --overwrite
```

## Tips for Real Projects

- Keep each step as a separate function. It makes debugging and iteration easier.
- Use `Mesh.analysis` to check for non-manifold edges and boundary edges.
- If a CSG result looks wrong, preview the parts in isolation and verify their
  relative positions.
- Favor explicit dimensions. It makes prints repeatable and enables parameterization.
