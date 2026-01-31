# Modeling — Loft

Loft creates a surface between a series of profiles. Profiles must share the
same hole topology. If a path is provided, profiles are translated to sampled
positions and rotated to follow the path direction (parallel-transport frames
to minimize twist).

```python
from impression.modeling import loft
from impression.modeling.drawing2d import make_rect
from impression.modeling import Path3D
```

## loft(profiles, path=None)

```python
from impression.modeling import loft, Path3D, Line3D
from impression.modeling.drawing2d import make_rect

def build():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.6, 1.4)),
        make_rect(size=(0.8, 0.8)),
    ]
    path = Path3D.from_points([(0, 0, 0), (0, 0, 1.0), (0, 0, 2.0)])
    return loft(profiles, path=path, cap_ends=True)
```

Example: `docs/examples/loft/loft_example.py`

## End Caps

Use `start_cap` and `end_cap` to round or taper the ends of a loft:

- `none` (default): no extra cap geometry
- `flat`: cap with the base profile (same as `cap_ends=True`)
- `taper`: linearly shrink the profile to a tip
- `dome`: half‑circle profile (true dome)
- `slope`: steeper start, gentler finish (inverse‑dome)

If either `start_cap` or `end_cap` is not `none`, the loft is automatically
closed at both ends. `cap_ends=True` remains as a backward‑compatible shortcut
for a flat cap.

Cap length is additive by default: the cap extends beyond the path endpoints.
Use `start_cap_length` / `end_cap_length` (in model units) to control how far
the cap blends. The blend eases to the last **non‑degenerate** profile over the
specified length. Use `cap_scale_dims` to control which axes scale:

- `both` (default): uniform scale in X/Y until the smallest dimension collapses
- `smallest`: scale only the limiting dimension

Caps use eased profiles (linear, sine, or quadratic) rather than pure linear ramps.
If no length is provided, `cap_steps * path_step` is used.

```python
from impression.modeling import loft
from impression.modeling.drawing2d import make_rect

def build():
    profiles = [make_rect(size=(1.0, 1.0))] * 5
    return loft(
        profiles,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=2.0,
        end_cap_length=3.0,
        cap_scale_dims="both",
    )
```
