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
