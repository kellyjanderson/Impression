# Modeling — Morph

Morphing blends two 2D profiles into a new profile using a parameter `t`.
Profiles must have the same number of holes (topology match).

```python
from impression.modeling import morph
from impression.modeling.drawing2d import make_circle, make_ngon
```

## morph(profile_a, profile_b, t)

- `t=0.0` → first profile
- `t=1.0` → second profile

```python
from impression.modeling import morph
from impression.modeling.drawing2d import make_circle, make_ngon

def build():
    a = make_circle(radius=1.0)
    b = make_ngon(sides=6, radius=1.0)
    return morph(a, b, t=0.5)
```

Example: `docs/examples/morph/morph_example.py`
