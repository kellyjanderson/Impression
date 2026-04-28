# Modeling — B-Spline Curves

Impression now has first-class B-spline curve primitives for authored 2D and
3D curve ownership.

```python
from impression.modeling import BSpline2D, BSpline3D
```

## Posture

These primitives own authored curve truth:

- control points
- degree
- knot vector
- explicit closure policy

They do not own fitting policy. Parameterization and knot-selection decisions
for inferred or fitted curves belong to the fit-policy lane, not to the
primitive object itself.

## BSpline2D

Use `BSpline2D(...)` when the curve is authored in a planar 2D domain.

```python
curve = BSpline2D(
    control_points=[(0.0, 0.0), (0.3, 0.8), (0.8, 0.2), (1.0, 0.0)],
    degree=3,
    knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    closure="open",
)
```

## BSpline3D

Use `BSpline3D(...)` when the authored curve lives directly in 3D space.

```python
curve = BSpline3D(
    control_points=[
        (0.0, 0.0, 0.0),
        (0.2, 0.3, 0.5),
        (0.8, -0.1, 1.0),
        (1.0, 0.0, 1.2),
    ],
    degree=3,
    knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    closure="periodic",
)
```

## Closure Policy

Closure is explicit and must be one of:

- `"open"`
- `"closed"`
- `"periodic"`

Impression does not infer closure from repeated endpoints alone.

## Current Scope

This initial primitive layer is about durable authored ownership.

Evaluation, derivative access, and deterministic sampling are defined in the
next B-spline execution layer.
