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

## Evaluation and Sampling

Both primitives now provide:

- `evaluate(t)` for deterministic point evaluation
- `derivative(t)` for first-derivative access
- `tangent(t)` for normalized tangent access
- `sample(n_samples=...)` for deterministic ordered sampling

Closure handling is explicit:

- `"open"` samples the authored parameter range directly
- `"closed"` closes sampled output without guessing closure from repeated
  endpoints alone
- `"periodic"` wraps parameter evaluation through the authored domain

## Fit Policy Records

Fit-backed workflows should keep parameterization policy explicit rather than
burying it in helper behavior.

```python
from impression.modeling import ParameterizationPolicyRecord

policy = ParameterizationPolicyRecord(
    method="chord_length",
    domain_start=0.0,
    domain_end=1.0,
)
```

Current initial scope:

- `"uniform"`
- `"chord_length"`
- `"centripetal"`

These policy records assign replayable parameter values to ordered evidence
before later knot and fit-configuration stages run.

Knot policy is also explicit:

```python
from impression.modeling import KnotCountPolicyRecord, KnotPlacementPolicyRecord

count_policy = KnotCountPolicyRecord(strategy="fixed", control_point_count=6)
placement_policy = KnotPlacementPolicyRecord(placement_method="average_parameter")
```

Initial knot-policy scope:

- fixed control-point count selection
- uniform internal knot placement
- average-parameter knot placement
