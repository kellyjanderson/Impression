# Advanced Surface Patch Families

Advanced patch families are surface-native authoring records. They stay as
`SurfaceBody` or `SurfacePatch` data until an explicit tessellation call such
as `tessellate_surface_body(...)` is made for preview, export, or analysis.

Use these routes when primitives or ruled lofts are not the natural authored
shape.

## Public API Inventory

- `BSplineSurfacePatch` for non-rational tensor-product spline surfaces.
- `NURBSSurfacePatch` for explicitly rational tensor-product surfaces.
- `SweepSurfacePatch` for path/profile surfaces.
- `SubdivisionSurfacePatch` plus `make_subdivision_surface(...)` for authored
  control-cage surfaces.
- `ImplicitSurfacePatch`, `ImplicitFieldNode`, `make_implicit_field_node(...)`,
  and `make_implicit_surface(...)` for safe declarative implicit fields.
- `HeightmapSurfacePatch` and `DisplacementSurfacePatch` for sampled height and
  displaced-source surfaces.
- `loft_plan_sections(...)` and `loft_execute_plan(...)` for loft output that
  selects ruled, B-spline, NURBS, or sweep patch families from explicit intent.

## Loft Output Families

Simple compatible two-station loft transitions stay `ruled` by default. Ask
for richer output only when the authored model carries that intent.

```python
from impression.modeling import Station, as_section, loft_execute_plan, loft_plan_sections
from impression.modeling.drawing2d import make_rect

stations = [
    Station(t=0.0, section=as_section(make_rect(size=(1.0, 1.0))), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    Station(t=1.0, section=as_section(make_rect(size=(0.7, 1.3))), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
]

plan = loft_plan_sections(stations, samples=16, smooth_intent=True)
body = loft_execute_plan(plan)
```

`smooth_intent=True` emits `BSplineSurfacePatch` sidewalls. NURBS output
requires explicit rational intent and explicit positive finite weights:

```python
plan = loft_plan_sections(
    stations,
    samples=16,
    rational_intent=True,
    rational_weights=1.25,
)
body = loft_execute_plan(plan)
```

Sweep output requires an explicit `Path3D` guide and records path/profile
references:

```python
from impression.modeling import Path3D

guide = Path3D.from_points([(0, 0, 0), (0.2, 0, 0.5), (0, 0, 1)])
plan = loft_plan_sections(stations, samples=16, sweep_path=guide)
body = loft_execute_plan(plan)
```

Unsupported loft families, missing NURBS weights, and missing sweep paths
refuse during planning or surface execution. They do not fall back to mesh
geometry.

## Subdivision Authoring

Use `make_subdivision_surface(...)` when the authored truth is a control cage.
The helper wraps one `SubdivisionSurfacePatch` in a `SurfaceBody`; the patch
still owns cage, face, crease, and subdivision-level validation.

```python
from impression.modeling import make_subdivision_surface

body = make_subdivision_surface(
    control_points=[
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ],
    faces=((0, 1, 2, 3),),
    creases=({"edge": (0, 1), "sharpness": 2.0},),
    subdivision_level=1,
)
```

Subdivision tessellation is approximate by definition and records the
approximation boundary in tessellation metadata.

## Implicit Authoring

Implicit authoring is limited to safe declarative field nodes. Arbitrary
expression evaluation is not accepted.

```python
from impression.modeling import make_implicit_field_node, make_implicit_surface

field = make_implicit_field_node("sphere", parameters={"radius": 0.75})
body = make_implicit_surface(
    field=field,
    bounds=(-1, 1, -1, 1, -1, 1),
)
```

Bounds must be finite and positive in all three axes. Unsafe field payloads
and unbounded tessellation requests refuse before mesh creation.

## Sampled Surfaces

Use `HeightmapSurfacePatch` for native sampled height surfaces and
`DisplacementSurfacePatch` when an authored source surface is displaced by a
sample grid. Both patch families persist through `.impress` as surface-native
payloads; mesh or triangle wrapper payloads are rejected.

## Persistence And Tessellation

The `.impress` format stores advanced patch family payloads directly. It
rejects unknown families, malformed family payloads, unsafe implicit fields,
and mesh-derived wrapper data.

Tessellation is the boundary where meshes are produced:

```python
from impression.modeling import preview_tessellation_request, tessellate_surface_body

mesh = tessellate_surface_body(body, preview_tessellation_request()).mesh
```

Approximate families record lossiness or approximation metadata in the mesh
result; authored `SurfaceBody` state remains surface-native.
