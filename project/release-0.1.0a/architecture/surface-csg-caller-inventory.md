# Surface CSG Caller Inventory

This inventory records authored routes that depend on surface CSG readiness.
The source of truth for executable checks is `surface_csg_caller_inventory()`.

## Caller Classes

- Public CSG API: `csg.boolean_union`, `csg.boolean_difference`, and
  `csg.boolean_intersection` route `backend="surface"` through
  `surface_boolean_result`.
- Feature builders: Impression-owned feature builders, such as hinges, are
  expected to call the shared CSG gate before invoking surface booleans.
- Primitive builders: boolean-dependent primitive helpers must use surface CSG
  for authored surface output or emit an explicit diagnostic.

## Rules

- Hidden mesh fallback is forbidden for authored surface routes.
- Mesh output is valid only when a caller selects an explicit mesh compatibility
  route such as `backend="mesh"` or a named mesh compatibility result.
- Unsupported surface CSG returns a diagnostic result or raises a surface
  eligibility error; it must not produce a mesh as substitute geometry.

## Shared Helpers

- `surface_csg_feature_gate(...)` returns a stable
  `SurfaceCSGFeatureGateDiagnostic` for support checks.
- `assert_no_hidden_surface_csg_mesh_fallback(...)` fails any authored surface
  route that produces `Mesh` or `MeshGroup`.

## Change History

- 2026-05-27: Limited the inventory to Impression-owned CSG callers.
