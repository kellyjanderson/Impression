# Patchwise Loft Repair Structure

## Topic

Research on the local structure and seam model needed for future patchwise loft
repair.

## Findings

### Patchwise repair needs local boundary truth, not just nearby sample points

A patchwise repair result cannot be defined from arbitrary nearby geometry
alone. The minimum useful local structure is likely:

- the local boundary loops or open boundaries around the damaged area
- enough neighboring patch context to recover tangent / normal intent
- local seam or adjacency anchors
- a bounded target region to replace

That is more than a point cloud and less than a full recovered body topology.

### The `SurfaceBody` seam and adjacency model is the right structural target

The current surfaced kernel already has first-class structure for:

- `SurfaceSeam`
- `SurfaceAdjacencyRecord`
- `SurfaceShell`
- `SurfaceBody`

That means patchwise repair should not invent an unrelated local connectivity
model. It should aim to emit repaired local structure that can be expressed in
the same seam/adjoining-boundary language.

This is important because the repaired result eventually needs to integrate back
into a surfaced shell, not remain a disconnected local approximation.

### Patchwise repair is probably a mode of broader surface reconstruction, not a standalone primitive

The idea reads most cleanly as:

- one mode of a broader surface reconstruction system

rather than:

- one isolated patchwise-loft primitive with its own incompatible data model

Why:

- it depends on local boundary recovery
- it depends on seam integration
- it likely needs multiple patch families over time
- it overlaps conceptually with band-style loft repair and future span repair

So the better framing is:

- patchwise repair is a local reconstruction mode
- loft-like surfaces are one reconstruction strategy inside that mode

### The first useful patchwise lane should probably target simple local patch families

The current required surface families are:

- planar
- ruled
- revolution

That suggests the first patchwise experiments should focus on local repairs
that can be expressed as:

- planar replacement
- ruled bridging
- simple revolved-like local spans

This is much more realistic than trying to make the first patchwise lane solve
general freeform local patch fitting.

### Patchwise repair should prefer local exactness over large implicit reinterpretation

Patchwise repair is valuable because it can stay local. That advantage is lost
if the tool immediately becomes a broad heuristic refitter.

So the early acceptance posture should be:

- local boundaries are explicit
- local seam intent is explicit
- repaired patch set integrates into the surrounding shell model
- approximation is reported if exact local fit is not possible

## Implications

Recommended first patchwise-repair contract:

- recover a bounded local patch neighborhood
- express repaired boundaries and seams in the existing `SurfaceBody` model
- treat patchwise repair as a mode of future surface reconstruction tooling
- begin with simple local patch families and explicit diagnostics

This gives the idea a realistic bridge from exploratory future feature to
eventual surfaced implementation.

## References

- `project/future-features.md`
- `src/impression/modeling/surface.py`
- `src/impression/modeling/loft.py`
- `tests/test_surface_kernel.py`
- `docs/modeling/mesh-tools.md`
