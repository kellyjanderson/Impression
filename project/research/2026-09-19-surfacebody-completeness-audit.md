# SurfaceBody Completeness Audit

Date: 2026-09-19  
Repository baseline reviewed: `main@ad25e6cefa1df1617b7c20c6ab3ecd3545afbabf`  
Prior review commit: `255631a4a60bbea8a069f87be699cc68b9b7f274`

## Purpose

This document audits `SurfaceBody` as a modeling-kernel construct.

The question is not whether Impression has classes named `SurfaceBody`, `SurfaceShell`, and `SurfacePatch`. It does.

The question is:

> Does the current surface model contain enough geometry and topology to behave as a complete solid/surface representation on which loft, trimming, CSG, tessellation, and later modeling operations can rely without representation-specific escape routes?

The answer at the reviewed baseline is **no**.

The existing code is useful scaffolding, and many advanced patch evaluators now exist, but the body/shell/boundary layer is still materially incomplete. Too much solid behavior is implemented in operation-specific modules, especially `csg.py`, rather than being supported by a complete common surface kernel.

---

# 1. What exists today

## 1.1 Patch families

`src/impression/modeling/surface.py` currently defines these concrete patch families:

- `PlanarSurfacePatch`
- `HeightmapSurfacePatch`
- `DisplacementSurfacePatch`
- `RuledSurfacePatch`
- `RevolutionSurfacePatch`
- `BSplineSurfacePatch`
- `NURBSSurfacePatch`
- `SweepSurfacePatch`
- `SubdivisionSurfacePatch`
- `ImplicitSurfacePatch`

This is substantially broader than the original surface-first scaffold.

The patch-family capability matrix marks all of these as `available` for storage/evaluation-related purposes and advertises CSG participation for most or all families.

That breadth makes the missing common-kernel functionality more important, not less important.

---

## 1.2 Current common `SurfacePatch` contract

The abstract base class currently guarantees:

- family identity;
- rectangular `ParameterDomain`;
- zero or more `TrimLoop` values;
- attached transform;
- metadata;
- `point_at(u, v)`;
- `derivatives_at(u, v)`;
- canonical geometry payload;
- derived normal/frame;
- sampling;
- approximate bounds;
- stable hash identity.

This is enough to **evaluate** a surface.

It is not enough to **operate on** arbitrary surfaces as B-rep faces.

That difference is the largest SurfaceBody gap.

---

## 1.3 Current trim representation

`TrimLoop` currently contains:

- an array of 2D UV points;
- category `outer` or `inner`;
- orientation normalization;
- domain validation.

A patch may have at most one outer trim loop.

This is useful for polygonal parameter-space clipping and tessellation.

It is not yet a complete trimmed-face boundary representation.

---

## 1.4 Current seam / adjacency representation

Current records include:

- `SurfaceBoundaryRef(patch_index, boundary_id)`
- `SurfaceAdjacencyRecord(source, target, seam_id, continuity)`
- `SurfaceSeam(seam_id, boundaries, continuity, metadata)`

A seam has one or two boundary refs.

`SurfaceShell` stores:

- patches;
- a `connected` boolean;
- seams;
- adjacency records;
- transform;
- metadata.

The shell constructor validates reference ranges and calls the boundary sampler for referenced patch boundaries.

This establishes references, but it does not yet establish full shared-boundary truth.

---

## 1.5 Current `SurfaceBody`

`SurfaceBody` currently stores:

- one or more shells;
- attached transform;
- metadata;
- stable identity;
- patch iteration;
- approximate bounds.

That is essentially a **container**.

It does not itself expose the solid/topological operations expected from the canonical modeling representation.

---

# 2. The central mismatch

The intended architecture says:

> `SurfaceBody` is the canonical 3D modeling representation.

The implemented class currently behaves more like:

> immutable storage for a collection of surface shells.

The difference is visible in where behavior lives.

For example, body classification, Boolean eligibility, containment, shell reconstruction, cut-fragment logic, validity gates, and many topology decisions live in `csg.py`.

That is backwards for a reusable surface kernel.

CSG should be a client of common surface-body laws, not the only place those laws exist.

---

# 3. Missing boundary topology constructs

## 3.1 Missing oriented boundary-use / coedge construct

The repository's own earlier architecture identified this correctly, but the runtime model still lacks it.

A shared edge/seam needs **oriented uses** by each participating patch.

Current `SurfaceBoundaryRef` only says:

```text
patch index + boundary id
```

It does not encode:

- traversal orientation;
- loop membership;
- trim/p-curve identity;
- which side of the patch is material;
- exact per-face use of a shared 3D edge.

### Required construct

Conceptually:

```python
PatchBoundaryUse(
    id,
    patch_ref,
    seam_ref,
    loop_ref,
    trim_curve_ref,
    orientation,
    loop_role,
)
```

Without this, seam assembly, shell orientation, trimming, and Boolean reconstruction remain under-specified.

---

## 3.2 Missing explicit boundary-loop object

A patch currently owns a flat tuple of `TrimLoop` values.

A complete face topology needs explicit loop identity and ordered boundary uses.

Required concept:

```text
SurfaceBoundaryLoop
    id
    role: outer | inner
    ordered PatchBoundaryUses
```

This is important for:

- exact holes;
- Boolean-created trim loops;
- seam traversal;
- orientation;
- shell validation;
- persistent identity after splitting.

A UV polygon alone is not enough durable topology.

---

## 3.3 Seam lacks canonical shared 3D geometry

Current `SurfaceSeam` records participating boundary refs and continuity.

It does not own a canonical shared curve.

Therefore two adjacent faces can conceptually refer to "the same seam" without the seam itself containing the geometry that proves what that shared boundary is.

### Required

A seam needs a canonical 3D boundary representation, for example:

```text
SurfaceSeam
    id
    shared_curve_3d
    boundary_uses[]
    classification
    continuity
```

Each boundary use needs a patch-local 2D p-curve/trim mapping corresponding to that shared 3D curve.

That provides one source of geometric boundary truth.

---

## 3.4 Duplicate seam and adjacency truth

`SurfaceShell` stores both:

- `seams`;
- `adjacency`.

This permits two topological truths unless one is strictly derived from the other.

Current construction validates references but does not enforce that adjacency is exactly the projection of seam ownership.

### Recommendation

Make seams + oriented boundary uses canonical.

Derive adjacency.

Do not persist both as independent authoritative kernel state.

---

# 4. Missing trim geometry

## 4.1 Trim loops are sampled polylines only

`TrimLoop.points_uv` is an array of points.

That means a Boolean intersection curve on a B-spline, NURBS, revolution, sweep, or implicit-derived face is forced toward a sampled representation unless some separate route preserves more information.

For a general surface kernel, trim boundaries need actual curve objects.

### Required

At minimum a common 2D curve protocol supporting:

- line;
- polyline;
- circular/conic arc where applicable;
- B-spline/NURBS curve;
- sampled curve with declared tolerance.

A trim loop should reference ordered curve segments, not only a polygon.

---

## 4.2 Missing p-curve / 3D-curve pairing

A robust trimmed face needs both:

- patch-local parameter-space boundary curve;
- corresponding shared/world 3D curve.

This is especially important after CSG.

The kernel must be able to verify:

```text
patch(point_on_pcurve) ~= shared_3d_curve(point)
```

within tolerance.

The current `TrimLoop` representation does not express that contract.

---

## 4.3 No durable trim-segment identity

Boolean operations repeatedly split existing boundaries and create new ones.

A flat point array cannot preserve:

- which segment was inherited;
- which segment was generated by intersection;
- which seam owns it;
- orientation lineage;
- provenance across subsequent CSG operations.

This missing identity contributes directly to CSG re-entry complexity.

---

# 5. Missing common patch operations

The current common patch interface is evaluation-centric.

To make all patch families interoperable under the same modeling operations, every patch family needs either native support or a declared adapter for the following common kernel operations.

## 5.1 Parameter inversion / projection

Missing from the base contract:

```text
project_point(point3d) -> candidate (u,v) + residual
closest_parameter(point3d)
invert_curve(curve3d) -> pcurve
```

CSG and seam reconstruction need this routinely.

Today these abilities are implemented piecemeal in operation-specific code.

---

## 5.2 Boundary evaluation

There is internal boundary sampling logic, but not a strong common patch contract that returns durable boundary geometry.

Required:

```text
boundary_ids()
boundary_curve(boundary_id)
boundary_parameterization(boundary_id)
```

Trim-created boundaries must participate in the same model as native domain boundaries.

---

## 5.3 Patch splitting

There is no universal:

```text
split(trim_curves) -> SurfaceFragments
```

or equivalent common-kernel operation on `SurfacePatch`.

This is one of the biggest CSG blockers.

The generic Boolean algorithm should discover intersection curves, map them to both faces, and ask the surface kernel to split faces. It should not need a separate patch-family Boolean implementation for every pair.

---

## 5.4 Subdomain extraction

A complete parametric patch API should support extracting or representing a bounded subdomain without losing family identity where possible.

Required concept:

```text
subpatch(domain, trims) -> SurfacePatch
```

For many Boolean fragments, the underlying analytic/parametric geometry does not change; only the valid domain changes.

---

## 5.5 Higher derivatives / curvature

Only first derivatives are guaranteed by the base contract.

General surface quality, continuity, adaptive intersection, fairing, and adaptive tessellation need optional or common access to:

- second derivatives;
- principal curvature or curvature estimates;
- singularity diagnostics.

Not every family must provide exact values, but the capability needs a common contract.

---

## 5.6 Periodicity and singular boundaries

The common patch contract does not make periodicity/singularity first-class.

These matter for:

- cylinders;
- revolutions;
- spheres;
- closed sweeps;
- NURBS surfaces;
- trim wrapping;
- CSG intersection curves crossing parameter seams.

Required patch topology metadata includes:

- U periodicity;
- V periodicity;
- singular boundaries/poles;
- canonical parameter seam handling.

Without this, generic trimming remains fragile.

---

## 5.7 Exact/conservative bounds contract

`bounds_estimate()` samples a small grid by default.

That is not a sufficiently strong kernel contract for:

- Boolean broad-phase rejection;
- containment;
- intersection dispatch;
- correctness-sensitive pruning.

Each patch family needs either:

- exact bounds;
- mathematically conservative bounds;
- or an explicit bound-quality classification.

A sampled estimate must never be mistaken for a guaranteed enclosure.

---

# 6. Missing shell validity law

## 6.1 `connected` is stored, not proven

`SurfaceShell.connected` is currently a boolean field supplied to the constructor.

A canonical kernel should derive connectivity from topology.

Storing it as truth permits disagreement with actual seams/boundary uses.

### Recommendation

Connectivity should be computed/validated.

If cached, it must be derived state.

---

## 6.2 No complete manifold-edge validation

A closed orientable 2-manifold shell normally requires each shared edge to have exactly two compatible oriented face uses, except for explicitly open surface bodies.

Current seam records permit one or two boundaries, but the shell does not establish the stronger manifold law.

Required validity checks include:

- every closed-shell boundary use belongs to exactly one shared seam;
- every interior seam has exactly two uses;
- uses have opposite compatible traversal;
- no non-manifold edge has 3+ face uses unless Impression deliberately supports non-manifold bodies;
- no dangling boundary exists on a closed shell.

---

## 6.3 No shell orientation / inside-outside invariant

A solid shell requires coherent orientation.

The current model lacks a strong common notion of:

- outward face orientation;
- reversed face use;
- shell signed orientation;
- void-shell orientation.

This is currently reconstructed or reasoned about in operation-specific code.

It belongs in SurfaceBody law.

---

## 6.4 No canonical open/closed validity report on `SurfaceShell`

CSG has body-classification and validity functions, but `SurfaceShell` itself does not provide a first-class:

```text
validate_topology()
classification -> open | closed | invalid
```

That causes downstream consumers to invent their own validity rules.

---

## 6.5 No self-intersection validity law

A shell can be topologically closed and still geometrically invalid because it self-intersects.

There is loft-specific self-intersection logic, but not a general SurfaceBody validity contract.

Required:

- local face self-intersection checks where applicable;
- non-adjacent face intersection checks;
- tolerance-aware touching/coincidence classification;
- explicit validity result.

---

# 7. Missing multi-shell solid semantics

`SurfaceBody` allows multiple shells.

But the common model does not define what multiple shells mean.

Possible meanings include:

- disconnected solid components;
- outer shell + cavities;
- nested material/island shells;
- open sheet groups.

These are not equivalent.

## Required shell role / nesting model

The body needs enough information to classify:

- exterior material shell;
- cavity/void shell;
- disconnected component shell;
- open sheet shell, if open bodies remain allowed.

Alternatively shell nesting can be derived from containment plus orientation, but the result must be explicit and validated.

This is essential for CSG.

---

# 8. Current CSG contradicts multi-shell SurfaceBody capability

`_canonicalize_surface_boolean_body()` currently rejects bodies unless:

```text
body.shell_count == 1
```

with narrow special handling elsewhere.

That means `SurfaceBody` advertises one-or-more-shell storage, but the canonical Boolean path does not accept the general body model.

This is a kernel completeness gap, not merely a CSG feature gap.

A complete solid representation must permit CSG over arbitrary valid shell sets.

---

# 9. Missing body-level geometric queries

A canonical solid body should expose common queries independent of which operation needs them.

Current `SurfaceBody` lacks first-class body methods/services for:

## 9.1 Point classification

Required:

```text
classify_point(p) -> inside | outside | boundary
```

This is fundamental to:

- CSG fragment selection;
- containment;
- shell nesting;
- user queries;
- later modeling operations.

Today related logic is fragmented through CSG routes and primitive-specific tests.

---

## 9.2 Body/body relation

Required common query:

```text
relation(other) ->
    disjoint
    touching
    overlap
    containment
    equal
```

Current CSG has relation classifications, but this capability belongs below CSG.

---

## 9.3 Ray/surface intersection

A generic body query layer should support ray intersections against arbitrary patches, used by:

- point classification;
- picking;
- containment;
- diagnostics.

---

## 9.4 Closest point / distance

Useful common operation:

```text
closest_point(p)
distance_to(p)
```

This also helps robust classification near tolerance boundaries.

---

# 10. Missing body mutation/construction primitives

Impression uses immutable records, which is appropriate.

"Mutation" here means operations returning new valid surface objects.

The common kernel lacks generic operations such as:

- replace/split one patch while preserving shell topology;
- insert seam;
- merge coincident seams;
- rebuild boundary loops;
- orient shell;
- compose shells;
- extract connected shell components;
- remove unused fragments;
- normalize body topology;
- validate and return a healed bounded result.

Today CSG contains bespoke assembly code for many of these actions.

These should become reusable SurfaceBody/kernel utilities.

---

# 11. Missing fragment abstraction

A Boolean should not immediately have to turn every split piece into an ad hoc route-specific record.

A common intermediate construct is needed:

```text
SurfaceFragment
    source_patch
    underlying surface geometry
    bounded parameter region
    boundary loops
    generated/inherited boundary uses
    orientation
    provenance
```

This is the common unit needed by:

- CSG;
- trimming;
- splitting;
- clipping;
- shell repair;
- future fillets/chamfers.

Current CSG defines several fragment records internally. Their existence is evidence that the surface kernel is missing the shared abstraction.

---

# 12. Missing generic intersection contract at the SurfaceBody boundary

The repository now has a substantial `surface_intersections.py` registry with exact, declared-tolerance, and adapter routes.

This is progress.

But the capability is still not fully integrated as a common face/body service.

A complete kernel needs one normalized operation:

```text
intersect_surfaces(face_a, face_b, tolerance)
    -> 3D intersection curves
    -> p-curves on face_a
    -> p-curves on face_b
    -> degeneracy/contact records
```

Every supported patch family must enter this same result model.

Whether the solver underneath is analytic, iterative, subdivision-based, sampled, or implicit is an implementation detail.

---

# 13. Missing generic reconstruction contract

After splitting and classifying fragments, the kernel needs reusable reconstruction:

```text
fragments
    -> boundary-use graph
    -> seams
    -> loops
    -> oriented shells
    -> shell nesting
    -> SurfaceBody
    -> validity report
```

Today much of this lives in `csg.py` under names such as shell assembly, cap construction, orientation, seam-use pairing, and validity gates.

That logic should be factored toward the SurfaceBody kernel because CSG is not the only future operation that needs it.

---

# 14. Empty-body representation is external

`SurfaceBody` requires at least one shell.

Boolean results represent emptiness through result status/classification rather than a `SurfaceBody` instance.

This is not necessarily wrong, but the kernel contract should make it deliberate.

Two viable models are:

1. **No empty SurfaceBody** — operations return a result wrapper with `body=None`.
2. **Explicit EmptyBody/empty SurfaceBody** — emptiness is a first-class geometric value.

The current behavior should be formalized because generic composition otherwise needs repeated special cases.

---

# 15. Surface family capability matrix overstates common interoperability

The current capability matrix marks advanced patch families `available` and advertises `csg` capability broadly.

However, availability currently means very different things across families:

- evaluatable/storable;
- supported by one specialized CSG route;
- supported through promotion;
- supported only for some family pairs;
- supported only within a bounded sampling adapter;
- refused for other combinations.

A single `"csg"` operation flag is therefore too coarse.

## Recommendation

Separate capabilities such as:

```text
evaluate
project_point
inverse_map
exact_boundary
split_by_pcurve
intersection_exact
intersection_declared_tolerance
classify_fragment
trim_preserving
csg_full
```

A family should not claim `csg_full` until the generic Boolean pipeline can consume it against every promoted family under the defined tolerance policy.

---

# 16. Required common SurfacePatch protocol

The eventual common patch protocol should contain or provide adapters for at least:

```text
identity / family
parameter domain
periodicity
singularities

point_at
first_derivatives
optional second_derivatives

guaranteed/conservative bounds

project_point / inverse parameter query

native boundary curves
trim loops as curve loops
p-curves

3D <-> 2D boundary consistency

split by parameter curves
bounded subpatch / fragment creation

transform

serialization
provenance
```

Patch families may use different algorithms, but they must satisfy one semantic contract.

---

# 17. Required SurfaceShell protocol

A complete shell should own/derive:

```text
patches
boundary loops
patch boundary uses
shared seams
orientation
connectivity
open/closed status
manifold status
self-intersection validity
bounds
topological identity
```

And expose:

```text
validate()
open_boundaries()
connected_components()
orient()
classify_point() or support body classifier
```

---

# 18. Required SurfaceBody protocol

A canonical SurfaceBody should support:

```text
shells
shell nesting/material roles
transforms
stable identity

validate()
classification
bounds (guaranteed)

classify_point()
relation(other)
intersections(other)

connected solid components
cavities

generic fragment reconstruction
```

It should not require clients to inspect metadata to understand whether it is a valid closed solid.

---

# 19. Non-CSG missing constructs/functions inventory

The following are missing or incomplete even before discussing Boolean execution:

### Boundary/topology

- oriented patch boundary use/coedge;
- explicit surface boundary loop;
- seam-owned canonical 3D curve;
- patch-local p-curve reference;
- boundary-use orientation;
- loop identity;
- derived-only adjacency law;
- manifold-edge validation;
- shell orientation;
- shell nesting / cavity semantics.

### Trim geometry

- curve-based trim segments;
- exact/spline p-curves;
- durable trim-segment identity;
- trim/3D-edge consistency checks;
- generic trim splitting.

### Patch geometry

- common inverse mapping;
- common closest-point projection;
- guaranteed bounds;
- explicit periodicity;
- explicit singularities;
- common subpatch extraction;
- common patch splitting;
- optional common curvature/higher derivative access.

### Shell/body geometry

- general point classification;
- body relation query;
- general containment;
- generic ray intersection;
- closest-point/distance;
- general self-intersection validation.

### Reconstruction

- reusable SurfaceFragment;
- generic boundary graph reconstruction;
- generic shell assembler;
- generic seam rebuilder;
- shell orientation repair;
- connected-component extraction;
- cavity/nesting classifier;
- generic validity/healing result.

### Capability law

- fine-grained capability declarations;
- one normalized tolerance policy shared across all common operations;
- clear exact vs declared-tolerance vs sampled behavior at the common interface.

---

# 20. SurfaceBody should own laws, not necessarily all algorithms

Making SurfaceBody complete does **not** mean putting thousands of lines of algorithms inside the `SurfaceBody` class.

A good architecture can use kernel services/modules:

```text
surface/
    geometry
    intersections
    trimming
    topology
    classification
    reconstruction
    validity
```

The critical point is that these services operate on the common SurfaceBody/SurfacePatch abstractions and are reusable by all modeling operations.

The current 339 KB `surface.py` and 843 KB `csg.py` indicate responsibilities need decomposition as the contracts are completed.

---

# 21. Relationship to loft

The loft review proposed:

```text
3DBSpline
  -> Stations
  -> explicit topology graph
  -> transition cells
  -> surface patches
  -> SurfaceBody
```

For that to remain clean, loft should stop after producing:

- bounded patches;
- boundary-use intent;
- caps;
- explicit shared-boundary relationships.

Then the SurfaceBody kernel should perform:

- seam realization;
- orientation;
- shell assembly;
- closure validation;
- general body validity.

Loft should not own a private B-rep law.

---

# 22. Relationship to CSG

The desired CSG property is:

> No Boolean fails merely because the operands are made from different patch families.

That requirement cannot be solved sustainably by adding more pair-specific routes to `csg.py`.

It requires the SurfaceBody common protocol described above.

A patch family may use a different solver internally, but all must be reducible to the same Boolean primitives:

1. intersect;
2. map intersection curves to both parameter spaces;
3. split faces;
4. classify fragments;
5. select fragments by Boolean operation;
6. construct generated boundaries/caps where necessary;
7. assemble seams/loops/shells;
8. validate.

That is the bridge between this audit and the separate CSG audit.

---

# 23. Recommended architectural posture

The original surface-first direction was correct.

The implementation should now finish the promise of that direction:

> `SurfaceBody` must become a real surface/B-rep kernel abstraction, not just a container around increasingly capable patch evaluators.

The next repair plan should therefore prioritize **common topology and face operations** over adding more special-case feature code.

---

# 24. Non-negotiable completion criteria

A future plan should not mark SurfaceBody complete until all of these are true:

1. Shared boundaries have one canonical seam truth.
2. Every patch boundary participates through an oriented boundary use.
3. Boundary loops are explicit topology.
4. Trims can preserve non-polyline intersection curves.
5. Every patch family provides/adapts inverse mapping required by trimming.
6. Every patch family can be split into bounded fragments through the common API.
7. Bounds used for correctness are guaranteed conservative.
8. Periodic and singular surfaces have explicit parameter-topology rules.
9. Shell connectivity is derived, not trusted from a boolean field.
10. Closed-shell manifold validity is computed.
11. Shell orientation is computed/validated.
12. Multiple shells have defined material/cavity/component semantics.
13. Point classification works for arbitrary valid bodies.
14. Generic fragment reconstruction can produce a valid SurfaceBody.
15. The same reconstruction code is reusable by CSG, trimming, and future modeling operations.
16. SurfaceBody validity does not depend on tessellating to a mesh.
17. Mesh repair is never required to turn an invalid surface topology into a valid modeled solid.
18. Advanced patch-family support is expressed through common capabilities, not feature-specific metadata assumptions.
