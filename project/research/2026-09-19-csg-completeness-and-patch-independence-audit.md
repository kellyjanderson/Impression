# CSG Completeness and Patch-Independence Audit

Date: 2026-09-19  
Repository baseline reviewed: `main@ad25e6cefa1df1617b7c20c6ab3ecd3545afbabf`  
Related review commits:
- loft topology audit: `255631a4a60bbea8a069f87be699cc68b9b7f274`
- SurfaceBody completeness audit: `ea45f681132e6d4b0b4912d963960ddd39c950b8`

## Purpose

This document audits surfaced CSG with one target requirement:

> For valid closed `SurfaceBody` operands, Boolean union, difference, and intersection must not fail merely because the bodies are composed of different supported patch families.

This does **not** require every pair to use the same numerical algorithm internally.

It requires every patch family to satisfy a common Boolean-kernel contract so that operation semantics are independent of representation.

The reviewed code does not currently meet that requirement.

The strongest evidence is architectural: `src/impression/modeling/csg.py` is approximately 843 KB and contains a large collection of family-specific, primitive-specific, loft-specific, sampled-specific, and refusal-specific execution routes.

---

# 1. Current public API is already pointed in the right direction

Public Boolean functions accept only `SurfaceBody`:

- `boolean_union()`
- `boolean_difference()`
- `boolean_intersection()`

They explicitly reject `Mesh` and `MeshGroup` as modeling operands.

This is correct.

The repository also explicitly tries to prevent hidden mesh fallbacks.

That policy should remain.

The problem is below the public boundary: surfaced Boolean execution is still fragmented by representation.

---

# 2. Current canonicalization is narrower than SurfaceBody

`_canonicalize_surface_boolean_body()` currently requires:

- exactly one shell;
- connected shell;
- closed-valid classification;

with a narrow special case for primitive cylinders.

This means general valid `SurfaceBody` values are not Boolean operands.

## Missing support

A complete Boolean must support:

- multiple disconnected material components;
- outer shells with cavity shells;
- results that naturally become multiple shells;
- results that merge multiple shells into one;
- difference that introduces cavities;
- intersection that returns multiple disconnected pieces.

The one-shell gate is a fundamental blocker.

---

# 3. Family support is controlled by a pair matrix

The code maintains:

- `_SURFACE_BOOLEAN_EXECUTABLE_FAMILY_PAIRS`
- `SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX`
- `SURFACE_CSG_SOLVER_REGISTRY`
- route pair classes
- support states
- refusal diagnostics

This makes the architecture fundamentally **pair-oriented**.

## Why this is the wrong completion target

With N patch families, pair-specific support grows approximately with N², multiplied by three Boolean operations and contact/degeneracy cases.

That is exactly the combinatorial behavior visible in the current module.

The target architecture should instead ask:

> Can this patch satisfy the common intersection / inversion / splitting / fragment contract?

If yes, it can participate with any other patch satisfying that contract.

---

# 4. Current CSG dispatch is a ladder of special routes

`surface_boolean_result()` currently tries, in order, specialized routes including:

- B-spline/NURBS body route;
- sweep/subdivision body route;
- polygon-loft field route;
- family planning gate;
- branching loft difference;
- loft pair CSG;
- single-shell loft/primitive route;
- loft primitive trim-fragment route;
- trivial result;
- loft route selection / adapter;
- primitive implicit route;
- ruled unsupported-cutter diagnostic;
- box/analytic path;
- final generic unsupported result.

This is a strong signal that the kernel abstraction beneath CSG is incomplete.

A generic Boolean algorithm should not need to know whether a body came from:

- loft;
- primitive authoring;
- subdivision;
- sweep;
- heightmap;
- displacement.

Those are geometry representations/provenance, not Boolean semantics.

---

# 5. Explicit current failure: generic unsupported terminal

The final fallback in `surface_boolean_result()` is:

```text
Surface boolean <operation> execution is not implemented yet
after canonical input preparation.
```

That is the clearest direct evidence that the current support matrix is not execution-complete.

---

# 6. B-spline / NURBS body route is incomplete

The current B-spline/NURBS bridge:

- requires exactly two operands;
- searches for a rectangular participating patch overlap;
- can refuse if no such overlap exists;
- currently supports success only for **intersection**;
- reconstructs by replacing/trimming a source patch;
- preserves existing shell seams/adjacency.

## Missing functionality

General B-spline/NURBS CSG requires:

- arbitrary intersection curves, not only rectangular overlap;
- union;
- difference;
- multiple intersecting face pairs;
- multiple intersection loops;
- face fragmentation;
- generated boundary edges;
- shell reconstruction;
- multi-shell results;
- subsequent CSG re-entry.

The current route is evidence/prototype functionality, not complete Boolean support.

---

# 7. Sweep / subdivision body route is incomplete

The current sweep/subdivision body route:

- requires exactly two operands;
- finds one participating candidate pair;
- may rely on rectangular overlap;
- gathers pair evidence;
- currently supports success only for **intersection**;
- reconstructs largely by retaining one source patch/body.

## Missing functionality

- union;
- difference;
- arbitrary multi-face intersection;
- full contour splitting;
- fragment classification;
- fragment retention;
- shell assembly;
- new seams and loops;
- arbitrary subdivision/sweep combinations;
- multi-shell results.

Again, this is a route-specific bridge rather than complete CSG.

---

# 8. Sampled/implicit families are explicitly treated as a boundary/refusal class

The current route taxonomy puts:

- implicit;
- heightmap;
- displacement

into `SAMPLED_SURFACE_CSG_FAMILIES` and often classifies them as a sampled boundary/refusal route.

There are substantial specialized systems for:

- implicit composition;
- heightmap composition;
- displacement composition;
- promotion;
- representation refusal;
- sampling budgets;
- no-hidden-mesh proof.

These are useful capabilities, but they do not satisfy the target requirement.

## Required posture

A valid body containing one of these patch families must either:

1. expose a common surface intersection/splitting representation at declared tolerance, or
2. be canonically converted/promoted into another **surface representation** that does.

The Boolean operation should then proceed through the same fragment pipeline.

The caller should not receive "unsupported because this patch family is sampled" if the body is otherwise a supported modeled surface.

---

# 9. Implicit CSG composition is useful but not a universal replacement

Implicit field composition can represent many Boolean results elegantly.

However, making implicit composition a special CSG escape path creates two issues:

1. exact/parametric bodies may lose useful patch structure;
2. subsequent modeling operations may now see a representation with different capabilities.

## Recommended role

Implicit composition can be a valid internal solver/reconstruction strategy when chosen deliberately.

But it must produce a canonical `SurfaceBody` whose capabilities meet the same downstream contract.

It should not be a semantic exception to CSG.

---

# 10. Heightmap-preserving CSG is inherently representationally limited

A single-valued heightmap cannot represent arbitrary solid Boolean results.

The code correctly contains representability checks/refusals.

The error is not that heightmap-preserving CSG can refuse.

The error would be allowing that refusal to become the end of the **body Boolean**.

## Required behavior

If a Boolean result cannot remain a heightmap:

- promote/reconstruct it into a general patch representation;
- continue the Boolean;
- return a valid `SurfaceBody`.

"Cannot preserve heightmap family" is not equivalent to "Boolean cannot be done."

---

# 11. Displacement-preserving CSG has the same issue

Displacement-preserving composition depends on:

- common source identity;
- frame compatibility;
- domain compatibility;
- sampling budget.

Those are legitimate constraints for preserving displacement semantics.

They are not legitimate constraints for the body Boolean itself.

## Required distinction

```text
Can preserve displacement representation?  maybe no
Can compute Boolean of the represented solids? yes
```

The second must not depend on the first.

---

# 12. Ruled-patch unsupported cutter routes are representation leakage

The code contains explicit ruled unsupported-cutter diagnostics.

A ruled face should be just another parameterized face to the generic Boolean layer.

If a cutter intersects it, the kernel needs to:

- compute the intersection curve;
- invert/map it to UV;
- split the ruled face;
- classify fragments.

A "ruled cutter unsupported" route shows that these common operations are not yet complete.

---

# 13. Loft-specific CSG is a major architectural smell

There is extensive CSG machinery specifically for loft bodies:

- loft eligibility;
- loft pair selection;
- branching loft policies;
- branch decomposition;
- loft/primitive source normalization;
- cut-loop construction;
- generated caps;
- fragment topology;
- seam/shell assembly;
- loft-specific validity;
- loft CSG provenance.

Some of this code contains valuable generic algorithms.

But it should not be keyed to "loftness."

## Key observation

Once loft has produced a valid `SurfaceBody`, CSG should not care that loft produced it.

If it does, SurfaceBody has failed to encapsulate its geometry/topology.

## Recommendation

Extract generic parts into:

- surface intersection;
- trimming;
- fragment classification;
- reconstruction;
- validity.

Then delete loft-specific Boolean dispatch wherever the same operation can run from body topology alone.

---

# 14. Branching loft decomposition should become generic body decomposition

Branching loft CSG currently has specialized branch graph logic.

The generic problem is:

> A body may contain multiple connected topological regions/components and intersections may affect only some of them.

That should be handled through:

- shell connected components;
- face/fragment adjacency;
- generic component decomposition.

Not through provenance-specific loft branch graphs.

---

# 15. Primitive-specific implicit adapters should not be required for general CSG

The code recognizes primitives and creates exact implicit nodes for:

- box;
- sphere;
- cylinder;
- some affine ruled/polygon loft cases.

This is useful optimization.

But a general CSG system must not depend on recognizing a body as a primitive.

An arbitrary B-rep box and a primitive-authored box should Boolean identically.

Primitive recognition can select a faster solver; it must not determine whether the operation is possible.

---

# 16. Box/box analytic path is currently a bounded special case

There is substantial machinery for an initial box slicing / orthogonal planar route.

This is a useful kernel-development fixture.

It is not a complete planar CSG implementation.

General planar-faced polyhedra require:

- arbitrary plane orientation;
- arbitrary polygonal trims;
- multiple cuts per face;
- coplanar overlap handling;
- complete fragment graph;
- shell reconstruction.

The generic planar implementation should emerge from the common face-splitting pipeline, not remain box-specific.

---

# 17. Coplanar contact remains incomplete

Public union validation explicitly identifies coplanar loft overlap as potentially unsupported.

Coplanar/coincident faces are not a rare edge case in CAD.

They occur constantly in:

- unions of attached solids;
- repeated features;
- stacked extrusions;
- cuts aligned with existing faces;
- mirrored/arrayed features.

## Required capabilities

- coincident face overlap region construction;
- partial coincident overlap;
- orientation-aware ownership;
- duplicate-face elimination;
- face-touch union;
- coplanar edge splitting;
- tolerance-stable classification.

A Boolean system that cannot robustly handle coincident/coplanar surfaces is not complete.

---

# 18. Point/edge-only contact semantics need complete policy

The code classifies:

- point touch;
- edge touch;
- face touch;
- near touch;
- overlap;
- containment;
- equal.

It labels edge-/point-only contact as non-manifold for union reconstruction.

This classification is useful.

The missing piece is a single documented operation policy for each case.

For example:

### Union

- face-touch may produce one manifold solid if topology/orientation permit;
- edge/point touch may legitimately produce a multi-component or non-manifold result depending on Impression's solid model policy.

### Intersection

- result may be lower-dimensional.

### Difference

- tangent/touching cutter may produce no volumetric change.

The result contract must decide whether Impression models only volumetric solids or also lower-dimensional results.

Right now behavior is route-specific.

---

# 19. Lower-dimensional Boolean results need explicit semantics

`SurfaceBooleanClassification` currently includes:

- open;
- closed;
- empty.

But surface/surface contact can yield:

- point;
- curve;
- sheet;
- solid.

A solid-only API may legitimately classify point/curve/sheet intersection as empty **solid volume**, but that must be explicit.

Otherwise contact behavior becomes ambiguous.

## Recommendation

Define result dimensionality separately from body classification.

For example:

```text
dimension: 3 | 2 | 1 | 0 | empty
body: SurfaceBody | None
contact_geometry: ...
```

This also improves diagnostics and downstream feature logic.

---

# 20. Current body relation often relies on bounds

Several routes use bounds to classify disjoint/touching/overlap or to detect candidate contact.

Bounds are appropriate for broad phase.

They are not sufficient for final geometric relation.

The SurfaceBody audit already noted that `bounds_estimate()` is sampling-based for general patches.

## Required

- guaranteed conservative bounds for broad phase;
- exact/tolerance-aware geometric classification for narrow phase.

No success/no-cut decision should be based solely on approximate sampled bounds.

---

# 21. Generic Boolean pipeline required

The desired architecture should reduce every 3D solid Boolean to the same logical stages.

## Stage 1: normalize valid operands

- validate `SurfaceBody`;
- preserve multi-shell semantics;
- normalize transforms;
- establish one tolerance policy.

## Stage 2: broad-phase face pairing

- guaranteed conservative face bounds;
- shell/component bounds;
- skip provably disjoint pairs.

## Stage 3: surface-surface intersection

For each candidate pair:

```text
SurfaceIntersectionResult
    3D curves / overlap regions
    p-curves on A
    p-curves on B
    contact / degeneracy classification
    residual / tolerance evidence
```

Solver can vary by family pair internally.

Result contract must not vary.

## Stage 4: split faces

Each participating face is split by its p-curves into `SurfaceFragment` values.

This requires the common SurfacePatch functionality identified in the SurfaceBody audit.

## Stage 5: classify fragments

For each fragment:

```text
inside | outside | boundary
```

against the opposing body.

Coincident fragments receive explicit ownership/orientation classification.

## Stage 6: select fragments by operation

### Union

Keep exterior fragments.

### Intersection

Keep interior fragments.

### Difference

Keep base exterior fragments and cutter-interior boundary fragments with reversed orientation where required.

This stage should be operation-generic.

## Stage 7: reconstruct topology

- reuse inherited boundary uses;
- create generated intersection boundary uses;
- form boundary loops;
- pair shared edges into seams;
- orient shells;
- find connected components;
- classify outer/cavity nesting.

## Stage 8: validate

- manifold law;
- closure;
- orientation;
- no invalid self-intersection;
- tolerance consistency;
- no dangling generated edges.

## Stage 9: return

Return one canonical result independent of source patch families.

---

# 22. Patch-family responsibility under the generic pipeline

Each patch family must implement/adapt the same small set of capabilities.

## Mandatory for full CSG

1. conservative bounds;
2. surface evaluation;
3. parameter projection/inversion;
4. intersection participation;
5. p-curve generation;
6. face splitting / bounded fragment representation;
7. boundary evaluation;
8. serialization of resulting trimmed patch.

If a family cannot natively do one step, an adapter may convert it to a more general surface representation.

The adapter must remain surfaced; mesh fallback remains prohibited.

---

# 23. Promotion should be representation conversion, not operation refusal

For some families, generic CSG may require promotion.

Examples:

- heightmap -> subdivision/B-spline/implicit bounded representation;
- displacement -> detached general surface representation;
- subdivision -> refined parametric fragments;
- implicit -> extracted bounded surface patches at declared tolerance.

The key rule:

> Promotion may change patch family, but it must not change Boolean semantics.

And:

> Failure to preserve the original patch family is not a Boolean failure.

---

# 24. Exact versus declared-tolerance CSG

Not all families can support algebraically exact intersections.

The API should distinguish:

- exact result;
- declared-tolerance result;
- invalid/non-convergent result.

This is already partially present in solver records.

The missing step is to make this a **quality property of one generic Boolean result**, not a family-pair eligibility barrier.

A B-spline/subdivision intersection can be accepted at declared tolerance if residual and topology validation pass.

---

# 25. Boolean completeness must include repeated re-entry

A route is not complete if it works only on pristine authored operands.

Required test:

```text
A boolean B -> R1
R1 boolean C -> R2
R2 boolean D -> R3
...
```

Every result must preserve enough geometry/topology to participate again.

This is where durable:

- trim identity;
- seam identity;
- p-curves;
- fragment provenance;
- shell validity

matter.

Several current specialized routes retain metadata evidence, but metadata is not a substitute for complete reconstructed topology.

---

# 26. Union completeness requirements

Union must handle at least:

- disjoint bodies;
- containment;
- equal bodies;
- partial overlap;
- face-touch;
- coplanar partial overlap;
- arbitrary oriented planar faces;
- curved face intersections;
- multiple intersection loops;
- multiple shells/components;
- cavity elimination;
- cavity preservation;
- repeated union;
- mixed patch families.

No result may retain overlapping closed shells and still claim success.

---

# 27. Difference completeness requirements

Difference must handle:

- disjoint no-cut;
- cutter fully inside base -> cavity;
- cutter containing base -> empty;
- partial through-cut;
- tangent cutter;
- coplanar cutter;
- cutter intersecting multiple faces;
- multiple cutters;
- sequential cutters;
- cutter with multiple shells;
- base with cavities;
- generated caps;
- arbitrary curved cut loops;
- mixed patch families.

The existing difference success gate is valuable and should remain, but it must guard a generic executor rather than many special routes.

---

# 28. Intersection completeness requirements

Intersection must handle:

- disjoint -> empty;
- containment -> contained body;
- equal -> equivalent body;
- partial overlap;
- disconnected overlap regions;
- multiple shells;
- coincident regions;
- lower-dimensional contact semantics;
- mixed patch families;
- repeated re-entry.

Current higher-order routes that support only intersection are useful stepping stones, but not overall CSG completion.

---

# 29. Missing CSG-specific common constructs

The SurfaceBody audit identified common kernel gaps.

For CSG specifically, the following constructs should become stable reusable types.

## 29.1 SurfaceIntersectionCurve

Needs:

- canonical 3D curve;
- p-curve on first face;
- p-curve on second face;
- orientation;
- tolerance/residual;
- endpoints / closed-loop status;
- degeneracy/contact classification.

## 29.2 SurfaceFragment

Needs:

- source face/patch;
- bounded parameter region;
- boundary loops;
- inherited/generated boundary uses;
- orientation;
- source provenance.

## 29.3 FragmentClassification

Needs:

- inside/outside/boundary;
- representative witness;
- tolerance state;
- coincident ownership when applicable.

## 29.4 ReconstructedShell

Needs:

- oriented fragments;
- seam graph;
- loop graph;
- closure/manifold report;
- component/nesting classification.

These should not be loft-specific or box-specific.

---

# 30. Current support/refusal tracker should change purpose

The repository has extensive infrastructure proving that unsupported pairs refuse cleanly and do not fall back to meshes.

That was valuable during staged development.

For the intended completion state, the matrix should transition from:

> Which family pairs are unsupported?

to:

> Which mandatory common patch capabilities remain incomplete?

A more useful completion matrix would be family x capability, not family x family x operation.

Example:

| Family | bounds | inverse map | intersect | p-curve | split | trimmed persistence | full CSG |
|---|---|---|---|---|---|---|---|
| planar | ... | ... | ... | ... | ... | ... | ... |
| ruled | ... | ... | ... | ... | ... | ... | ... |
| revolution | ... | ... | ... | ... | ... | ... | ... |
| bspline | ... | ... | ... | ... | ... | ... | ... |
| nurbs | ... | ... | ... | ... | ... | ... | ... |
| sweep | ... | ... | ... | ... | ... | ... | ... |
| subdivision | ... | ... | ... | ... | ... | ... | ... |
| implicit | ... | ... | ... | ... | ... | ... | ... |
| heightmap | ... | ... | ... | ... | ... | ... | ... |
| displacement | ... | ... | ... | ... | ... | ... | ... |

Once every promoted family satisfies the required row, the N² Boolean pair matrix becomes largely a solver-dispatch detail rather than a feature support gate.

---

# 31. Code organization problem

`csg.py` currently contains:

- data model;
- capability registry;
- refusal tracking;
- sampled/implicit composition;
- loft CSG;
- primitive CSG;
- intersection dispatch adapters;
- contact classification;
- fragment construction;
- cap construction;
- shell assembly;
- validity;
- public API;
- mesh union utility.

This size and breadth make incomplete paths difficult to detect.

## Recommended decomposition

Conceptually:

```text
csg/
    api.py
    operands.py
    broad_phase.py
    intersections.py
    fragments.py
    classification.py
    selection.py
    reconstruction.py
    validity.py
    result.py
```

Patch-family solvers belong under the generic surface/intersection layer, not as independent Boolean implementations.

---

# 32. No-hidden-mesh remains a hard invariant

The desired full implementation should keep the current principle:

> Modeled CSG never silently drops to mesh CSG.

Mesh Boolean can remain as an explicit terminal/tool operation.

If surfaced CSG cannot meet its tolerance/validity contract, it should return an explicit failure.

However, once the common patch capabilities are complete, **patch family alone must no longer be a valid failure reason**.

---

# 33. Proposed failure taxonomy for completed CSG

Legitimate Boolean failures should be things like:

- invalid input body;
- non-manifold input unsupported by solid model policy;
- numerical non-convergence within declared tolerance;
- singular/degenerate case exceeding robustness policy;
- resource/budget limit explicitly requested by caller;
- result violates manifold/validity contract.

Illegitimate final failures include:

- "ruled cutter unsupported";
- "B-spline union unsupported";
- "sweep route only supports intersection";
- "heightmap result not representable as heightmap";
- "loft/primitive pairing unsupported";
- "multi-shell operand unsupported".

Those are implementation gaps, not semantic reasons the Boolean cannot exist.

---

# 34. Required invariants for the repair plan

When a future plan is generated from this review, it should enforce:

1. CSG accepts any valid closed `SurfaceBody`, including multi-shell bodies.
2. CSG does not dispatch semantically on producer provenance such as loft.
3. Primitive recognition is optimization only.
4. Patch family controls solver implementation, not Boolean availability.
5. Every promoted patch family satisfies the common Boolean capabilities.
6. Intersection results use one normalized contract.
7. Every intersection curve has usable parameter-space representation on both participating faces.
8. Every participating face can be split into persistent trimmed fragments.
9. Fragment classification is representation-independent.
10. Coincident/coplanar ownership has one generic policy.
11. Fragment selection is operation-generic.
12. Shell reconstruction is generic and reusable outside CSG.
13. Multi-shell and cavity semantics are explicit.
14. Every successful result passes a common manifold/closure/orientation validity gate.
15. Every successful result can immediately re-enter another CSG operation.
16. No mesh conversion occurs inside surfaced CSG.
17. Inability to preserve an input patch family triggers surfaced promotion, not Boolean refusal.
18. Family-pair support matrices are no longer the primary completion architecture.
19. Exact versus declared-tolerance quality is reported in the result.
20. CSG failure reasons describe geometry/numerics/validity, not missing family routes.

---

# 35. Relationship to the other two audits

The three reviews form one dependency chain.

## Loft audit

Loft becomes:

```text
explicit topology evolution -> patches
```

It stops inferring semantic correspondence.

## SurfaceBody audit

SurfaceBody becomes:

```text
complete trimmed-surface / B-rep kernel
```

It owns shared topology, splitting, classification support, reconstruction, and validity law.

## CSG audit

CSG becomes:

```text
generic Boolean over SurfaceBody
```

It no longer needs special knowledge of loft, primitive, heightmap, displacement, sweep, or subdivision provenance.

That decomposition is the architectural path most likely to satisfy the stated requirement:

> no CSG should fail because of the patch types from which the objects are made.
