# Loft Primitive Intersection And Cut-Loop Kernel ACD

Date: 2026-07-16
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`
- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-cut-shell-geometric-kernel.md`
- Predecessor spec:
  `project/release-0.1.0a/specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md`

## Change Intent

Define the first geometric kernel stage for cut-producing loft/primitive CSG:
turn primitive/loft intersection sources into patch-local, closed cut loops on
loft side and cap patches.

Surface Spec 421 adapter records are evidence, not executable trim loops. This
ACD prevents implementation from treating box bounds edges, sampled projections,
or open curve fragments as completed cut boundaries.

## Current Architecture

The current route can:

- identify a supported loft/primitive route
- classify loft fragments against primitive bounds
- emit adapter records with patch-local mapping attempts
- refuse intersecting cases before shell assembly

It does not define:

- primitive source regions for box, sphere, and cylinder cutters
- patch-local inversion residuals and clipping rules
- how curves close against existing loft boundaries, cap trims, and station
  seams
- tangent, grazing, duplicate, or zero-area loop policy

## Target Architecture

The intersection and cut-loop kernel owns:

- `LoftPrimitiveIntersectionSourceRecord`: normalized primitive/loft source
  curve or analytic region, primitive family, primitive face/region id, affected
  loft patch ids, and tolerance evidence.
- `LoftPrimitivePatchLocalCutCurveRecord`: one inverted/clipped curve segment
  in a loft patch parameter domain with residual diagnostics.
- `LoftPrimitiveCutBoundaryLoopRecord`: one closed oriented trim loop on a loft
  side or cap patch, assembled from new cut curves plus existing patch
  boundaries, cap trims, or station seams.
- `build_loft_primitive_intersection_sources(...)`
- `map_loft_primitive_cut_curves_to_patch_domains(...)`
- `build_loft_primitive_cut_boundary_loops(...)`

The kernel must refuse before cap/topology work when:

- a primitive family has no supported source-region representation
- a curve cannot be inverted into patch-local coordinates within tolerance
- a cut curve cannot close into a non-degenerate loop
- tangent or near-touch cases cannot be canonicalized as no-cut/touching cases

## Non-Goals

- Generated primitive caps.
- Boolean operation selection.
- Shell/seam assembly.
- Reference artifact generation.
- Mesh or raster fallback.

## Canonical Document Impact

- `surface-csg-trim-fragment-reconstruction-architecture.md` should gain a
  loft-specific source-normalization and cut-loop stage.
- `lofted-body-csg-reference-architecture.md` should distinguish adapter
  evidence from executable cut-loop evidence.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Spec 422 cannot build shells because no architecture defines
    closed loft patch cut loops.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Resolution provided by this ACD:
  - Defines the source and cut-loop kernel stage required before cap/topology
    work.
- Follow-on artifact:
  - Final specs for source normalization and patch-local cut-loop construction.
- Resolution status:
  - proposed; ready for manifest review.

## Compatibility And Migration Strategy

- Existing Surface Spec 421 adapter records remain the predecessor input.
- Existing exact reuse/no-cut routes do not enter this kernel.
- Unsupported primitive source regions return structured diagnostics instead of
  falling back to mesh intersections.

## Application Integration Contract

- App type: library-only
- User/caller surface: public Boolean API consumers using intersecting
  loft/primitive operands
- Invocation route: `surface_boolean_result` -> loft route selection -> Surface
  Spec 421 adapter -> this source/cut-loop kernel
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed cut-loop records or deterministic unsupported /
  invalid-loop diagnostics with `no_mesh_fallback=True`
- Integration validation: helper tests for source records and public-route
  refusal tests for unsupported/tangent/inversion failures

## Specification Manifest for Discovery

### Candidate Spec: Loft Primitive Intersection Source Normalization

Discovery purpose:
- Normalize box, sphere, and cylinder primitive source regions into durable
  cut-source records before patch-local trimming.

Responsibilities:
- Functions/methods:
  - primitive source-region enumerator
  - loft/primitive source normalizer
  - unsupported primitive-region diagnostic builder
- Data structures/models:
  - `LoftPrimitiveIntersectionSourceRecord`
  - `LoftPrimitiveAnalyticRegionRecord`
  - primitive-region diagnostic
- Dependencies/services:
  - Surface Spec 421 adapter records
  - primitive family metadata
  - tolerance policy helpers
- Returns/outputs/signals:
  - normalized source records
  - unsupported-region diagnostics
  - no-mesh proof
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive family classification and CSG tolerance policy
  - Additions to existing reusable library/module: `src/impression/modeling/csg.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by primitive source-region and loft patch counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers
- Invocation route:
  - loft route selection to adapter to source normalizer
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - source records or unsupported diagnostics
- Integration validation:
  - unit tests plus public-route diagnostic tests
- User-accessible surface:
  - public Boolean API result diagnostics
- Integration route:
  - public Boolean route to loft adapter to source normalizer
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - source-record tests plus public-route diagnostics for supported and unsupported primitive source regions
- Unwired risk:
  - helper records could exist while public loft CSG still reports generic adapter-only refusal
- Incomplete status risk:
  - later stages would embed primitive-specific shortcuts
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - box, sphere, and cylinder considered; unsupported source regions refuse
- Test strategy:
  - source-record tests for supported primitives and unsupported-region tests
- Data ownership:
  - CSG owns source records; primitive constructors own primitive metadata
- Routes:
  - adapter to source normalizer
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Surface Spec 421 implemented

Open questions / nuance discovered:
- Sphere and cylinder may produce analytic regions rather than planar face
  records; final spec should choose exact supported subset.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Intersection Source Normalization candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19
  - Score after update: 19
  - Split decision: review for split; cohesive source-normalization boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: Loft Patch-Local Cut Loop Construction.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 19

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after review
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked existing prerequisite
  - Artifact: `project/release-0.1.0a/specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md`

Split decision:
- Review for split.
- Cohesion reason: primitive source normalization is one boundary before
  patch-domain work.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft Patch-Local Cut Loop Construction

Discovery purpose:
- Convert normalized source records into closed patch-local cut loops on loft
  side and cap patches.

Responsibilities:
- Functions/methods:
  - patch-local inversion and clipping
  - cut arrangement builder
  - loop closure diagnostic builder
- Data structures/models:
  - `LoftPrimitivePatchLocalCutCurveRecord`
  - `LoftPrimitiveCutBoundaryLoopRecord`
  - loop closure diagnostic
- Dependencies/services:
  - intersection source records
  - loft patch domains and station interval metadata
  - existing patch-local CSG curve mapping helpers
- Returns/outputs/signals:
  - closed cut loops
  - residual/tolerance diagnostics
  - no-mesh proof
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface CSG patch-local curve mapper
  - Additions to existing reusable library/module: loft cut-loop helpers in `src/impression/modeling/csg.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by affected patch and curve segment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers
- Invocation route:
  - source records to patch-local loop builder
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - closed cut loops or deterministic loop/inversion diagnostics
- Integration validation:
  - unit tests for crossing, partial crossing, tangent/grazing refusal, and cap-loop interaction
- User-accessible surface:
  - public Boolean API result diagnostics
- Integration route:
  - source normalizer to patch-local cut-loop builder through the loft CSG route
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - cut-loop tests plus public-route refusal tests for tangent, grazing, and open-loop cases
- Unwired risk:
  - cap and topology work could consume synthetic or open boundaries instead of route-produced cut loops
- Incomplete status risk:
  - cap/topology stages would receive open or mismatched boundaries
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - tangent and zero-area loops refuse unless classified as no-cut/touching
- Test strategy:
  - cut-loop unit tests and public-route refusal tests
- Data ownership:
  - CSG owns cut-loop records; loft patches own parameter domains
- Routes:
  - source normalizer to loop builder
- Reuse/extraction decision:
  - reuse patch-local mapper and add loop closure helpers
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Intersection Source Normalization

Open questions / nuance discovered:
- Existing cap trims and station seams must participate in loop closure.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Patch-Local Cut Loop Construction candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19
  - Score after update: 19
  - Split decision: review for split; cohesive cut-loop construction boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: generated-cap/topology ACD candidates.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 19

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after source normalization spec exists
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate
  - Artifact: Loft Primitive Intersection Source Normalization

Split decision:
- Review for split.
- Cohesion reason: inversion, clipping, arrangement, and closure validate one
  patch-local loop contract.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Manifest Review History

- Pass 1 - Template/readiness review:
  - Added explicit reachability fields for both candidates.
  - No unresolved readiness blockers remained.
- Pass 2 - Rescore:
  - Loft Primitive Intersection Source Normalization: 19.
  - Loft Patch-Local Cut Loop Construction: 19.
  - No candidate reached the 25+ forced-split threshold.
- Pass 3 - Split review:
  - Both 16-24 candidates remain cohesive because each owns one library-stage
    boundary with one route and one validation surface.
- Pass 4 - Prerequisite review:
  - Source normalization has no missing prerequisite.
  - Cut-loop construction is sequenced after source normalization.
- Pass 5 - Final manifest readiness review:
  - No parent-only responsibilities, missing prerequisites, or unresolved
    blockers remain.
  - Both candidates are ready for final specification promotion in sequence.

## Specification Conformance

- Parent specs created or affected:
  - Surface Spec 422 - cut-shell parent currently blocked by this missing stage.
- Canonical child specs:
  - pending.
- Paired test specs:
  - pending.

## Conformance Checklist

- [ ] Source normalization final spec exists.
- [ ] Cut-loop construction final spec exists.
- [ ] Final specs are sequenced before generated cap/topology work.
- [ ] Public route keeps adapter-only refusal until closed loops are available.
- [ ] Canonical architecture is updated after implementation conforms.

## Closure Criteria

- Normalized source records and closed cut-loop records are implemented,
  tested, and consumed by downstream generated cap/topology specs.
- Adapter-only payloads are no longer mistaken for executable cut boundaries.

## Closure Notes

- Canonical architecture updated:
  - pending
- Archived or removed scaffolding:
  - pending
- Follow-up ACDs:
  - none

## Change History

- 2026-07-16 - Initial split from cut-shell umbrella. Reason: source
  normalization and patch-local loop closure are distinct architectural
  prerequisites before generated cap or shell assembly work.
