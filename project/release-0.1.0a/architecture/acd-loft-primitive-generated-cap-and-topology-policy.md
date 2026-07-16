# Loft Primitive Generated Cap And Topology Policy ACD

Date: 2026-07-16
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`
- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-cut-shell-geometric-kernel.md`
- Predecessor ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Change Intent

Define how cut-producing loft/primitive CSG creates generated primitive cap
fragments and selects Boolean result topology.

This is separate from cut-loop construction because a closed loop does not yet
answer which generated cap family is valid, which fragments survive, which
fragments reverse orientation, or whether the result is exterior, cavity,
multi-shell, empty, or refused.

## Current Architecture

Current code can classify loft patches as survive/discard/cut-cap contributors
against primitive bounds, but it cannot:

- create generated primitive cap fragments
- choose exact supported cap patch families
- distinguish exterior shell edits from interior cavity topology
- express operation-specific orientation for difference, union, and
  intersection
- refuse unsupported cap regions before shell assembly

## Target Architecture

This ACD introduces:

- `LoftPrimitiveGeneratedCapRecord`: generated primitive cap patch, cap source
  region, paired loft cut-loop ids, trim-loop payload, and no-mesh proof.
- `LoftPrimitiveUnsupportedCapDiagnostic`: unsupported cap reason, source
  primitive region, required patch family, and operation.
- `LoftPrimitiveOperationFragmentSelectionRecord`: selected retained loft
  fragments, generated caps, reversed fragments, discarded fragments, and
  operation rationale.
- `LoftPrimitiveResultTopologyRecord`: topology class such as `empty`,
  `exterior-shell`, `interior-cavity`, `multi-shell`, or `refused`.
- `build_loft_primitive_generated_caps(...)`
- `select_loft_primitive_operation_fragments(...)`
- `classify_loft_primitive_result_topology(...)`

## Non-Goals

- Patch-local curve inversion and loop closure.
- Seam/use pairing and durable shell construction.
- Runtime validity/persistence gates.
- Reference artifact generation.

## Canonical Document Impact

- `surface-csg-trim-fragment-reconstruction-architecture.md` should describe
  generated primitive caps and topology selection as a family-specific CSG cap
  policy.
- `lofted-body-csg-reference-architecture.md` should distinguish partial
  overlap, contained cutter cavity, union, intersection, and refusal outcomes.
- `surfacebody-csg-architecture.md` should document generated caps as
  surface-native patch fragments, not mesh caps.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Spec 422 cannot assemble a shell because cap generation and result
    topology are undefined.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Resolution provided by this ACD:
  - Defines generated cap and topology policies after cut loops exist.
- Follow-on artifact:
  - Final specs for generated cap construction and topology/fragment selection.
- Resolution status:
  - proposed; ready for manifest review.

## Compatibility And Migration Strategy

- Existing exact reuse cases bypass this ACD.
- Unsupported primitive cap representations refuse with structured diagnostics.
- Generated caps must be supported surface patch families with provenance.
- No mesh cap, tessellation cap, or raster cap is allowed.

## Application Integration Contract

- App type: library-only
- User/caller surface: public Boolean API consumers using cut-producing
  loft/primitive operands
- Invocation route: closed cut loops to generated cap builder to topology
  selector
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: generated cap and topology records, or unsupported cap /
  orientation diagnostics
- Integration validation: tests for planar box caps, sphere/cylinder supported
  or refused cap regions, interior cavity difference, partial-overlap
  difference, union, intersection, and touching/no-cut routing

## Specification Manifest for Discovery

### Candidate Spec: Loft Primitive Generated Cap Construction

Discovery purpose:
- Build supported primitive cap fragments from closed loft cut loops.

Responsibilities:
- Functions/methods:
  - generated primitive cap builder
  - cap patch family selector
  - unsupported cap diagnostic builder
- Data structures/models:
  - `LoftPrimitiveGeneratedCapRecord`
  - generated cap trim-loop payload
  - unsupported cap diagnostic
- Dependencies/services:
  - cut boundary loop records
  - primitive source-region records
  - Surface CSG cap policy
- Returns/outputs/signals:
  - generated cap fragments
  - cap-loop pairing ids
  - unsupported cap diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface CSG cap policy and planar cap helpers
  - Additions to existing reusable library/module: loft primitive cap builder in `src/impression/modeling/csg.py`
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
  - bounded by generated cap loop count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers
- Invocation route:
  - cut loops to generated cap builder
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - generated cap records or unsupported cap diagnostics
- Integration validation:
  - cap-builder tests and public-route unsupported-cap tests
- User-accessible surface:
  - public Boolean API result diagnostics
- Integration route:
  - cut-loop records to generated cap builder through the loft CSG route
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - cap-builder tests plus public-route unsupported-cap diagnostics
- Unwired risk:
  - shell assembly could invent caps or silently ignore unsupported primitive source regions
- Incomplete status risk:
  - shell assembly would remain open or invent unsupported caps
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported analytic cap regions refuse until represented by supported patch families
- Test strategy:
  - unit tests for cap construction and unsupported cap refusal
- Data ownership:
  - CSG owns generated cap records; primitive source records own source identity
- Routes:
  - cut-loop builder to generated cap builder
- Reuse/extraction decision:
  - reuse cap policy and add loft cap pairing records
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Patch-Local Cut Loop Construction

Open questions / nuance discovered:
- Sphere and cylinder support may need implicit or higher-order cap
  representation; final spec should define the initial supported subset.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Generated Cap Construction candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19
  - Score after update: 19
  - Split decision: review for split; cohesive generated-cap construction boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: Loft Primitive Fragment Topology And Operation Selection.

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
  - linked predecessor ACD candidate
  - Artifact: Loft Patch-Local Cut Loop Construction

Split decision:
- Review for split.
- Cohesion reason: cap family selection, cap trim payload, and cap refusal
  diagnostics are one generated-cap contract.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft Primitive Fragment Topology And Operation Selection

Discovery purpose:
- Select retained, discarded, reversed, and generated fragments and classify
  result topology for Boolean operations.

Responsibilities:
- Functions/methods:
  - operation-specific fragment selector
  - cavity/exterior topology classifier
  - orientation and normal policy resolver
- Data structures/models:
  - `LoftPrimitiveOperationFragmentSelectionRecord`
  - `LoftPrimitiveResultTopologyRecord`
  - orientation diagnostic
- Dependencies/services:
  - Surface Spec 421 fragment classifications
  - generated cap records
  - body relation/contact classifier
- Returns/outputs/signals:
  - selected fragment set
  - result topology class
  - orientation/refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: operation selector and body relation helpers
  - Additions to existing reusable library/module: loft topology selector in `src/impression/modeling/csg.py`
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
  - bounded by candidate fragment and cap count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers
- Invocation route:
  - fragment classifications plus generated caps to topology selector
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - selected fragments and topology records, or orientation diagnostics
- Integration validation:
  - tests for difference cavity, partial-overlap difference, union, intersection, touching/no-cut, and orientation refusal
- User-accessible surface:
  - public Boolean API result diagnostics
- Integration route:
  - generated cap and fragment classification records to operation topology selector
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - topology tests plus public API route tests for difference, union, intersection, touching, and orientation refusal
- Unwired risk:
  - seam assembly could build the wrong exterior, cavity, empty, or multi-shell topology
- Incomplete status risk:
  - shell assembly would not know whether it is building exterior, cavity, empty, or multi-shell topology
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - touching cases route to exact/no-cut outcomes; true cut cases require topology records
- Test strategy:
  - topology unit tests and public API route tests
- Data ownership:
  - CSG owns operation selection; source/cap records own geometry identity
- Routes:
  - generated cap builder to topology selector
- Reuse/extraction decision:
  - add to existing CSG operation selection helpers
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Generated Cap Construction

Open questions / nuance discovered:
- Difference must distinguish internal cavity boundaries from exterior shell edits.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Fragment Topology And Operation Selection candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19
  - Score after update: 19
  - Split decision: review for split; cohesive topology-selection boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: seam/shell/validity ACD candidates.

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
  - create final spec after cap construction spec exists
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate
  - Artifact: Loft Primitive Generated Cap Construction

Split decision:
- Review for split.
- Cohesion reason: Boolean operation semantics, topology class, and orientation
  are one selection decision.

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
  - Loft Primitive Generated Cap Construction: 19.
  - Loft Primitive Fragment Topology And Operation Selection: 19.
  - No candidate reached the 25+ forced-split threshold.
- Pass 3 - Split review:
  - Both 16-24 candidates remain cohesive because cap construction and topology
    selection are separate sequential stage boundaries.
- Pass 4 - Prerequisite review:
  - Generated caps are sequenced after cut-loop construction.
  - Topology selection is sequenced after generated cap construction.
- Pass 5 - Final manifest readiness review:
  - No parent-only responsibilities, missing prerequisites, or unresolved
    blockers remain.
  - Both candidates are ready for final specification promotion in sequence.

## Specification Conformance

- Parent specs created or affected:
  - Surface Spec 422 - blocked parent currently lacks cap/topology architecture.
- Canonical child specs:
  - pending.
- Paired test specs:
  - pending.

## Conformance Checklist

- [ ] Generated cap final spec exists.
- [ ] Topology/operation selection final spec exists.
- [ ] Unsupported cap regions refuse without mesh fallback.
- [ ] Difference cavity and exterior shell topology are both represented.
- [ ] Canonical architecture is updated after implementation conforms.

## Closure Criteria

- Generated cap records and topology records are implemented, tested, and
  consumed by seam/shell assembly.
- Unsupported cap and orientation cases refuse before invalid shells are built.

## Closure Notes

- Canonical architecture updated:
  - pending
- Archived or removed scaffolding:
  - pending
- Follow-up ACDs:
  - none

## Change History

- 2026-07-16 - Initial split from cut-shell umbrella. Reason: generated cap
  representation and Boolean topology policy are separate architecture from
  patch-local loop construction and shell assembly.
