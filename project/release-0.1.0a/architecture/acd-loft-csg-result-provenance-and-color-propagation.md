# Loft CSG Result Provenance And Color Propagation Architectural Change Document

Date: 2026-07-14
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-test-expansion-plan.md`
- Parent ACD, if any: `acd-loft-shell-connectivity-and-closure-evidence.md`

## Change Intent

Define how successful loft CSG results preserve source loft provenance,
authored colors, generated cap ownership, and cutter ownership.

This change is necessary because `Surface Spec 399` depends on successful loft
CSG result provenance, but the existing architecture only states that color and
metadata must survive. It does not yet define the ownership records or fallback
rules needed for implementation.

## Current Architecture

The lofted-body CSG architecture states that authored colors and metadata are
kernel inputs to the reference contract. Existing CSG architecture has general
result provenance concepts, but no loft-specific color/material ownership model
for:

- retained loft side fragments
- generated caps and cut boundaries
- primitive cutter fragments
- branch connector fragments
- loft/loft overlaps with competing source colors

## Target Architecture

Successful loft CSG results carry a loft-aware provenance map.

The target architecture has these components:

- `LoftCSGSourceFragmentRecord`: identifies source operand, loft transition,
  station interval, patch role, source patch id, and operation role.
- `LoftCSGResultFragmentRecord`: identifies output fragment and its source
  ownership chain.
- `LoftCSGColorOwnershipRecord`: resolves authored color/material for each
  result fragment.
- `LoftCSGGeneratedSurfaceStylePolicy`: deterministic fallback policy for caps,
  cut faces, seams, and synthesized bridge/connector surfaces.
- `LoftCSGMetadataPropagationReport`: verifies required metadata and color
  propagation for reference fixtures.

Color ownership rules:

- retained loft fragments inherit authored loft patch/face color
- retained primitive cutter fragments inherit cutter color
- generated cut faces inherit from the base operand unless a cutter-specific cut
  face style is explicitly present
- generated caps inherit nearest owning loft region, with a deterministic
  fallback semantic color when ownership is ambiguous
- loft/loft overlaps resolve by operation role and fragment classification, not
  by operand order alone
- every fallback must be recorded as fallback metadata, not silently presented
  as authored color

## Non-Goals

- Implementing successful loft CSG routes.
- Defining UI color controls.
- Promoting dirty artifacts to gold.
- Changing non-loft CSG color behavior except where shared provenance helpers
  are reused.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - replace metadata/color
    placeholder with the conformed ownership model.
  - `surfacebody-csg-architecture.md` - link loft-specific provenance as an
    extension of general CSG provenance.
  - `surface-csg-trim-fragment-reconstruction-architecture.md` - align fragment
    records with loft source ownership.
- Specs or plans affected:
  - `surface-399-loft-csg-metadata-color-propagation-v1_0.md` - superseded by
    child specs derived from this ACD after the single-shell loft CSG route ACD
    sequence.
  - `RT-LOFT-CSG-012` - remains unchecked until a successful result fixture can
    assert color propagation.

## Compatibility And Migration Strategy

- Existing CSG result metadata remains valid for non-loft cases.
- Loft-specific provenance is additive and may be absent on non-loft results.
- Initial tests should use bounded successful loft CSG cases produced by the
  single-shell loft CSG operation route sequence.
- Fallback colors are allowed only when explicitly marked as fallback
  ownership.

## Application Integration Contract

- App type: library-only
- User/caller surface: public CSG API returning `SurfaceBody` results
- Invocation route: `boolean_union`, `boolean_difference`, or
  `boolean_intersection`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: result body metadata and patch metadata expose source
  provenance and deterministic color/material ownership
- Integration validation: focused provenance unit tests plus reference fixture
  assertions for `RT-LOFT-CSG-012`

## Specification Manifest for Discovery

### Five-Pass Manifest Review Notes

- Pass 1: Added public boolean API and result-assembly route fields to prevent
  provenance/color work from landing as unobserved metadata helpers.
- Pass 2: Rechecked scoring against the active manifest-entry template; no UI,
  database, async, privacy, or cross-screen points apply.
- Pass 3: Kept both candidates below the split threshold; both remain in the
  16-24 review band with explicit cohesion explanations.
- Pass 4: Added cleanup fields for later promotion and parent/child coverage
  checks.
- Pass 5: Final rescore confirmed every readiness blocker resolution record is
  resolved; successful route geometry is represented by
  `acd-single-shell-loft-csg-operation-route.md`, and color ordering is
  represented by a predecessor candidate.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1: Rechecked provenance/color after creating the single-shell loft CSG
  route ACD; result geometry now has an explicit upstream ACD.
- Pass 2: Confirmed fragment provenance and color ownership should remain
  separate leaves because color can fail independently after provenance exists.
- Pass 3: Rescored both candidates; both remain in the 16-24 split-review band
  and neither crosses 25.
- Pass 4: Added readiness blocker resolution records tying route geometry to
  the new ACD and color ordering to the provenance candidate.
- Pass 5: Final review found no additional split needed.

### Candidate Spec: Loft CSG Fragment Provenance Map

Discovery purpose:
- Record source/output fragment lineage for loft CSG results.

Responsibilities:
- Functions/methods:
  - loft CSG provenance mapper
  - output fragment ownership resolver
- Data structures/models:
  - `LoftCSGSourceFragmentRecord`
  - `LoftCSGResultFragmentRecord`
- Dependencies/services:
  - CSG result fragments
  - loft patch role metadata
  - operation classification
- Returns/outputs/signals:
  - provenance map
  - missing provenance diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: general CSG provenance records
  - Additions to existing reusable library/module: loft-specific provenance
    records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes result metadata behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by output fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers inspecting loft CSG result metadata and
    reference fixtures generated from those results
- Invocation route:
  - CSG result assembly after a successful loft boolean operation
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - every loft CSG output fragment has source lineage or an explicit generated
    fragment reason
- Integration validation:
  - public boolean API result tests that inspect provenance on generated loft
    CSG outputs
- Incomplete status risk:
  - implemented in isolation if records exist but result assembly does not
    attach them to returned bodies
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - every output fragment must have a source ownership chain or an explicit
    generated-fragment reason
- Test strategy:
  - provenance map unit tests after successful route implementation
- Data ownership:
  - CSG owns result provenance; loft owns source patch role metadata
- Routes:
  - CSG result assembly to provenance map
- Reuse/extraction decision:
  - extend current CSG provenance records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need decide if source fragment ids are stable across `.impress` persistence
  or only within one runtime result.

Predecessor ACDs:
- `acd-single-shell-loft-csg-operation-route.md`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; successful route geometry is represented by predecessor ACD
    `acd-single-shell-loft-csg-operation-route.md`
- Required next action:
  - none
- Resolution artifact:
  - predecessor ACD `acd-single-shell-loft-csg-operation-route.md`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: source fragment mapping and missing
  provenance diagnostics are one result-lineage contract and must be validated
  together on returned CSG bodies.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft CSG Color Ownership Resolver

Discovery purpose:
- Preserve authored colors and deterministic generated-surface colors through
  loft CSG.

Responsibilities:
- Functions/methods:
  - color/material ownership resolver
  - generated surface fallback style policy
- Data structures/models:
  - `LoftCSGColorOwnershipRecord`
  - `LoftCSGGeneratedSurfaceStylePolicy`
- Dependencies/services:
  - loft CSG provenance map
  - source patch color metadata
- Returns/outputs/signals:
  - color propagation evidence
  - fallback color diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: color metadata helpers
  - Additions to existing reusable library/module: loft CSG color resolver
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes result metadata behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by output fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers and reference fixtures that inspect authored or
    generated output colors
- Invocation route:
  - provenance map to color ownership resolver during CSG result assembly
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - retained loft and cutter fragments preserve authored colors; generated
    surfaces receive explicit fallback style metadata
- Integration validation:
  - public boolean API result tests covering authored loft colors, cutter
    colors, and generated-surface fallback metadata
- Incomplete status risk:
  - implemented in isolation if fallback style metadata is calculated but not
    attached to result fragments
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - fallback colors are explicit metadata, never silent authored-color claims
- Test strategy:
  - authored loft/cutter color propagation tests
- Data ownership:
  - CSG result provenance owns output color lineage
- Routes:
  - provenance map to color ownership resolver
- Reuse/extraction decision:
  - add to existing CSG helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need final semantic fallback palette for generated cut/cap surfaces.

Predecessor candidates:
- `Loft CSG Fragment Provenance Map`

Predecessor ACDs:
- `acd-single-shell-loft-csg-operation-route.md`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; provenance ordering and route geometry are represented by predecessor
    candidate/ACD references
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Loft CSG Fragment Provenance Map`
  - predecessor ACD `acd-single-shell-loft-csg-operation-route.md`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: authored-color ownership and generated
  fallback style are the two branches of the same resolver and share the same
  provenance input.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Specification Conformance

- Parent specs created or affected:
  - `surface-399-loft-csg-metadata-color-propagation-v1_0.md` - superseded by
    child specs derived from this ACD and the single-shell loft CSG route ACD.
- Canonical child specs:
  - `../specifications/surface-411-loft-csg-fragment-provenance-map-v1_0.md` - canonical child from `Loft CSG Fragment Provenance Map`.
  - `../specifications/surface-412-loft-csg-color-ownership-resolver-v1_0.md` - canonical child from `Loft CSG Color Ownership Resolver`.
- Paired test specs:
  - none yet

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Parent specs are 100% represented by canonical child specs.
- [ ] Superseded parent specs are archived.
- [ ] Canonical child specs point to architecture or active ACD as primary ancestor.
- [ ] Paired test specs point to canonical child specs.
- [ ] Progression and indexes point to canonical child specs.
- [ ] Completed manifests are removed from active canonical architecture docs.
- [ ] Canonical architecture docs describe the conformed architecture.

## Closure Criteria

- Successful loft CSG results carry source/output fragment provenance.
- Authored colors are preserved for retained loft and cutter fragments.
- Generated surfaces have deterministic explicit fallback color metadata.
- `RT-LOFT-CSG-012` has a route to executable tests and reference fixture
  assertions.

## Closure Notes

- Canonical architecture updated:
  - none yet
- Archived or removed scaffolding:
  - none yet
- Follow-up ACDs:
  - none

## Change History

- 2026-07-14 - Initial draft. Reason: metadata/color propagation requires
  explicit loft CSG result provenance and ownership rules.
