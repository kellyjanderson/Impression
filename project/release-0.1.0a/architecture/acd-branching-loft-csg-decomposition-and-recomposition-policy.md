# Branching Loft CSG Decomposition And Recomposition Policy Architectural Change Document

Date: 2026-07-14
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/loft-nm-mn-decomposition-architecture.md`
- `project/release-0.1.0a/architecture/loft-planner-executor-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-test-expansion-plan.md`
- Parent ACD, if any: `acd-loft-shell-connectivity-and-closure-evidence.md`

## Change Intent

Define the architecture that decides whether branching loft bodies can execute
CSG through decomposition, must refuse as underconstrained, or must wait for
additional authored constraints.

This change is necessary because the existing architecture names branching
policy and N-to-M decomposition, but does not yet define how a decomposed branch
CSG plan is recomposed into a valid result or how decomposition evidence feeds
the CSG planner.

## Current Architecture

The current lofted-body CSG architecture states that branching lofts are a
separate policy and may require decomposition or refusal. The N-to-M / M-to-N
architecture defines general decomposition of many-to-many loft transitions.

What is still missing:

- a CSG-specific branching policy record
- a branch decomposition plan that identifies sub-bodies and operation scopes
- a recomposition contract for sub-body CSG results
- refusal criteria for underconstrained or non-recomposable branch structures
- validation that decomposed branch CSG does not lose loft provenance

## Target Architecture

Branching loft CSG has a dedicated policy layer between loft eligibility and CSG
operation planning.

The target architecture has these components:

- `BranchingLoftCSGPolicyRecord`: classifies a branching loft operand as
  `single-shell-executable`, `decomposition-required`,
  `underconstrained-refusal`, or `unsupported-branch-form`.
- `BranchDecompositionPlan`: identifies branch sub-bodies, shared joint
  boundaries, source transition ids, and operation scopes.
- `BranchSubBodyCSGPlan`: maps each decomposed branch/body segment to a CSG
  operation against primitive or loft cutters.
- `BranchRecompositionRecord`: defines how sub-body CSG results rejoin at branch
  interfaces, including seam, cap, and provenance requirements.
- `BranchingLoftCSGDiagnostic`: structured refusal or incomplete-policy signal
  that never falls back to mesh.

Branching CSG is executable only when:

- the loft decomposition graph is resolved
- each sub-body has valid shell connectivity/closure evidence
- branch joint boundaries have deterministic ownership
- recomposition can rebuild a single valid result body or intentionally return a
  declared multi-body result
- provenance maps every output fragment to source loft branch and cutter
  ownership

## Non-Goals

- Defining generic N-to-M loft decomposition from scratch.
- Implementing successful single-shell loft CSG routes.
- Allowing underconstrained branching topology through CSG.
- Converting branch CSG to mesh execution.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - replace branching policy
    placeholder with conformed policy and recomposition contract.
  - `loft-nm-mn-decomposition-architecture.md` - link CSG-specific consumption
    of resolved decomposition records.
  - `loft-planner-executor-architecture.md` - describe emitted branch graph
    evidence needed by CSG policy.
- Specs or plans affected:
  - `surface-397-branching-loft-csg-execution-policy-v1_0.md` - current spec is
    superseded by child specs derived from this ACD.
  - `surface-398-underconstrained-branching-loft-csg-refusal-fixture-v1_0.md` -
    already covers one refusal case; future specs should preserve that refusal.
  - `RT-LOFT-CSG-007` and `RT-LOFT-CSG-008` - sequence after branch
    decomposition and recomposition policy exists.

## Compatibility And Migration Strategy

- Existing branching lofts continue to refuse through the CSG eligibility gate
  until the branch policy classifies them more specifically.
- Underconstrained branching refusal remains valid and should not be weakened by
  adding decomposition support.
- Decomposition support should be introduced first as diagnostic policy records,
  then as executable sub-body plans for bounded fixtures.

## Application Integration Contract

- App type: library-only
- User/caller surface: public CSG API with branching loft operands
- Invocation route: `boolean_union`, `boolean_difference`, or
  `boolean_intersection`
- Wiring owner/module: `src/impression/modeling/loft.py` and
  `src/impression/modeling/csg.py`
- Observable result: branching lofts either execute through declared
  decomposition/recomposition or refuse with structured diagnostics
- Integration validation: branch policy unit tests and public CSG API fixture
  probes for `RT-LOFT-CSG-007`, `RT-LOFT-CSG-008`, and `RT-LOFT-CSG-014`

## Specification Manifest for Discovery

### Five-Pass Manifest Review Notes

- Pass 1: Added app-type and public boolean API route fields so branch policy
  work cannot be implemented as disconnected helpers.
- Pass 2: Rechecked scoring against the active manifest-entry template; no UI,
  database, async, or privacy points apply.
- Pass 3: Kept both candidates below the split threshold; the recomposition
  candidate remains in the 16-24 review band with an explicit cohesion reason.
- Pass 4: Added cleanup fields for later spec promotion and parent/child
  coverage checks.
- Pass 5: Final rescore confirmed no 25+ candidates and every readiness
  blocker resolution record is resolved; branch graph and shell evidence are
  represented as predecessor candidates/ACDs.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1: Rechecked branch graph, policy, and recomposition leaves against the
  new single-shell route ACD; branching remains a separate route family.
- Pass 2: Confirmed the branch graph evidence candidate resolves the former
  branch-graph gap and should remain the first leaf in this ACD.
- Pass 3: Rescored all three leaves; all remain below 25, and policy plus
  recomposition remain in the 16-24 split-review band with cohesion reasons.
- Pass 4: Added readiness blocker resolution records to each candidate.
- Pass 5: Final review found no split needed; shell evidence and provenance
  sequencing are represented by predecessor ACDs.

### Candidate Spec: Loft Branch Graph Evidence

Discovery purpose:
- Expose loft branch graph evidence needed by branching CSG policy
  classification.

Responsibilities:
- Functions/methods:
  - loft branch graph evidence builder
  - branch joint diagnostic builder
- Data structures/models:
  - `LoftBranchGraphEvidence`
  - branch joint record
- Dependencies/services:
  - loft decomposition graph
  - loft shell connectivity evidence
- Returns/outputs/signals:
  - branch graph evidence
  - underconstrained branch diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: N-to-M decomposition concepts
  - Additions to existing reusable library/module: loft branch graph evidence
    helper
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
  - bounded by station, branch, and transition count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using branching loft operands
- Invocation route:
  - loft executor branch/decomposition evidence to CSG branch policy
- Wiring owner/module:
  - `src/impression/modeling/loft.py`
- Observable result:
  - branching loft operands expose branch graph evidence or branch-specific
    diagnostics before CSG policy classification
- Integration validation:
  - loft executor tests plus public boolean API branch-policy probes
- Incomplete status risk:
  - implemented in isolation if branch evidence is not exposed to CSG policy
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - branch joints require explicit ownership and transition membership
- Test strategy:
  - branch graph evidence tests for executable and underconstrained forms
- Data ownership:
  - loft owns branch graph evidence
- Routes:
  - loft decomposition graph to CSG branch policy
- Reuse/extraction decision:
  - add to existing loft module
- UI field/control inventory:
  - not applicable

Predecessor ACDs:
- `acd-loft-shell-connectivity-and-closure-evidence.md`

Open questions / nuance discovered:
- Branch graph evidence is separate from CSG policy so branch topology can be
  validated before boolean operation selection.

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
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; shell prerequisites are represented by predecessor ACD
    `acd-loft-shell-connectivity-and-closure-evidence.md`
- Required next action:
  - none
- Resolution artifact:
  - predecessor ACD `acd-loft-shell-connectivity-and-closure-evidence.md`
- Resolution status:
  - resolved

Split decision:
- Small.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Branching Loft CSG Policy Classification

Discovery purpose:
- Classify branching loft operands before CSG execution.

Responsibilities:
- Functions/methods:
  - branching loft CSG classifier
  - underconstrained branch diagnostic builder
- Data structures/models:
  - `BranchingLoftCSGPolicyRecord`
  - `BranchingLoftCSGDiagnostic`
- Dependencies/services:
  - loft shell validity evidence
  - loft decomposition graph
- Returns/outputs/signals:
  - branch policy class
  - refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG eligibility diagnostics
  - Additions to existing reusable library/module: branch policy helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG acceptance/refusal for branching lofts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by branch graph size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using branching loft operands
- Invocation route:
  - public boolean API call to CSG eligibility and branch policy classification
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/loft.py`
- Observable result:
  - branching loft operands are classified as executable, decomposition
    required, or refused with diagnostics before CSG execution
- Integration validation:
  - public boolean API tests covering executable, decomposition-required, and
    refusal classifications
- Incomplete status risk:
  - implemented in isolation if branch diagnostics are produced but not used by
    the CSG eligibility route
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - missing or underconstrained branch graphs refuse
- Test strategy:
  - policy classification tests for executable, decomposition-required, and
    refusal branch forms
- Data ownership:
  - loft owns branch graph; CSG owns policy classification
- Routes:
  - loft branch graph to CSG branch policy
- Reuse/extraction decision:
  - add to existing modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need decide whether declared multi-body results are allowed or whether every
  decomposed branch CSG must recompose to one body.

Predecessor candidates:
- `Loft Branch Graph Evidence`

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
  - none; branch topology evidence is represented by predecessor candidate
    `Loft Branch Graph Evidence`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Loft Branch Graph Evidence`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: the score is in the 16-24 band because the
  candidate spans classification and refusal diagnostics, but those diagnostics
  are the observable output of the same policy decision.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Branch Decomposition And Recomposition Records

Discovery purpose:
- Define executable branch sub-body CSG plans and recomposition evidence.

Responsibilities:
- Functions/methods:
  - branch decomposition planner adapter
  - branch recomposition validator
- Data structures/models:
  - `BranchDecompositionPlan`
  - `BranchSubBodyCSGPlan`
  - `BranchRecompositionRecord`
- Dependencies/services:
  - N-to-M decomposition records
  - loft shell connectivity evidence
  - CSG result provenance
- Returns/outputs/signals:
  - executable branch plan
  - recomposition validity signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: N-to-M decomposition concepts
  - Additions to existing reusable library/module: CSG-specific branch records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes branching CSG execution behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by branch/sub-body count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using decomposable branching loft operands
- Invocation route:
  - branch policy classification to decomposition plan to sub-body CSG to
    recomposition validation
- Wiring owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Observable result:
  - decomposable branch operands produce sub-body CSG plans and recomposition
    evidence or refuse with a branch-specific diagnostic
- Integration validation:
  - public boolean API tests with bounded decomposable branch fixtures and
    underconstrained refusal fixtures
- Incomplete status risk:
  - implemented in isolation if decomposition records do not feed CSG execution
    and recomposition validation
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - sub-body results must carry source branch ids and recomposition seams
- Test strategy:
  - decomposition/recomposition record tests plus bounded CSG probes
- Data ownership:
  - loft owns decomposition; CSG owns operation/recomposition acceptance
- Routes:
  - decomposition plan to sub-body CSG to recomposition validator
- Reuse/extraction decision:
  - consume N-to-M decomposition architecture; add CSG-specific records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need determine canonical result shape for branch CSG at a joint: one body,
  multi-shell body, or explicit multi-body collection.

Predecessor candidates:
- `Branching Loft CSG Policy Classification`

Predecessor ACDs:
- `acd-loft-shell-connectivity-and-closure-evidence.md`
- `acd-loft-csg-result-provenance-and-color-propagation.md`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
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
- Total: 18.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; policy, shell evidence, and provenance sequence are represented by
    predecessor candidate/ACD references
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Branching Loft CSG Policy Classification`
  - predecessor ACD `acd-loft-shell-connectivity-and-closure-evidence.md`
  - predecessor ACD `acd-loft-csg-result-provenance-and-color-propagation.md`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: decomposition and recomposition are one
  branch execution contract and should be split only if implementation proves
  the records are independently deliverable.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Specification Conformance

- Parent specs created or affected:
  - `surface-397-branching-loft-csg-execution-policy-v1_0.md` - superseded by
    child specs derived from this ACD.
- Canonical child specs:
  - `../specifications/surface-408-loft-branch-graph-evidence-v1_0.md` - canonical child from `Loft Branch Graph Evidence`.
  - `../specifications/surface-409-branching-loft-csg-policy-classification-v1_0.md` - canonical child from `Branching Loft CSG Policy Classification`.
  - `../specifications/surface-410-branch-decomposition-and-recomposition-records-v1_0.md` - canonical child from `Branch Decomposition And Recomposition Records`.
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

- Branching lofts are classified by a dedicated CSG policy record.
- Underconstrained branch cases still refuse explicitly.
- At least one decomposable branch fixture produces an executable sub-body CSG
  plan or a documented non-executable diagnostic with no mesh fallback.
- Canonical lofted-body CSG architecture describes branch decomposition and
  recomposition policy.

## Closure Notes

- Canonical architecture updated:
  - none yet
- Archived or removed scaffolding:
  - none yet
- Follow-up ACDs:
  - none

## Change History

- 2026-07-14 - Initial draft. Reason: branching loft CSG specs require explicit
  decomposition and recomposition architecture.
