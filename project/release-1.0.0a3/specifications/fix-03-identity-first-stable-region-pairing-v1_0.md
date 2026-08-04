# Fix 03: Identity-First Stable Region Pairing (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/loft-nm-mn-decomposition-architecture.md`
Source artifact: `testingImp/references/impression-issues.md` issue 3
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - stable region identity already exists in current topology records.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted identity resolution and bounded enumeration,
  identity/residue records, the loft dependency, paired/refusal outputs, reuse of
  current diagnostics, one module addition, and candidate-search performance.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
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
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 12.5
- Split decision: remain whole; identity reduction is the preconditioned input step
  of the existing bounded enumeration method and produces one planner result.

## Source Field Carryover

- Source purpose: let explicitly identified multi-region stations bypass combinatorial ambiguity.
- Source responsibilities by category:
  - Functions/methods: identity-first resolver and subset candidate enumerator.
  - Data structures/models: matched identity pairs and unmatched residue.
  - Dependencies/services: loft correspondence planner.
  - Returns/outputs/signals: reduced search input or identity-conflict refusal.
  - Reusable code plan: existing ambiguity diagnostics and branch limit.
  - Performance-sensitive behavior: resolved pairs must not consume search branches.
  - UI, database, async, write, security, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: stable output ordering follows source station order.
- Source split/provenance notes: not applicable.

## Purpose

Resolve unambiguous region identities before combinatorial candidate enumeration.

## Problem And Outcome

Adjacent stations with many identical regions can enumerate more than the
default 64 ambiguity branches even when every region has stable explicit
identity. Explicit one-to-one identity must remove candidates before subset
enumeration, leaving the branch limit for genuinely ambiguous residue.

## Scope

- Pair unique compatible region identities before geometric candidate creation.
- Remove resolved regions from ambiguity enumeration.
- Diagnose duplicate, missing, or contradictory identities rather than guessing.
- Preserve the existing branch limit for unresolved candidates.

Not in scope: raising or removing `ambiguity_max_branches`, or inventing identity
for anonymous regions.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- `src/impression/modeling/loft.py`: region correspondence planning near bounded
  split/merge candidate enumeration.
- Focused tests in loft correspondence/inference modules.
- Reproduction from the stable multi-region station in the test-modeling issue list.

## Chosen Defaults / Parameters

- Unique exact region IDs pair before geometry; source order controls output order.
- Duplicates/contradictions refuse; anonymous residue uses the existing 64-branch default.

## Data Ownership

- Source of truth: station `Region` identity records.
- Read ownership: loft correspondence planning.
- Write ownership: the planner creates derived pair/residue records only.
- Derived/cache data: pairings are recomputable from adjacent stations.
- Privacy/logging constraints: diagnostics include IDs/counts, not model source.

## Dependencies And Routes

- Domain/service dependencies: `src/impression/modeling/loft.py` correspondence planner.
- Database, GUI, and background/concurrency routes: not applicable.

## Prerequisite Handling

- Architecture feedback artifacts: none; status not applicable.
- Already implemented prerequisites: stable region IDs and ambiguity diagnostics.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed.

## Application Integration

- App type: library-only.
- User/caller surface: `Loft(...)` with multi-region stations.
- Invocation route: station normalization -> identity pairing -> residual candidate enumeration.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: a deterministic plan or named identity conflict.
- Integration validation: 65+ region fixture through public `Loft` planning.
- Incomplete status risk: a helper not wired before enumeration would not reduce branches.

## Reuse And Extraction Plan

- Existing code to reuse: ambiguity diagnostics and `ambiguity_max_branches` enforcement.
- Current reuse readiness: add one pre-enumeration step to the existing module.
- Extraction/wrapping/new reusable modules: none.
- Additions to existing library/modules: identity-pair reduction in `loft.py`.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing region IDs; derived matched-pair and unmatched-residue collections.
- Functions/methods: identity-pair resolver; `_enumerate_subset_assignment_candidates(...)`.
- UI fields/elements/components: not applicable.

## Performance Contract

- ID pairing is O(n); enumeration receives only unmatched regions and keeps the 64-branch cap.

## Error And State Behavior

- Duplicate/contradictory IDs fail before candidate enumeration with stable IDs in diagnostics.
- Anonymous ambiguity retains existing refusal behavior.

## Test Strategy

- Unit tests: identified, shuffled, mixed, duplicate, contradictory, and anonymous sets.
- Integrated route tests: public loft planning with more than 64 identified regions.
- Service/DB and GUI tests: not applicable.
- Production-data rule: generated local stations only.

## Contract

Input is two station region sets. Unique matching identities are deterministic
assignments; only the unmatched residue is passed to bounded inference. Output
ordering remains stable. Identity contradictions are invalid input with named
source and target regions.

## Acceptance Criteria

- More than 64 identity-matched regions plan successfully at the default limit.
- The planner does not visit ambiguity branches for resolved pairs.
- Duplicate or contradictory IDs fail with deterministic diagnostics.
- Truly ambiguous anonymous input still obeys the configured branch limit.

## Verification

[Paired test specification](../test-specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)

## Readiness Checklist

- [x] Ancestors, template, complete score, carryover, canonical status, and ledger are explicit.
- [x] No unresolved gap, blocker, missing prerequisite, or split coverage remains.
- [x] Routing, defaults, ownership, reuse, functions/models, performance, and errors are explicit.
- [x] App route and integrated proof are explicit; non-applicable surfaces are named.
- [x] Tests avoid production data and acceptance criteria are testable.
