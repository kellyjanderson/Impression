# Fix 02: Coplanar Loft Face-Touch Union

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #243](https://github.com/kellyjanderson/Impression/issues/243)
Split provenance: none
Canonical status: Canonical
Review Score: 17
Prerequisites:
- none - the current `SurfaceBody` topology, tolerance policy, and result envelope are the baseline

## Source Field Carryover

- Source purpose: Fuse closed loft bodies that share an exact designed coplanar face into one validated surface shell without mesh fallback.
- Source responsibilities by category:
  - Functions/methods: classify coincident contacts, remove interior patch pairs, assemble/validate union
  - Data structures/models: coincident-contact evidence containing patch ids, orientation, overlap/domain match, and tolerance
  - Dependencies/services: surface evaluator, patch bounds, seam graph, CSG result/validity gates
  - Returns/outputs/signals: `SurfaceBooleanResult(status=succeeded, body=<one closed SurfaceBody>)` or precise refusal
  - UI surfaces/components: not applicable; library-only with preview/export consumers
  - UI fields/elements: not applicable
  - Reusable code plan: extend existing loft-pair CSG and public surface-union validator
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes; operands remain immutable
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: candidate bounds pruning prevents unconditional all-patch dense comparison
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: none

## Purpose

Fuse closed loft bodies that share an exact designed coplanar face into one validated surface shell without mesh fallback.

## Scope

- Owns:
  - candidate contact pruning and exact coincident trimmed-domain classification
  - opposite-orientation interior patch-pair removal
  - remaining patch/seam/adjacency reconstruction and one-shell validation
  - public union regression and real enclosure composition proof

- Does not own:
  - partial-overlap trimming or arbitrary intersecting union families
  - mesh repair or mesh-to-surface conversion

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none
- Issue-level split disposition: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: exact face-touch union is one solver branch, one result shell, and one public-route proof.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - contact classification, patch removal, shell assembly, and result validation
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - existing patch/seam/body invariants if extension is required
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - contact classification, patch removal, shell assembly, and result validation
- Tests:
  - `tests/test_coplanar_loft_union_outcome.py` - exact issue regression and negative controls
  - `tests/test_surface_csg.py` - public result and no-mesh invariants

## Chosen Defaults / Parameters

- use the existing modeling tolerance unless a caller supplies one
- treat patches as the removable interior pair only when trimmed domains match and normals are opposite within tolerance
- require exactly one closed output shell with complete seams and operand witnesses
- refuse ambiguous, partial-domain, or near-coplanar contacts rather than returning overlapping shells

## Data Ownership

- source of truth: input `SurfaceBody` patch topology and evaluator domains
- read ownership: CSG classifier reads immutable operand patches and tolerance
- write ownership: result assembler creates new patches/seams; operands are unchanged
- derived/cache data: contact candidates and reconstructed adjacency are recomputable
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - existing `execute_loft_pair_csg`, surface bounds/evaluation, seam rebuild, and public union validity gate
  - library route: `boolean_union` -> route selection -> coincident face-touch executor -> shared validator
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - owns face-touch classification and shell merger transition
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - surface body/seam records, tolerance policy, result envelope, and overlapping-shell invalid gate
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none
- Progression handling:
  - this leaf may proceed after independent review canonicalizes it

## Application Integration

- App type: library-only
- User/caller surface: public `boolean_union` consumed by modeling scripts and downstream preview/export
- Invocation route: surface operands -> union route selection -> coincident classifier -> shell assembly -> result gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: one closed surfaced union or explicit structured refusal
- Integration validation: public API fixture plus preview/export of the composed enclosure
- Incomplete status risk: completion requires the declared integrated route and prerequisite sequence to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: public `boolean_union` consumed by modeling scripts and downstream preview/export is the consuming public route and public API fixture plus preview/export of the composed enclosure

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: surface tolerance, patch bounds, result evidence, seam/adjacency rebuild, validity gates
- Current reuse readiness:
  - readiness: add a face-touch branch to the existing surface CSG module
- Extraction/wrapping needed:
  - extraction: reusable coincident-contact record/classifier inside existing CSG boundaries
- Additions to existing library/modules:
  - readiness: add a face-touch branch to the existing surface CSG module
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - `CoincidentPatchContact` - operand/patch ids, orientation relation, trimmed-domain match, tolerance evidence
- Functions/methods:
  - candidate contact classifier
  - interior-pair filter and remaining-shell assembler
  - union diagnostic/result evidence
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- candidate patch pairs are pruned by spatial bounds
- no whole-body dense sampling or mesh tessellation
- fixture completes within the normal focused-test timeout

## Error And State Behavior

- partial-domain or ambiguous contact returns unsupported/invalid with contact diagnostic
- open seams, duplicate shells, missing operand witnesses, or shell_count != 1 blocks success
- operands remain unchanged and no partial body is returned

## Test Strategy

- Unit tests:
  - contact candidate pruning, domain equivalence, orientation, patch removal, seam assembly, and negative classification
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - public `boolean_union` on exact face-touch lofts, near-coplanar and partial-overlap negatives, then preview/export consumer smoke
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- The issue fixture returns `status=succeeded` with one non-null closed `SurfaceBody`.
- The opposite-oriented shared interior patches are absent and seam coverage is complete.
- No duplicate overlapping shell or partial body can report success.
- Near-coplanar, partial-domain, or ambiguous contacts are not misclassified as exact face-touch union.
- No mesh fallback is invoked, and a real enclosure composition validates through the public route.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in front matter and matches a completed independent calculation.
- [x] Current implementation-spec template was loaded; its path is recorded below.
- [x] Independent adversarial recount completed.
- [x] No unresolved placeholder is hidden as implementation-ready behavior.
- [x] Source responsibilities are carried into durable sections.
- [x] Canonical status is Canonical.
- [x] Prerequisites are linked or marked not applicable.
- [x] Missing/stale architecture is tracked in the active ACD.
- [x] Missing prerequisite behavior is linked or marked not applicable.
- [x] Split coverage is recorded for issue-level splits.
- [x] Review ledger records the completed request-scoped passes.
- [x] Implementation owner/module and reuse/extraction decisions are named.
- [x] UI fields/elements and concurrency are explicit or not applicable.
- [x] Defaults, data ownership, app type, route, performance, privacy, and test strategy are explicit.
- [x] Acceptance criteria are observable and testable.
- [x] Independent `review specs` confirms cohesion, scoring, canonical status, and release responsibility coverage.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 17; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 2 x 2 = 4
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 17
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
