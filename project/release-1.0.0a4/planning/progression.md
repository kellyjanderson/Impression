# v1.0.0a4 Corrective Release Progression

Date: 2026-08-04
Status: Ready for implementation

Only the 19 canonical implementation leaves and their paired canonical test
specifications are executable anchors in this progression. Archived parents do
not appear as implementation work.

## Source Documents

- Architecture:
  - [Preview Reload Coordination ACD](../architecture/acd-preview-reload-coordination.md)
  - [Loft Identity And Junction Correctness ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
  - [Surface Boolean Correctness And API Boundary ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
- Manifest: [Known-Issue Intake](known-issue-intake.md); retained as release-scope evidence, not specification scaffolding
- Final specs: [Canonical implementation specification index](../specifications/README.md)
- Test specs: [Canonical test specification index](../test-specifications/README.md)
- Fixed-point review: [four-pass refinement ledger](../spec-refinement-history/a4-open-issues-20260804-165103.md)
- Progression template: `/Users/k/Documents/Projects/.agents/process/templates/progression-template.md`

## Completion Rule

Progression checkboxes are truth markers. Check implementation only when the
linked feature leaf is implemented. Check route wiring only when the declared
user/caller route reaches the behavior. Check validation only when the paired
test specification passes through the real route. Check status updates only
after implementation and route validation are both complete.

If implementation discovers a prerequisite gap, leave the current item
unchecked and add `Status: Missing prerequisite - <path>` beneath it. Add or
move the prerequisite ahead of the dependent work; do not relabel the dependent
leaf blocked when an explicit prerequisite artifact or implementation is the
next action.

## Dependency Waves

| Wave | Canonical leaves | Dependency outcome |
|---:|---|---|
| 1 | 01A, 02, 03, 06, 08A, 09A | Independent coordination, topology, union, trim, and evidence foundations |
| 2 | 01B, 01C2A, 05A, 09B | Typed invalidation/apply, exact pairing, and truthful difference gate |
| 3 | 01C1, 01C2B, 05B | Complete preview routes/state and synthetic identity lineage |
| 4 | 04A, 08B | Junction planning and lineage-backed branch decomposition |
| 5 | 04B, 08C | Closed junction and difference result-shell execution |
| 6 | 07A, 07B | Public runtime API migration followed by docs/package conformance |

Work within a wave may proceed concurrently when its listed prerequisites are
already complete. A later wave may begin per leaf as soon as that leaf's own
prerequisites are complete; the whole preceding wave need not finish first.

## Wave 1: Independent Foundations

### Fix 01A: Preview Watch Request Coordination

- [x] Implement bounded latest-request coordination and watcher delivery behavior.
  - Specification: [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
  - Prerequisites: none
- [x] Wire coordination into the preview watcher and build-scheduler library route consumed by `PyVistaPreviewer`.
- [x] Validate real filesystem delivery, burst coalescing, and scheduler behavior.
  - Test specification: [Fix 01A Test](../test-specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
- [x] Update progression and preview ACD status after route validation.

### Fix 02: Coplanar Loft Face-Touch Union

- [x] Implement coincident contact classification, interior-pair removal, and one-shell union assembly.
  - Specification: [Fix 02](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md)
  - Prerequisites: none
- [x] Wire the corrected executor into the public `boolean_union` library route used by preview and export.
- [x] Validate public union fixtures and composed-enclosure preview/export consumption.
  - Test specification: [Fix 02 Test](../test-specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.

### Fix 03: Named Hole Identity Pairing

- [x] Implement identity-first named-hole pairing with deterministic anonymous fallback and conflict diagnostics.
  - Specification: [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md)
  - Prerequisites: none
- [x] Wire pairing through public loft planning and `Loft(...)` execution.
- [x] Validate crossed-name, duplicate/conflict, and anonymous-control routes.
  - Test specification: [Fix 03 Test](../test-specifications/fix-03-named-hole-identity-pairing-v1_0.md)
- [x] Update progression and loft ACD status after route validation.

### Fix 06: Expanded Planner Configuration Propagation

- [x] Implement immutable propagation of public planner configuration through every expanded planning route.
  - Specification: [Fix 06](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)
  - Prerequisites: none
- [x] Wire options through direct, nested-expansion, and ambiguity-enumeration library routes.
- [x] Validate below-, at-, and above-limit public planner behavior.
  - Test specification: [Fix 06 Test](../test-specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)
- [x] Update progression and loft ACD status after route validation.

### Fix 08A: Loft Difference Trim-Fragment Construction

- [x] Implement intersection-to-trim conversion, fragment construction, provenance, and precise refusal.
  - Specification: [Fix 08A](../specifications/fix-08a-loft-difference-trim-fragment-construction-v1_0.md)
  - Prerequisites: none
- [x] Wire trim-fragment construction into the loft surface-difference executor route.
- [x] Validate trim/fragment fixtures and project cutter references.
  - Test specification: [Fix 08A Test](../test-specifications/fix-08a-loft-difference-trim-fragment-construction-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.

### Fix 09A: Difference Geometry-Change Evidence

- [x] Implement normalized change witnesses and unchanged/ambiguous geometry comparison.
  - Specification: [Fix 09A](../specifications/fix-09a-difference-geometry-change-evidence-v1_0.md)
  - Prerequisites: none
- [x] Wire evidence production into every surfaced difference executor.
- [x] Validate the witness/comparator matrix through registered difference routes.
  - Test specification: [Fix 09A Test](../test-specifications/fix-09a-difference-geometry-change-evidence-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.

## Wave 2: First Derived Contracts

### Fix 01B: Preview Module Cache Invalidation

- [x] Implement generation-based entry/transitive user-module invalidation.
  - Specification: [Fix 01B](../specifications/fix-01b-preview-module-cache-invalidation-v1_0.md)
  - Prerequisite: [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
- [x] Wire forced intent from preview coordination into the CLI scene-factory cache boundary.
- [x] Validate mtime-neutral refresh and dependency rediscovery through the live-preview loader route.
  - Test specification: [Fix 01B Test](../test-specifications/fix-01b-preview-module-cache-invalidation-v1_0.md)
- [x] Update progression and preview ACD status after route validation.

### Fix 01C2A: Preview Current-Generation Scene Apply

- [x] Implement current-generation admission, UI-thread scene apply, and stale/post-shutdown rejection.
  - Specification: [Fix 01C2A](../specifications/fix-01c2a-preview-current-generation-scene-apply-v1_0.md)
  - Prerequisite: [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
- [x] Wire admitted build results into the preview-window renderer-thread state handler.
- [x] Validate generation/state behavior and offscreen scene application.
  - Test specification: [Fix 01C2A Test](../test-specifications/fix-01c2a-preview-current-generation-scene-apply-v1_0.md)
- [x] Update progression and preview ACD status after route validation.

### Fix 05A: Count-Changing Exact Region Pairing

- [x] Implement exact identity pairing plus explicit residual birth/death classification.
  - Specification: [Fix 05A](../specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md)
  - Prerequisite: [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md)
- [x] Wire exact pairing into the public loft-planning route.
- [x] Validate identity-first count-changing planning and conflict diagnostics.
  - Test specification: [Fix 05A Test](../test-specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md)
- [x] Update progression and loft ACD status after route validation.

### Fix 09B: Difference Public Success Gate

- [x] Implement registry-wide success/no-cut classification using normalized geometry-change evidence.
  - Specification: [Fix 09B](../specifications/fix-09b-difference-public-success-gate-v1_0.md)
  - Prerequisite: [Fix 09A](../specifications/fix-09a-difference-geometry-change-evidence-v1_0.md)
- [x] Wire the gate into public `boolean_difference` and every registered surfaced executor.
- [x] Validate public/registry outcomes and the rotated snap-groove false-success regression.
  - Test specification: [Fix 09B Test](../test-specifications/fix-09b-difference-public-success-gate-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.

## Wave 3: Preview Completion And Synthetic Lineage

### Fix 01C1: Preview Refresh Input Wiring

- [x] Implement saved-file and `R` input normalization with preserved forced-refresh intent.
  - Specification: [Fix 01C1](../specifications/fix-01c1-preview-refresh-input-wiring-v1_0.md)
  - Prerequisites: [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md), [Fix 01B](../specifications/fix-01b-preview-module-cache-invalidation-v1_0.md)
- [x] Wire filesystem events and the existing preview-window `R` binding into coordination and cache generations.
- [x] Validate save and key-event routes through controller, CLI callback, and offscreen/real-command smoke.
  - Test specification: [Fix 01C1 Test](../test-specifications/fix-01c1-preview-refresh-input-wiring-v1_0.md)
- [x] Update preview docs, progression, and ACD status after route validation.

### Fix 01C2B: Preview Last-Good Camera And Error State

- [x] Implement camera preservation, last-good scene retention, error display, and recovery behavior.
  - Specification: [Fix 01C2B](../specifications/fix-01c2b-preview-last-good-camera-error-state-v1_0.md)
  - Prerequisite: [Fix 01C2A](../specifications/fix-01c2a-preview-current-generation-scene-apply-v1_0.md)
- [x] Wire failure/recovery state into the preview-window renderer-thread handler.
- [x] Validate camera/error/recovery behavior with offscreen preview failure and recovery smoke.
  - Test specification: [Fix 01C2B Test](../test-specifications/fix-01c2b-preview-last-good-camera-error-state-v1_0.md)
- [x] Update preview docs, progression, and ACD status after route validation.

### Fix 05B: Synthetic Station Identity Lineage

- [x] Implement deterministic synthetic IDs plus predecessor/successor identity lineage.
  - Specification: [Fix 05B](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
  - Prerequisite: [Fix 05A](../specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md)
- [x] Wire lineage-bearing expanded plans into `Loft(...)` surface execution.
- [x] Validate lifecycle records and the rail-pair regression through public loft routes.
  - Test specification: [Fix 05B Test](../test-specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
- [x] Update progression and loft ACD status after route validation.

## Wave 4: Junction And Branch Planning

### Fix 04A: Hole Junction Plan Records

- [x] Implement validated junction direction, lineage, boundary inputs, and stable diagnostics.
  - Specification: [Fix 04A](../specifications/fix-04a-hole-junction-plan-records-v1_0.md)
  - Prerequisites: [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md), [Fix 05B](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
- [x] Wire junction-event records from loft planning into the surface executor boundary.
- [x] Validate birth/death resolution and lifecycle records through planner-consumer routes.
  - Test specification: [Fix 04A Test](../test-specifications/fix-04a-hole-junction-plan-records-v1_0.md)
- [x] Update progression and loft ACD status after route validation.

### Fix 08B: Loft Difference Branch Decomposition

- [x] Implement branch eligibility, bounded decomposition, and a complete recomposition map.
  - Specification: [Fix 08B](../specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md)
  - Prerequisite: [Fix 05B](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
- [x] Wire lineage-backed sub-body cut planning into the difference executor.
- [x] Validate branch fixtures and the audio-cube branched-cutter regression.
  - Test specification: [Fix 08B Test](../test-specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.

## Wave 5: Closed Surface Execution

### Fix 04B: Hole Junction Surface Execution

- [x] Implement junction patches, seam/orientation handling, exact terminal cap count, and closure validation.
  - Specification: [Fix 04B](../specifications/fix-04b-hole-junction-surface-execution-v1_0.md)
  - Prerequisite: [Fix 04A](../specifications/fix-04a-hole-junction-plan-records-v1_0.md)
- [x] Wire junction execution into `Loft(...)` and the published split/merge example route.
- [x] Validate closed `SurfaceBody` output, cap count, and showcase behavior.
  - Test specification: [Fix 04B Test](../test-specifications/fix-04b-hole-junction-surface-execution-v1_0.md)
- [x] Update loft docs, progression, and ACD status after route validation.

### Fix 08C: Loft Difference Result-Shell Reconstruction

- [x] Implement retained-fragment classification, cutter-derived boundaries, seam rebuild, and closed result-shell validation.
  - Specification: [Fix 08C](../specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md)
  - Prerequisites: [Fix 08A](../specifications/fix-08a-loft-difference-trim-fragment-construction-v1_0.md), [Fix 08B](../specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md), [Fix 09B](../specifications/fix-09b-difference-public-success-gate-v1_0.md)
- [x] Wire reconstructed results into public `boolean_difference` and preview/export consumers.
- [x] Validate public cut fixtures plus preview/export consumer smoke with truthful failure behavior.
  - Test specification: [Fix 08C Test](../test-specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md)
- [x] Update surface-boolean docs, progression, and ACD status after route validation.
  - Exact rectangular-loft/axis-aligned-box cuts reconstruct a closed changed
    surface shell; rotated and underconstrained branching candidates remain
    precise no-mesh refusals.

## Wave 6: Public Boolean Contract

### Fix 07A: Surface-Only Boolean Runtime API

- [x] Implement surface-only public signatures, runtime guards, exports, and result types while separating mesh utilities.
  - Specification: [Fix 07A](../specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md)
  - Prerequisites: [Fix 02](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md), [Fix 08C](../specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md), [Fix 09B](../specifications/fix-09b-difference-public-success-gate-v1_0.md)
- [x] Wire the runtime contract through public `impression.modeling` exports and boolean functions.
- [x] Validate the public signature/runtime matrix and actionable mesh-operand errors.
  - Test specification: [Fix 07A Test](../test-specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md)
- [x] Update progression and surface-boolean ACD status after route validation.
  - Documentation, examples, inventory guards, and installed-wheel conformance
    remain owned by the following Fix 07B leaf.

### Fix 07B: Surface Boolean Docs And Package Contract

- [x] Implement documentation, example, inventory-guard, and clean-package conformance for the surface-only API.
  - Specification: [Fix 07B](../specifications/fix-07b-surface-boolean-docs-package-contract-v1_0.md)
  - Prerequisite: [Fix 07A](../specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md)
- [x] Wire the public contract through installed-package docs, tutorials, and examples.
- [x] Validate documentation assertions and clean-wheel smoke against the runtime API.
  - Test specification: [Fix 07B Test](../test-specifications/fix-07b-surface-boolean-docs-package-contract-v1_0.md)
- [x] Update release docs, progression, and surface-boolean ACD status after route validation.
  - All 19 canonical implementation leaves and their paired test contracts are
    now complete; release-candidate qualification remains a separate release
    gate.

## Specification Canonicalization

- [x] Split every parent or umbrella spec required by the scoring policy.
- [x] Verify 100% parent responsibility coverage.
- [x] Move all uncovered responsibilities into children and re-verify.
- [x] Mark the 19 retained children/leaves canonical after coverage reached 100%.
- [x] Archive the eight superseded parent/intermediate specs.
- [x] Update indexes and this progression to reference canonical children only.
- [x] Record completed refinement in the request-scoped ledger and ACD conformance sections.
