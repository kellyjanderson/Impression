# v1.0.0a4 Known-Issue Intake

Date: 2026-08-04
Status: Canonical release scope

## Evidence Boundary

All included items are open GitHub issues and were reproduced against installed
Impression `1.0.0a3`. The diagonal audio-cube reproduction ledger and script are
the source geometry evidence. GitHub remains the issue tracker; this document
records release disposition and canonical specification coverage.

## Included Issues

| Issue | Confirmed failure | Canonical implementation leaf or leaves |
| --- | --- | --- |
| [#242](https://github.com/kellyjanderson/Impression/issues/242) | Missed a3 obligation: watcher delivery is slow and `R` does not guarantee module invalidation | [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md), [Fix 01B](../specifications/fix-01b-preview-module-cache-invalidation-v1_0.md), [Fix 01C1](../specifications/fix-01c1-preview-refresh-input-wiring-v1_0.md), [Fix 01C2A](../specifications/fix-01c2a-preview-current-generation-scene-apply-v1_0.md), [Fix 01C2B](../specifications/fix-01c2b-preview-last-good-camera-error-state-v1_0.md) |
| [#243](https://github.com/kellyjanderson/Impression/issues/243) | Coplanar loft union retains overlapping shells and is rejected as invalid | [Fix 02](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md) |
| [#244](https://github.com/kellyjanderson/Impression/issues/244) | Named hole paths are preserved but ignored during pairing | [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md) |
| [#245](https://github.com/kellyjanderson/Impression/issues/245) | Hole split/merge emits an internal closure cap and a non-closed body | [Fix 04A](../specifications/fix-04a-hole-junction-plan-records-v1_0.md), [Fix 04B](../specifications/fix-04b-hole-junction-surface-execution-v1_0.md) |
| [#246](https://github.com/kellyjanderson/Impression/issues/246) | Count-changing expansion loses exact identities and resets caller limits | [Fix 05A](../specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md), [Fix 05B](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md), [Fix 06](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md) |
| [#247](https://github.com/kellyjanderson/Impression/issues/247) | Public surfaced booleans advertise and return mesh modeling types | [Fix 07A](../specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md), [Fix 07B](../specifications/fix-07b-surface-boolean-docs-package-contract-v1_0.md) |
| [#248](https://github.com/kellyjanderson/Impression/issues/248) | Branched loft cutters are refused; accepted cuts can return unchanged geometry | [Fix 08A](../specifications/fix-08a-loft-difference-trim-fragment-construction-v1_0.md), [Fix 08B](../specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md), [Fix 08C](../specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md), [Fix 09A](../specifications/fix-09a-difference-geometry-change-evidence-v1_0.md), [Fix 09B](../specifications/fix-09b-difference-public-success-gate-v1_0.md) |

## Coverage Decisions

- #246 is split because preserving identity through synthetic station creation
  changes topology data ownership, while propagating `ambiguity_max_branches`
  changes planner configuration plumbing. Either can regress independently.
- #248 is split because constructing cut geometry is a kernel execution change,
  while refusing unchanged success is a public result-validation invariant that
  protects every difference route.
- #243 remains separate from #248. Coincident face-touch union removes an
  opposite-oriented interior patch pair; difference reconstructs trimmed
  fragments and cutter-derived caps.
- #247 follows successful surface-only correction work so implementation does
  not remove compatibility before the replacement public routes pass their
  release fixtures.

## Release Sequencing Constraints

1. Fixes 03, 05A, 05B, and 06 establish deterministic topology identity and
   planning configuration before Fix 04A and Fix 04B validate split/merge
   junction planning and execution.
2. Fix 02 establishes coincident-contact classification and shell merge
   validation before the broader cut work.
3. Fix 09A and Fix 09B land before Fix 08C is claimed complete so every new cut
   route is protected by the no-op success gate.
4. Fix 07A and Fix 07B land after Fixes 02, 08A-08C, and 09A-09B demonstrate
   that public surfaced routes replace the legacy mesh-shaped contract.
5. Fixes 01A, 01B, 01C1, 01C2A, and 01C2B form an independent preview lane.

## Progression Rule

The 19 linked leaves are the only implementation anchors. The completed
fixed-point review verified 100% responsibility coverage; the final progression
must preserve their prerequisites and must not route archived split parents.
