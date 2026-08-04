# v1.0.0a4 Known-Issue Intake

Date: 2026-08-04
Status: Draft release scope

## Evidence Boundary

All included items are open GitHub issues and were reproduced against installed
Impression `1.0.0a3`. The diagonal audio-cube reproduction ledger and script are
the source geometry evidence. GitHub remains the issue tracker; this document
records release disposition and draft specification coverage.

## Included Issues

| Issue | Confirmed failure | Draft implementation leaf or leaves |
| --- | --- | --- |
| [#242](https://github.com/kellyjanderson/Impression/issues/242) | Missed a3 obligation: watcher delivery is slow and `R` does not guarantee module invalidation | [Fix 01](../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md) |
| [#243](https://github.com/kellyjanderson/Impression/issues/243) | Coplanar loft union retains overlapping shells and is rejected as invalid | [Fix 02](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md) |
| [#244](https://github.com/kellyjanderson/Impression/issues/244) | Named hole paths are preserved but ignored during pairing | [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md) |
| [#245](https://github.com/kellyjanderson/Impression/issues/245) | Hole split/merge emits an internal closure cap and a non-closed body | [Fix 04](../specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md) |
| [#246](https://github.com/kellyjanderson/Impression/issues/246) | Count-changing expansion loses exact identities and resets caller limits | [Fix 05](../specifications/fix-05-count-changing-region-identity-preservation-v1_0.md), [Fix 06](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md) |
| [#247](https://github.com/kellyjanderson/Impression/issues/247) | Public surfaced booleans advertise and return mesh modeling types | [Fix 07](../specifications/fix-07-surface-only-public-boolean-api-v1_0.md) |
| [#248](https://github.com/kellyjanderson/Impression/issues/248) | Branched loft cutters are refused; accepted cuts can return unchanged geometry | [Fix 08](../specifications/fix-08-loft-surface-difference-cut-execution-v1_0.md), [Fix 09](../specifications/fix-09-surface-difference-no-op-result-gate-v1_0.md) |

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

1. Fixes 03, 05, and 06 establish deterministic topology identity and planning
   configuration before Fix 04 validates split/merge junction execution.
2. Fix 02 establishes coincident-contact classification and shell merge
   validation before the broader cut work.
3. Fix 09 lands before Fix 08 is claimed complete so every new cut route is
   protected by the no-op success gate.
4. Fix 07 lands after Fixes 02, 08, and 09 demonstrate that public surfaced
   routes replace the legacy mesh-shaped contract.
5. Fix 01 is independent and may proceed in parallel after spec review.

## Progression Rule

No draft leaf in this intake is an implementation anchor. `review specs` must
rescore, split if required, verify 100% responsibility coverage, and mark final
canonical leaves before a progression is created.
