# v1.0.0a3 Fix Release Progression

Date: 2026-08-04
Status: Planned

Only canonical implementation leaves that passed the current Review Score and
their paired verification leaves appear here. A feature and its test leaf have
separate completion state.

## Release Hygiene And Safety

- [x] [Fix 06: Remove Accidental Half-Pipe Release Payload](../specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)
- [x] [Fix 06 Test](../test-specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)
- [x] [Fix 12: Documentation Policy Test Migration](../specifications/fix-12-documentation-policy-test-migration-v1_0.md)
- [x] [Fix 12 Test](../test-specifications/fix-12-documentation-policy-test-migration-v1_0.md)
- [x] [Fix 08: Safe Documentation Archive Extraction](../specifications/fix-08-safe-docs-archive-extraction-v1_0.md)
- [x] [Fix 08 Test](../test-specifications/fix-08-safe-docs-archive-extraction-v1_0.md)
- [ ] [Fix 07: Reference Review Linux Lifecycle](../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)
- [ ] [Fix 07 Test](../test-specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)

## Runtime, Preview, And Export

- [x] [Fix 09: User-Model Loader Module Identity](../specifications/fix-09-user-model-loader-module-identity-v1_0.md)
- [x] [Fix 09 Test](../test-specifications/fix-09-user-model-loader-module-identity-v1_0.md)
- [x] [Fix 10: SurfaceBody Preview and Export Consumption](../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)
- [x] [Fix 10 Test](../test-specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)
- [ ] [Fix 11: Export Manufacturing Integrity Gate](../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)
- [ ] [Fix 11 Test](../test-specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)

## Test-Modeling Loft And Boolean Corrections

- [ ] [Fix 01: TopologyPath Loft Input Preservation](../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)
- [ ] [Fix 01 Test](../test-specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)
- [ ] [Fix 02: Protected Loft Corner Tessellation](../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)
- [ ] [Fix 02 Test](../test-specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)
- [ ] [Fix 03: Identity-First Stable Region Pairing](../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)
- [ ] [Fix 03 Test](../test-specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)
- [ ] [Fix 05: Multi-Opening Loft Wall Integrity](../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)
- [ ] [Fix 05 Test](../test-specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)
- [ ] [Fix 04: Coplanar Loft-Body Union Outcome](../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)
- [ ] [Fix 04 Test](../test-specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)

## Final Artifact Qualification And Publication Readiness

- [ ] [Fix 13A: Release Artifact Build and Qualification](../specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [ ] [Fix 13A Test](../test-specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [ ] [Fix 13B: Qualified Prerelease Publication](../specifications/fix-13b-qualified-prerelease-publication-v1_0.md)
- [ ] [Fix 13B Test](../test-specifications/fix-13b-qualified-prerelease-publication-v1_0.md)

## Review Score Summary

| Fix | Score | Split decision |
|---:|---:|---|
| 01 | 12.5 | Cohesive leaf |
| 02 | 14 | Cohesive leaf |
| 03 | 12.5 | Cohesive leaf |
| 04 | 13 | Cohesive leaf |
| 05 | 17 | Retain after explicit split review: one loft-output transaction |
| 06 | 8.5 | Cohesive leaf |
| 07 | 19 | Retain after explicit split review: one GUI process lifecycle |
| 08 | 19 | Retain after explicit split review: one atomic extraction boundary |
| 09 | 14.5 | Cohesive leaf |
| 10 | 23 | Retain after explicit split review: one shared consumer adapter with two route proofs |
| 11 | 22.5 | Retain after explicit split review: one validated atomic export transaction |
| 12 | 12.5 | Cohesive leaf |
| 13 parent | 26 | Forced split; 100% covered and superseded |
| 13A | 22 | Retain after explicit split review: one artifact qualification transaction |
| 13B | 16 | Retain after explicit split review: one external publication transaction |

- Forced splits (`25+`): Fix 13, completed into Fix 13A and Fix 13B.
- Readiness blockers and unresolved parent coverage: none.
- Terminal review pass: [ledger](spec-review-ledger-20260804-040607.md) pass 2, new leaves `none`.
