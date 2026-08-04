# v1.0.0a3 Fix Release Progression

Date: 2026-08-04
Status: In Progress

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
- [x] [Fix 14: Archive Retired Modeling Experiments](../specifications/fix-14-archive-retired-modeling-experiments-v1_0.md)
- [x] [Fix 14 Test](../test-specifications/fix-14-archive-retired-modeling-experiments-v1_0.md)
- [x] [Fix 07: Reference Review Linux Lifecycle](../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)
- [x] [Fix 07 Test](../test-specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)

## Runtime, Preview, And Export

- [x] [Fix 09: User-Model Loader Module Identity](../specifications/fix-09-user-model-loader-module-identity-v1_0.md)
- [x] [Fix 09 Test](../test-specifications/fix-09-user-model-loader-module-identity-v1_0.md)
- [x] [Fix 10: SurfaceBody Preview and Export Consumption](../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)
- [x] [Fix 10 Test](../test-specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)
- [x] [Fix 11: Export Manufacturing Integrity Gate](../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)
- [x] [Fix 11 Test](../test-specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)

### Fix 15: Preview PNG Export

- [x] Implement one-shot off-screen PNG capture in the existing preview renderer.
  - Specification: [Fix 15](../specifications/fix-15-preview-png-export-v1_0.md)
- [x] Wire `preview --screenshot PATH` to bypass watched-preview control-file handoff.
  - Specification: [Fix 15](../specifications/fix-15-preview-png-export-v1_0.md)
- [x] Validate the installed CLI route with a live preview control file present and inspect the generated PNG.
  - Test specification: [Fix 15 Test](../test-specifications/fix-15-preview-png-export-v1_0.md)
- [x] Update CLI documentation and mark Fix 15 status only after route validation.

### Fix 16: Preview Sharp-Edge Normals

- [x] Add sharp-edge vertex splitting to smooth uniform-color and per-face-color preview actors.
  - Specification: [Fix 16](../specifications/fix-16-preview-sharp-edge-normals-v1_0.md)
- [x] Route the split threshold through the existing preview feature-edge angle and disable it for flat shading.
  - Specification: [Fix 16](../specifications/fix-16-preview-sharp-edge-normals-v1_0.md)
- [x] Render and inspect the original, loft, and diagonal audio-cube assemblies through `preview --screenshot`.
  - Test specification: [Fix 16 Test](../test-specifications/fix-16-preview-sharp-edge-normals-v1_0.md)
- [x] Update release documentation and progression after the three-model route passes.

### Fix 17: Pointed Cone Shell Connectivity

- [x] Correct pointed-cone connectivity without changing frustum cap-adjacency status.
  - Specification: [Fix 17](../specifications/fix-17-pointed-cone-shell-connectivity-v1_0.md)
- [x] Cover both apex orientations and the two-radius frustum contrast.
  - Test specification: [Fix 17 Test](../test-specifications/fix-17-pointed-cone-shell-connectivity-v1_0.md)
- [x] Confirm the corrected constructor in the terminal full release suite.

## Test-Modeling Loft And Boolean Corrections

- [x] [Fix 01: TopologyPath Loft Input Preservation](../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)
- [x] [Fix 01 Test](../test-specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)
- [x] [Fix 02: Protected Loft Corner Tessellation](../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)
- [x] [Fix 02 Test](../test-specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)
- [x] [Fix 03: Identity-First Stable Region Pairing](../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)
- [x] [Fix 03 Test](../test-specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)
- [x] [Fix 05: Multi-Opening Loft Wall Integrity](../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)
- [x] [Fix 05 Test](../test-specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)
- [x] [Fix 04: Coplanar Loft-Body Union Outcome](../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)
- [x] [Fix 04 Test](../test-specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)

## Final Artifact Qualification And Publication Readiness

- [x] Hydrate Git LFS reference artifacts in every release job checkout.
  - Specification: [Fix 13A](../specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [x] Install the Linux font used by text and loft qualification tests and provide the default Arial-compatible fallback.
  - Specification: [Fix 13A](../specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [x] Make the preview PNG help assertion independent of CI terminal width and ANSI rendering.
  - Test specification: [Fix 13A Test](../test-specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [x] Run exact serialized-reference comparisons on macOS while retaining Linux build and integration CI.
  - Specification: [Fix 13A](../specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
- [ ] Pass the complete tag-triggered test and artifact-qualification jobs.
  - Test specification: [Fix 13A Test](../test-specifications/fix-13a-release-artifact-build-qualification-v1_0.md)
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
| 14 | 13 | Cohesive leaf: archive-before-removal transaction for retired experiment ownership |
| 15 | 16 | Retain after explicit split review: help, routing, and rendering are one `--screenshot` command contract |
| 16 | 10 | Cohesive leaf: one shared preview actor-normal policy |
| 17 | 5.5 | Cohesive leaf: one primitive connectivity signal |

- Forced splits (`25+`): Fix 13, completed into Fix 13A and Fix 13B.
- Readiness blockers and unresolved parent coverage: none.
- Original release set terminal review: [ledger](spec-review-ledger-20260804-040607.md) pass 2, new leaves `none`.
- Fix 14 terminal review: [ledger](spec-review-ledger-20260804-071535.md) pass 1, new leaves `none`.
- Fix 15 terminal review: [ledger](spec-review-ledger-20260804-preview-png.md) pass 1, new leaves `none`.
- Fix 15 help-surface review: [ledger](spec-review-ledger-20260804-preview-png-help.md) pass 1, new leaves `none`.
- Fix 16 terminal review: [ledger](spec-review-ledger-20260804-preview-sharp-edges.md) pass 1, new leaves `none`.
- Fix 17 terminal review: [ledger](spec-review-ledger-20260804-pointed-cone.md) pass 1, new leaves `none`.
