# Impression v1.0.0a4 Corrective Release Definition

Date: 2026-08-04
Status: Proposed
Issue set: [#242](https://github.com/kellyjanderson/Impression/issues/242) through [#248](https://github.com/kellyjanderson/Impression/issues/248)
Base release: `v1.0.0a3`

## Intent

`v1.0.0a4` is a corrective alpha release that makes live preview iteration
responsive and closes the surface-first loft and CSG failures exposed by the
diagonal audio-cube test model. The release does not broaden Impression with a
new modeling family. It makes already-public preview, loft, and boolean routes
honor their documented contracts.

The watcher and forced-refresh correction is a carried release obligation from
`v1.0.0a3`. It was expected to ship there, remains broken in the published a3
behavior, and is therefore mandatory scope for a4 rather than a discretionary
preview enhancement.

## User-Visible Outcomes

- `impression preview` notices top-level and transitive model changes without a
  watcher-added delay, and `R` is a definitive cache-invalidating refresh.
- Named holes and named regions retain identity across loft stations, including
  count-changing transitions and synthetic planning stations.
- Hole split/merge lofts produce closed surface bodies with valid junctions and
  exactly the two requested terminal caps.
- Coplanar loft-body union produces one validated shell instead of retaining
  overlapping shells or returning a partial result.
- Surface difference produces real USB, acoustic, and snap-pocket cuts when the
  route reports success; unchanged geometry cannot be labeled successful.
- Public modeling booleans accept and return surfaced modeling types only.

## Architecture Transitions

- [Preview Reload Coordination ACD](architecture/acd-preview-reload-coordination.md)
- [Loft Identity And Junction Correctness ACD](architecture/acd-loft-identity-and-junction-correctness.md)
- [Surface Boolean Correctness And API Boundary ACD](architecture/acd-surface-boolean-correctness-and-api-boundary.md)

These ACDs describe desired architecture that is not yet true in `v1.0.0a3`.
Canonical architecture remains unchanged until implementation conforms and an
explicit reconciliation pass closes the ACDs.

## Canonical Specification Set

Independent fixed-point review produced 19 canonical implementation leaves and
19 paired canonical test specifications, indexed under
[Specifications](specifications/README.md) and
[Test Specifications](test-specifications/README.md). Eight superseded parent
specifications remain archived for split provenance.

The issue-to-leaf disposition is recorded in
[Known-Issue Intake](planning/known-issue-intake.md).

## Release Gates

The release candidate may be tagged only after:

1. Independent `review specs` has reached a fixed point, assigned valid Review
   Scores, and canonicalized every retained leaf; the final progression must
   route only these canonical leaves.
2. Every canonical implementation leaf and paired test leaf is complete.
3. The real filesystem watcher route meets the specified latency budget on a
   supported local filesystem, and the visible `R` route forces a fresh build.
   Failure of either behavior blocks a4 publication.
4. The complete test-model reproduction suite succeeds without the documented
   grouped-body, topology-native notch, separated-rail, or flat-rim workarounds.
5. Every successful surface boolean result passes closure, seam, operand-witness,
   and geometry-change validation with no hidden mesh fallback.
6. Public API signature, documentation, examples, and installed-package smoke
   tests agree on the surface-only boolean contract.
7. The full configured test suite passes on supported macOS and Linux lanes.
8. Wheel, source distribution, and documentation archive are built once from
   the candidate commit and pass clean-install qualification before prerelease
   publication.

## Exclusions

- No general-purpose replacement for every higher-order surface intersection
  family beyond the exact loft fixtures in this release.
- No preview UI redesign or new keybinding family.
- No mesh repair, mesh modeling fallback, or implicit conversion of mesh
  operands into surface bodies.
- No probabilistic loft matching expansion; authored identities remain primary
  and geometric matching remains the fallback for unnamed residue.
- No claim that the broader `0.1.0a` CSG completion program is finished.

## Planning State

This release has 19 cohesive canonical leaves with 100% issue responsibility
coverage. The next planning artifact is a dependency-ordered implementation
progression over only those leaves; review did not create it implicitly.

Exact next workflow action: create the final a4 implementation progression.
