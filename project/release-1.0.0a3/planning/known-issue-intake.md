# v1.0.0a3 Known-Issue Intake

Date: 2026-08-04
Status: Release scope decision

## Evidence Boundary

Included items below were reproduced in the test-modeling project, reported in a
live GitHub issue, or confirmed against `origin/main` source on 2026-08-04. Local
review notes were treated as leads and included only when the current source
exposed a bounded corrective change. This document records scope; GitHub remains
the issue tracker.

## Included In v1.0.0a3

| ID | Known issue | Current evidence | Planned leaf |
| --- | --- | --- | --- |
| A3-01 | Named `TopologyPath` correspondences cannot be passed to `Loft(...)` without losing identity | `testingImp/references/impression-issues.md`; `Loft` normalizes through `as_section` | [Fix 01](../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md) |
| A3-02 | A protected diagonal corner can disappear or drift after loft tessellation | `testingImp/models/audio_cube_diagonal_halves.py`; output changes with sample count | [Fix 02](../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md) |
| A3-03 | Stable multi-region stations exceed the 64-branch ambiguity limit despite explicit identities | test-modeling issue list; `ambiguity_max_branches=64` candidate enumeration | [Fix 03](../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md) |
| A3-04 | Coplanar loft-body fusion can collapse an earlier enclosure body | test-modeling issue list and grouped-body workaround | [Fix 04](../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md) |
| A3-05 | A wall loft with multiple openings can produce louver-like faces and hundreds of degenerates | test-modeling issue list and solid-wall-plus-cuts workaround | [Fix 05](../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md) |
| A3-06 | The experimental half-pipe example, adapter, and `build123d` runtime dependency remain on `main` | `examples/half_pipe.py`, `src/impression/cad.py`, `pyproject.toml` | [Fix 06](../specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md) |
| A3-07 | Linux headless reference-review UI tests can hang or exit 139 | GitHub issue [#227](https://github.com/kellyjanderson/Impression/issues/227) | [Fix 07](../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md) |
| A3-08 | Documentation ZIP extraction accepts untrusted member paths without containment validation | `src/impression/cli.py::_extract_docs_archive` | [Fix 08](../specifications/fix-08-safe-docs-archive-extraction-v1_0.md) |
| A3-09 | User-model loading deletes and reloads `impression.modeling` modules, allowing split class identities | `src/impression/cli.py` module cleanup/load path | [Fix 09](../specifications/fix-09-user-model-loader-module-identity-v1_0.md) |
| A3-10 | Normal scene collection recognizes mesh/polyline payloads but not a first-class `SurfaceBody` result | `src/impression/preview.py::_collect_datasets_from_scene` | [Fix 10](../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md) |
| A3-11 | STL export can write a non-watertight or degenerate result without a manufacturing integrity gate | `src/impression/cli.py::export` and `write_stl` path | [Fix 11](../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md) |
| A3-12 | Documentation-rule tests assert retired `agents/` and `project/agents/` paths | `tests/test_documentation_rules.py` | [Fix 12](../specifications/fix-12-documentation-policy-test-migration-v1_0.md) |
| A3-13 | Tag workflow publishes untested artifacts and does not explicitly mark alpha releases as prereleases | `.github/workflows/release.yml` | [Fix 13A](../specifications/fix-13a-release-artifact-build-qualification-v1_0.md) and [Fix 13B](../specifications/fix-13b-qualified-prerelease-publication-v1_0.md) ([split parent](../specifications/fix-13-release-workflow-artifact-qualification-v1_0.md)) |

## Accounted For But Deferred

| Candidate | Disposition |
| --- | --- |
| Broad public API curation and compatibility policy | Architectural/product scope larger than one corrective leaf; file a dedicated post-a3 issue after API inventory. |
| Comprehensive migration of every scene consumer to surface-first payloads | A3-10 fixes and proves the primary preview/export route only; remaining consumers need an explicit inventory and separate progression. |
| Full units and tolerance semantic redesign | Cross-cutting compatibility work; not required to correct the confirmed a3 failures. |
| General cancellation and concurrency architecture | Larger than the Linux lifecycle defect in A3-07. |
| Text topology/complexity rewrite and higher-order surface continuity | Feature programs, not alpha hotfixes. |
| Complete interchange/manufacturing-validation program | A3-11 adds a minimum refusal gate; richer repair and interchange behavior remains separate. |
| Atomic reference promotion, mirror-orientation policy, and package-import mutation cleanup | Require independent reproduction and sizing before commitment. |
| General code-quality, coverage-architecture, generated-code, and governance cleanup | Ongoing maintenance programs; they are not release claims for a3. |

## Recently Corrected; Not Reopened

| Item | Disposition |
| --- | --- |
| Installer references to the retired hinge modules | Corrected and released in `v1.0.0a2`; retained as an installed-artifact regression check. |
| Accidentally merged SDF experiment | Extracted before a2; absence remains a package-content regression check. |
| a2 release record and prerelease correction | Complete; A3-13 makes the metadata behavior automatic. |

## Change-Control Rule

A newly discovered issue may enter a3 only when it is reproduced, corrective
rather than expansive, normalized and rescored with the current implementation
template, below the forced-split threshold as a cohesive leaf, supplied with a
paired test specification, and added to this intake and progression. Otherwise it
is logged for the next release instead of silently enlarging a3.
