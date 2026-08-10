# Release Timing History

Date created: 2026-08-09
Current release: `v1.0.0a5`
Status: Active

## Purpose

Record wall-clock duration, outcome, and wait behavior for local qualification,
GitHub CI, build, publication, and post-publication verification. Future release
runs use this history instead of arbitrary polling intervals.

## Wait Policy

- For `v1.0.0a5`, GitHub PR/Actions polls wait exactly 150 seconds between
  non-terminal observations, as required for this release.
- For later releases, group successful observations by task category and use
  `ceil(P75 duration / 30) * 30` seconds as the initial wait.
- Keep future waits between 30 and 300 seconds. With fewer than three successful
  observations in a category, retain the latest explicit release instruction,
  otherwise default to 150 seconds.
- After the initial wait, poll at the same category interval until terminal.
  Record every wait and the terminal observation so the next estimate improves.
- Failed and interrupted runs remain in the ledger but do not set the success
  percentile; they inform timeout and test-design decisions.

## Current Recommendations

| Category | Successful samples | Recommended wait | Basis |
|---|---:|---:|---|
| GitHub PR CI | 0 | 150 s | Explicit v1.0.0a5 instruction; insufficient history |
| GitHub release workflow | 0 | 150 s | Explicit v1.0.0a5 instruction; insufficient history |
| Local focused CSG test | 2 | 150 s max command allowance | Successful runs were 3.32 s and 15.41 s; retain conservative command allowance until three samples |
| Full configured suite | 1 | 150 s observation interval | Successful current-release run was 440.69 s; retain explicit 150 s observation interval until three samples |
| Candidate build/qualification | 2 | 150 s observation interval | Initial and final local qualifications completed in 19.64 s and 18.98 s; retain explicit 150 s observation interval until three samples |

## v1.0.0a5 Ledger

| Phase | Started | Duration | Outcome | Evidence / note |
|---|---|---:|---|---|
| Fetch current origin state | 2026-08-09 22:14:30 PDT | 1 s | Passed | `origin/main` advanced to `91ca617` |
| Query latest GitHub releases | 2026-08-09 22:14:43 PDT | 1 s | Passed | `v1.0.0a4` confirmed latest prerelease |
| Direct feature-branch switch attempt | 2026-08-09 22:14:52 PDT | <1 s | Safely refused | Existing spec changes prevented checkout; no data lost |
| Preserve changes and create exact-main branch | 2026-08-09 22:14:59 PDT | 1 s | Passed | Created `codex/issues-267-268-a5-release` at `origin/main` |
| Reapply specification work | 2026-08-09 22:15:07 PDT | 1 s | Passed | Named stash applied cleanly |
| Baseline issue reproduction | 2026-08-09 22:15:36 PDT | 1 s | Expected failures | Both unions and second sequential groove refused on unchanged main |
| Post-fix issue reproduction | 2026-08-09 | 0.89 s | Passed | Both unions, batch, first cut, and second cut succeeded |
| Focused coverage attempt 1 | 2026-08-09 | 16.91 s | Failed | Two frame-dependent point assumptions in new tests |
| Export-quality union test attempt | 2026-08-09 | 164.72 s | Interrupted | Export tessellation was unsuitable for every focused unit run; retained for release artifact phase |
| Corrected focused test | 2026-08-09 | 3.06 s | Failed | Preview tessellation is intentionally not a watertightness proof |
| Focused coverage acceptance | 2026-08-09 | 15.41 s | Passed | 3 passed; XML and HTML coverage refreshed |
| Six-step self-contained smoke | 2026-08-09 | 3.32 s | Passed | 3 selected tests passed, including six re-entries |
| Six-step sibling acceptance | 2026-08-09 | 0.40 s | Passed | Six succeeded/closed/changed public results |
| Initial balanced attached-union STL | 2026-08-09 | 11.09 s | Rejected evidence | Open marching-cell facets; artifact not promoted |
| Initial balanced attached-union PNG | 2026-08-09 | 2.53 s | Rejected evidence | Visual inspection exposed marching-cell planes |
| Initial balanced six-groove STL | 2026-08-09 | 23.05 s | Rejected evidence | Open marching-cell facets; artifact not promoted |
| Initial balanced six-groove PNG | 2026-08-09 | 3.78 s | Rejected evidence | Visual inspection exposed marching-cell planes |
| Fine attached-union export gate | 2026-08-09 | about 145 s | Failed | Correctly raised non-watertight closed-body error before terminal extractor fix |
| Terminal extractor focused regression | 2026-08-09 | 1.95 s | Passed | 3 passed; union and six-cut export meshes watertight/manifold |
| Final attached-union STL generation | 2026-08-09 | 0.05 s | Passed | Export-fine terminal polygon-loft graph extraction |
| Final attached-union PNG generation | 2026-08-09 | 0.44 s | Passed | Visually inspected; attached feature visible |
| Final six-groove STL generation | 2026-08-09 | 0.09 s | Passed | Export-fine terminal polygon-loft graph extraction |
| Final six-groove PNG generation | 2026-08-09 | 0.13 s | Passed | Visually inspected; all six grooves visible |
| Final dirty STL manifold inspection | 2026-08-09 | 0.2 s | Passed | Union: 44 faces/0 open edges; grooves: 132 faces/0 open edges |
| Broader surface-CSG coverage attempt | 2026-08-09 | 34.74 s | Failed | One established two-body loft route was displaced by the new field route |
| Two-body compatibility regression smoke | 2026-08-09 | 1.58 s | Passed | 6 selected route tests passed after narrowing union to 3+ operands |
| Broader surface-CSG coverage acceptance | 2026-08-09 | 33.15 s | Passed | 262 passed; focused XML/HTML coverage refreshed |
| Complete configured coverage suite | 2026-08-09 | 440.69 s | Passed | 1,785 passed; 82.9% branch-aware coverage; canonical XML/HTML refreshed |
| Release metadata/workflow regression | 2026-08-09 | 0.5 s | Passed | 15 passed; package/runtime version reported `1.0.0a5` |
| Initial package build | 2026-08-09 | 1.88 s | Passed | Built `1.0.0a5` wheel and sdist; existing setuptools license deprecation warning only |
| Initial docs archive build | 2026-08-09 | 0.14 s | Passed | Built `impression-docs-v1.0.0a5.zip` |
| Initial fresh wheel qualification | 2026-08-09 | 8.55 s | Passed | Installed exact wheel; export/docs/version smoke and `pip check` passed |
| Initial fresh sdist qualification | 2026-08-09 | 8.96 s | Passed | Installed exact sdist; export/docs/version smoke and `pip check` passed |
| Initial immutable manifest verification | 2026-08-09 | 0.11 s | Passed | Wheel, sdist, docs archive, tag, prerelease state, versions, and hashes verified |
| Post-documentation spec/release regression | 2026-08-09 | 0.5 s | Passed | 25 spec, documentation, release metadata, and workflow tests passed |
| Final package rebuild | 2026-08-09 | 2.11 s | Passed | Rebuilt exact post-documentation `1.0.0a5` wheel and sdist |
| Final docs archive build | 2026-08-09 | 0.14 s | Passed | Rebuilt post-documentation `impression-docs-v1.0.0a5.zip` |
| Final fresh wheel qualification | 2026-08-09 | 7.98 s | Passed | Installed exact final wheel; export/docs/version smoke and `pip check` passed |
| Final fresh sdist qualification | 2026-08-09 | 8.64 s | Passed | Installed exact final sdist; export/docs/version smoke and `pip check` passed |
| Final immutable manifest verification | 2026-08-09 | 0.11 s | Passed | Final wheel, sdist, docs archive, tag, prerelease state, versions, and hashes verified |

## Publication Ledger

This section will be completed with PR CI, merge, tag workflow, build jobs,
publication, downloads, hash verification, and fresh-install durations before
the release is considered closed.
