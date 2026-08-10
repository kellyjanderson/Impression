# Release Timing History

Date created: 2026-08-09
Current release: `v1.0.0a5`
Status: Complete

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
| GitHub PR CI | 2 | 150 s | Release PR cycles completed in 579 s and 464 s; fewer than three samples retains the explicit interval |
| GitHub release workflow | 1 | 150 s | Tag workflow completed in 641 s; fewer than three samples retains the explicit interval |
| Local focused CSG test | 4 | 30 s | Successful runs were 1.95 s, 3.32 s, 15.41 s, and 33.15 s; P75 rounded up to 30 s |
| Full configured suite | 4 | 300 s | Successful local/CI observations were 440.69 s, 464 s, 519 s, and 579 s; rounded P75 exceeds the 300 s policy cap |
| Candidate build/qualification | 3 | 60 s | Successful aggregate qualifications were 18.98 s, 19.64 s, and 80 s; interpolated P75 rounded up to 60 s |

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
| Ledger-freeze docs archive rebuild | 2026-08-09 | 0.15 s | Passed | Included the frozen pre-publication timing ledger |
| Ledger-freeze installed-candidate smoke | 2026-08-09 | 0.58 s | Passed | Final docs archive extracted through installed wheel smoke |

## Publication Ledger

| Phase | Started | Duration | Outcome | Evidence / note |
|---|---|---:|---|---|
| Feature branch push | 2026-08-09 23:02 PDT | 3.41 s | Passed | Commit `5f45ec1`; LFS references uploaded |
| Open release PR #269 | 2026-08-09 23:02 PDT | 2.0 s | Passed | PR opened mergeable against exact `origin/main` base |
| PR #269 first `build-test` | 2026-08-09 23:02 PDT | 118 s | Passed | GitHub Actions run `31360522550` |
| PR #269 first `candidate-suite` | 2026-08-09 23:02 PDT | 579 s | Passed | GitHub Actions run `31360522550` |
| PR #269 first CI observation waits | 2026-08-09 23:02 PDT | 4 x 150 s | Passed | Polled only after exact instructed intervals until terminal green state |
| Timing evidence push | 2026-08-09 23:14 PDT | 2.11 s | Passed | Commit `34cc1ac` triggered final release-PR CI cycle |
| PR #269 final `build-test` | 2026-08-09 23:14 PDT | 120 s | Passed | GitHub Actions run `31361211840` |
| PR #269 final `candidate-suite` | 2026-08-09 23:14 PDT | 464 s | Passed | GitHub Actions run `31361211840` |
| PR #269 final CI observation waits | 2026-08-09 23:14 PDT | 3 x 150 s | Passed | Polled only after exact instructed intervals until terminal green state |
| Merge release PR #269 | 2026-08-09 23:22 PDT | 5.23 s | Passed | Merged to `main` as `96d80a8` |
| Push annotated tag `v1.0.0a5` | 2026-08-09 23:22 PDT | 2.56 s | Passed | Tag points to merge commit `96d80a8` |
| Release workflow `test` job | 2026-08-09 23:23 PDT | 519 s | Passed | Full candidate-suite step took 473 s |
| Release workflow `qualify` job | 2026-08-09 23:31 PDT | 80 s | Passed | Build 8 s; wheel smoke 21 s; sdist smoke 20 s; manifest/upload completed |
| Release workflow `publish` job | 2026-08-09 23:33 PDT | 33 s | Passed | Manifest, assets, release publication, and metadata verification passed |
| Complete release workflow | 2026-08-09 23:23 PDT | 641 s | Passed | GitHub Actions run `31361761993` |
| Release-workflow observation waits | 2026-08-09 23:23 PDT | 4 x 150 s | Passed | Polled only after exact instructed intervals until terminal success |
| Query live prerelease metadata | 2026-08-09 23:34 PDT | 0.3 s | Passed | Non-draft prerelease with exact three-asset set |
| Download live release assets | 2026-08-09 23:34 PDT | 0.90 s | Passed | Downloaded wheel, sdist, and docs archive independently |
| Verify live SHA-256 digests | 2026-08-09 23:34 PDT | 0.1 s | Passed | All local digests matched GitHub asset digests |
| Live wheel fresh-install qualification | 2026-08-09 23:34 PDT | 8.35 s | Passed | Version/export/docs smoke and `pip check` passed |
| Live sdist fresh-install qualification | 2026-08-09 23:34 PDT | 9.19 s | Passed | Version/export/docs smoke and `pip check` passed |
| Close issues #267 and #268 | 2026-08-09 23:35 PDT | 1.6 s | Passed | Both issues closed with published release evidence |

The closeout PR run remains durable in GitHub and is incorporated into the next
release's PR-CI sample set when that release initializes its timing history.
