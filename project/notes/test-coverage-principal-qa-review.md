---
created: 2026-07-23
---

# Test Coverage Review: Principal QA

## Review Frame

This pass evaluates test coverage from the perspective of a principal software
quality assurance engineer.

The question is not whether the repository contains many tests. The question
is whether the available evidence is sufficient to qualify a build for its
stated users and supported environments.

This pass evaluates:

- product-risk coverage;
- requirements and capability traceability;
- test levels and release gates;
- platform and dependency qualification;
- installed-artifact and upgrade behavior;
- functional, integration, system, and acceptance coverage;
- geometry and manufacturing oracles;
- security and malformed-input coverage;
- concurrency, recovery, and lifecycle behavior;
- performance, load, stress, and soak evidence;
- reference artifact authority;
- result retention, triage quality, and release auditability.

## Qualification Result

**Release qualification: FAIL**

The repository has broad internal execution coverage when tests are isolated:
84.3% of statements and 68.4% of branches. That is not the coverage of the
actual release gate.

The current CI job selects 94 of 1,560 collectable tests, approximately 6%.
When those five test modules are isolated to avoid their order-dependent hang,
they cover:

| CI selection measure | Coverage |
| --- | ---: |
| Lines | 37.5% |
| Branches | 9.0% |
| Combined report | 30.7% |

The exact CI ordering does not complete locally. It passes 79 tests and then
hangs in
`test_live_preview_process_builder_returns_mesh_dataset` after the process
controller tests have run.

The release workflow builds and publishes wheel, source, and docs artifacts
without running any tests or installing the artifacts it publishes.

There is therefore no qualified build, even if the selected GitHub job happens
to report green in a particular environment.

## Shared Measurement Facts

| Measure | Observed |
| --- | ---: |
| Production modules | 89 |
| Test modules | 77 |
| Collectable tests | 1,560 |
| Additional modules failing collection | 4 |
| Modules passing in isolated processes | 69 |
| Modules failing in isolated processes | 8 |
| Isolated test failures | 6 |
| Broad integration result without Qt shell | 50 failures, 727 passes, then stopped |
| Monolithic run with Qt shell | Python abort at 46% |
| Configured durable coverage artifacts before review | none |
| Approved review fixture records | 4 of 116 |

The refreshed coverage artifacts are:

- `project/coverage/coverage.xml`;
- `project/coverage/html/index.html`;
- `project/coverage/ci-isolated-coverage.xml`;
- `project/coverage/isolated-test-results.json`.

The broad coverage artifact combines per-module process runs. It must be labeled
as isolated execution coverage and not represented as a passing integrated
suite.

## Product-Risk Coverage Matrix

| Product risk | Current evidence | QA assessment |
| --- | --- | --- |
| Clean install | package builds in CI | not tested after installation |
| First model | private loader helper tests | public workflow not qualified |
| Preview | unit-heavy review UI tests | aggregate run can abort in Qt |
| STL export | many reference outputs | mostly dirty/unreviewed baselines |
| Loft and CSG | dense internal tests | aggregate failures and known dirty geometry |
| Text | nine isolated tests | fonts, packaging, glyph breadth, and downstream use incomplete |
| `.impress` persistence | broad unit and malformed-payload tests | no released-version compatibility corpus |
| CLI docs download | implementation exists | extraction and download paths uncovered |
| Security | some payload budgets | archive, model execution, filesystem, and dependency abuse gaps |
| Cancellation and workers | mocked/unit lifecycle tests | exact CI order hangs at process handoff |
| Platforms | Ubuntu, Python 3.13 workflow | declared Python 3.10+ and macOS/Windows unqualified |
| Performance | isolated durations observed manually | no budget, trend, stress, or memory gate |
| Release artifacts | wheel and source uploaded | no artifact smoke, signature, or install validation |

## Findings

### P0: The Actual CI Gate Is Narrow And Does Not Complete Reliably

`.github/workflows/ci.yml:54-61` runs only:

- `test_sdf.py`;
- `test_reference_review_preview_payload.py`;
- `test_reference_review_preview_payload_controller.py`;
- `test_reference_review_ui_shell.py`;
- `test_reference_review_ui_workflow.py`.

This is 94 tests, approximately 6% of the collectable suite. It omits nearly
all loft, CSG, surface, topology, persistence, transforms, text, STL, CLI,
documentation, and compatibility coverage.

The exact sequence hangs because the process-controller tests leave state that
prevents a later preview builder from completing. Running the five files in
separate processes produces 30.7% combined coverage, but that is not what CI
does.

Required qualification change:

- replace the hand-selected file list with explicit test levels;
- run a deterministic fast gate on every pull request;
- run integration and reference gates in isolated jobs;
- enforce per-test and per-job timeouts;
- fail on leaked processes, threads, and executors;
- publish collection count and test selection as build metadata.

### P0: The Release Workflow Publishes Untested Artifacts

`.github/workflows/release.yml:30-44` builds artifacts, packages docs, and
uploads them. It does not:

- run the test suite;
- consume a previously qualified commit artifact;
- install the wheel;
- import every packaged module;
- run the CLI entry points;
- execute a first-model preview/export smoke;
- inspect package data;
- verify the docs archive;
- test the source distribution;
- assert that release version and contents match the tag.

A source checkout test cannot qualify a wheel that was never installed.

Required qualification change:

- build once, retain immutable artifacts, and test those exact artifacts;
- install wheel and source distribution in clean environments;
- run public API, CLI, preview/export, `.impress`, and package-data smokes;
- require all release gates before upload;
- retain hashes, manifests, test results, and coverage with the release.

### P0: There Is No Stable Integrated System Test Result

The repository-level suite has three different failure modes:

1. collection stops with four missing hinge imports;
2. a broad run excluding those imports reaches 50 failures after 727 passes;
3. a run including the Qt UI test file aborts the Python process at
   `preview_controls.py:165`.

Most affected modules pass alone. This shows that system behavior depends on
test order and process state, not merely that individual features are broken.

QA cannot sign a build whose result changes when tests are grouped differently.

Required qualification change:

- establish a clean integrated test environment;
- isolate user-model loading from the product interpreter;
- enforce lifecycle teardown for Qt, VTK, pools, threads, and subprocesses;
- run order-randomized and repeated suites after baseline repair;
- retain crash traces and process inventories on abnormal termination.

### P1: Supported Runtime And Platform Claims Are Not Qualified

`pyproject.toml` declares Python `>=3.10`. CI and release use only Python 3.13
on Ubuntu. The product also has macOS-relevant preview behavior and optional Qt,
VTK, PyVistaQt, font, image, and native geometry dependencies.

There is no test matrix for:

- Python 3.10, 3.11, 3.12, and 3.13;
- macOS and Windows;
- minimum supported dependency versions;
- latest compatible dependency versions;
- headless versus interactive rendering;
- core install without the UI extra;
- UI install with the extra;
- native-wheel availability and fallback behavior.

Required qualification change:

- define the supported environment matrix explicitly;
- qualify minimum and current Python versions;
- run core tests on every supported OS;
- run renderer and native-library smokes on representative hardware;
- test dependency floors separately from the locked development environment.

### P1: Installation, Upgrade, And Distribution Behavior Are Uncovered

CI sets `PYTHONPATH=src` and installs `requirements.txt`. It builds the package
but does not install or test it.

There are no automated acceptance tests for:

- installation into a clean virtual environment;
- console script creation;
- optional extras;
- upgrading from a prior release;
- uninstalling and reinstalling;
- wheel versus source distribution parity;
- packaged fonts, QML, icons, examples, and docs;
- editable-install differences;
- dependency conflicts and missing native wheels.

Required qualification change:

- add clean-environment wheel and source-install jobs;
- run imports and console entry points outside the checkout;
- maintain a previous-release upgrade fixture;
- compare package manifests against an approved release manifest;
- test absent optional extras and actionable diagnostics.

### P1: Canonical User Workflows Are Not Covered End To End

The CLI test file exercises four private scene-loader behaviors. It does not run
the documented command workflow through an installed executable.

Missing system acceptances include:

- create a canonical public primitive;
- preview it;
- export it to STL;
- reopen or inspect the result;
- save and load `.impress`;
- edit a dependency and observe hot reload;
- fail safely on invalid model output;
- download and open docs;
- run representative documentation examples.

Internal geometry tests cannot substitute for the first experience a user has
with the product.

Required qualification change:

- turn documented workflows into executable acceptance tests;
- run them from a temporary user project outside the repository;
- assert output geometry, files, exit codes, diagnostics, and cleanup;
- keep one short acceptance in every pull request and a broader matrix before
  release.

### P1: Reference Artifacts Are Not Yet Release-Authoritative Evidence

The review fixture file contains 116 records:

| State | Count |
| --- | ---: |
| Unreviewed | 112 |
| Approved | 4 |
| Dirty STL path | 98 |
| Gold STL path | 4 |
| No artifact, diagnostic only | 14 |

The only `reference_image` test module does not collect because it imports
missing hinge APIs. Consequently, the visual/reference evidence system is not
available as one release gate.

The reference helpers are technically useful:

- STL text is canonicalized and compared exactly;
- images use mean absolute delta;
- silhouette and orientation comparators exist;
- dirty and approved lifecycle metadata exists.

But equality to a dirty baseline only proves that output did not change. It
does not prove that the output is correct.

Required qualification change:

- keep dirty fixtures out of release pass counts;
- require an accountable approval record for every release-authoritative
  baseline;
- expose approved, dirty, missing, invalidated, and diagnostic-only counts in
  CI;
- make reference collection independent of unrelated hinge imports;
- require physical invariant checks alongside visual approval.

### P1: Geometry Regression Equality Is Not Manufacturing Qualification

Many STL tests require minimum facet count, vertex count, file size, and exact
canonical text agreement. Those checks catch output drift but do not establish
manufacturing suitability.

Release-level geometry evidence should cover:

- watertightness and boundary edges;
- manifoldness and non-manifold edges;
- consistent orientation and signed volume;
- self-intersection;
- degenerate and zero-area faces;
- minimum feature and wall thickness;
- units and scale;
- disconnected islands;
- slicer import and repair warnings;
- geometric tolerance against authored intent;
- sharp-feature and correspondence preservation;
- cap continuity and closure.

The current dirty-reference workflow can faithfully preserve a known-soft
corner or defective endcap.

Required qualification change:

- define invariant sets by fixture purpose;
- fail on violated physical invariants before visual review;
- use approved geometry metrics and section comparisons;
- add slicer-oriented smoke checks for release representatives;
- retain diffs that identify which invariant changed.

### P1: Concurrency And Cancellation Tests Do Not Qualify Lifecycle Behavior

The suite includes detailed controller, stale-result, cancellation, and queue
tests. That is a strength. However, the exact CI sequence hangs when the
controller tests precede the process-backed preview builder.

The tests mostly qualify records, mocked dispatchers, or individual controller
methods. They do not prove that:

- workers terminate on normal close;
- cancellation stops expensive work;
- repeated open/close cycles leave no children;
- crashes release temporary payloads;
- process pools survive or reset after exceptions;
- renderer and worker shutdown order is safe;
- application exit completes within a bounded time.

Required qualification change:

- add process-level lifecycle tests with PID and thread accounting;
- exercise repeated open, render, cancel, close, and reopen cycles;
- enforce hard completion deadlines;
- inject worker exceptions and termination;
- fail on surviving children, pools, threads, temporary files, or locks.

### P1: Security And Abuse-Case Coverage Is Incomplete

The `.impress` decoder has valuable malformed-payload and budget tests. Other
attack surfaces are not covered to the same standard.

The docs archive extraction and download paths in `cli.py:148-235` have no
tests, including no traversal, absolute path, symlink, oversized archive, or
decompression-ratio cases.

Additional uncovered risks include:

- executing untrusted Python model code;
- model imports outside the project root;
- environment and filesystem mutation;
- malicious font and image inputs;
- pathological geometry causing memory or CPU exhaustion;
- unsafe URLs and redirects;
- temporary file permissions;
- dependency and release artifact integrity.

Required qualification change:

- maintain an abuse-case suite for every external input boundary;
- run resource-limited malformed-input tests;
- test path containment and symlink behavior;
- define which inputs are trusted and enforce that boundary in tests;
- include security regressions in release gating.

### P2: Nonfunctional Quality Has No Automated Qualification

There is no performance regression gate, memory budget, load test, stress test,
or soak test.

Observed isolated test durations already identify expensive product areas:

- reference STL expansion: 101 seconds;
- loft core: 43 seconds;
- loft showcase: 31 seconds;
- loft correspondence: 26 seconds;
- surface CSG: 17 seconds;
- text: 5 seconds.

These are test durations, not product benchmarks. They do not measure model
size scaling, preview latency, memory growth, cancellation latency, export
throughput, or long-session stability.

Required qualification change:

- define representative small, medium, and large models;
- measure plan, execute, tessellate, preview, save, load, and export phases;
- track peak memory and surviving process count;
- add repeated-edit and long-running preview soak tests;
- gate statistically significant regressions with stored baselines.

### P2: Requirements And Test Traceability Are Informal

Tests use descriptive names and the repository has specifications and
progression documents, but there is no machine-readable mapping from:

- product capability;
- requirement or claim;
- risk and severity;
- test level;
- automated case;
- reference artifact;
- supported environment;
- release gate.

Large matrices can therefore look comprehensive while a basic user workflow has
no acceptance test.

Required qualification change:

- create a capability-to-evidence manifest;
- require each release claim to identify its automated and manual evidence;
- identify deferred risks explicitly;
- report orphan tests and untested requirements;
- make dirty references and diagnostic-only fixtures visible as incomplete
  evidence.

### P2: CI Does Not Preserve Enough Evidence For Triage Or Audit

The workflows do not publish:

- JUnit test results;
- coverage XML or HTML;
- selected-test manifests;
- screenshots or STL diffs;
- crash traces;
- process/thread diagnostics;
- environment and dependency manifests;
- timing trends;
- the built artifact used by downstream test jobs.

Without retained evidence, a green or red job is difficult to compare over
time, and abnormal exits can erase the most valuable diagnostic context.

Required qualification change:

- publish structured test and coverage reports;
- retain reference diffs and failure artifacts;
- capture dependency, OS, Python, Qt, VTK, and renderer versions;
- retain per-test duration and timeout data;
- associate every release artifact with its exact qualification record.

### P2: Test Data And Environment Reproducibility Are Not Controlled End To End

The environment pins a large `requirements.txt`, but product behavior also
depends on fonts, renderer behavior, native libraries, GPU/display mode, image
decoders, and host filesystem behavior.

Text tests use a fixture font path when available, but packaging and broader
glyph coverage are not qualified. Rendering is forced offscreen in
`tests/conftest.py`, while the live product must also operate interactively.

Required qualification change:

- package and checksum authoritative test fonts and images;
- record native dependency and renderer versions with artifacts;
- separate deterministic headless image evidence from interactive smokes;
- test locale, path, and filesystem edge cases;
- control random seeds and numerical tolerance policy centrally.

### P2: Persistence Coverage Lacks A Released-Version Compatibility Corpus

`.impress` has one of the strongest unit-test areas in the repository, including
round trips, unsafe fields, malformed geometry, and atomic write behavior.

However, there is no clearly versioned corpus of documents produced by each
released Impression version and no automated forward/backward compatibility
policy.

Required qualification change:

- preserve canonical documents from every release;
- define supported read and write compatibility;
- run migration and rejection cases across the corpus;
- verify stable identities, units, metadata, and geometry after migration;
- test partial writes, interruption, and recovery at the installed-product
  boundary.

## Positive Quality Evidence

- The suite contains substantial domain-specific negative testing.
- `.impress` validation and round-trip coverage are broad.
- CSG, loft, surface, and topology internals are heavily executed.
- Reference infrastructure includes exact STL comparison, image delta,
  silhouettes, orientation checks, and promotion concepts.
- Async controller tests explicitly consider stale results and cancellation.
- Deprecation warnings fail tests by default.
- Sixty-nine test modules pass independently.
- Coverage is configured for branches and now has refreshed XML and HTML
  artifacts.

These are useful components of a qualification system. They do not yet form one
because the release path, integrated execution, environment matrix, and
evidence authority are missing.

## Recommended Quality Gate Architecture

### Pull Request Gate

- formatting, static analysis, and import-boundary checks;
- sub-ten-second unit and contract suite;
- installed-wheel first-model smoke;
- changed-line and changed-branch coverage;
- deterministic collection count;
- no leaked processes or threads.

### Integration Gate

- public CLI, preview, export, save/load, and docs workflows;
- process, Qt, VTK, and hot-reload lifecycle;
- public geometry invariants;
- malformed input and security boundaries;
- selected approved reference artifacts.

### Nightly Gate

- complete Python and platform matrix;
- full approved image/STL suite;
- property and fuzz tests;
- performance, memory, stress, and soak runs;
- randomized order and repeated execution;
- compatibility corpus.

### Release Gate

- consume immutable wheel and source artifacts;
- clean install and upgrade tests;
- full supported-environment qualification;
- all P0/P1 product requirements traced to passing evidence;
- zero dirty artifacts counted as release proof;
- retained reports, manifests, hashes, and failure artifacts.

## Release Acceptance Criteria

A build is qualifiable when:

- repository collection and the integrated suite are green;
- the exact CI command completes within a defined deadline;
- CI tests the supported product breadth rather than a hand-selected 6%;
- release tests consume the exact artifacts that will be published;
- every supported Python and OS claim has recorded evidence;
- canonical install-to-export workflows pass outside the checkout;
- critical security boundaries have abuse-case tests;
- geometry claims are supported by physical invariants, not only regression
  equality;
- approved reference evidence is clearly separated from dirty
  characterization;
- worker, renderer, and application lifecycle tests leave no resources behind;
- performance and memory remain within approved budgets;
- test, coverage, environment, and artifact evidence is retained with the
  release.
