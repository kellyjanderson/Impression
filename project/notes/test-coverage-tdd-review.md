---
created: 2026-07-23
---

# Test Coverage Review: Strict TDD

## Review Frame

This pass evaluates the test suite as a senior software engineer applying
strict test-driven development discipline.

The standard is not simply whether tests exist or whether lines execute. A
TDD-grade suite must provide:

- an always-green executable specification;
- fast and deterministic red-green-refactor feedback;
- tests expressed primarily through stable behavior and public contracts;
- isolation from order, process, renderer, filesystem, and import state;
- small tests whose failure identifies one behavior;
- confidence that implementation can be refactored without rewriting the
  specification;
- focused unit tests supported by contract and end-to-end acceptance tests;
- coverage measurement used as a gap detector, not as the test design goal.

The repository snapshot cannot prove whether any particular test was written
before its implementation. This review therefore does not claim a development
chronology. It evaluates whether the current suite can support disciplined TDD
work now.

## Evidence Basis

The following measurements were taken from the current checkout using the repo
`.venv` on Python 3.13.13.

| Measure | Observed |
| --- | ---: |
| Test modules | 77 |
| Test source lines | 35,101 |
| Declared `test_*` functions | 1,382 |
| Collectable test cases | 1,560 |
| Test modules passing in isolated processes | 69 of 77 |
| Test modules failing in isolated processes | 8 of 77 |
| Isolated test failures | 6 |
| Modules failing during collection | 4 |
| Isolated execution time with coverage | 289 seconds |
| Isolated line coverage | 84.3% |
| Isolated branch coverage | 68.4% |
| Combined line-and-branch report | 80.5% |
| Uncovered statements | 4,732 |
| Uncovered branch outcomes | 2,965 |

The coverage result is an execution estimate, not a green-suite result. It was
produced by running each test module in a separate process and combining the
data because the repository-wide run is not viable. It includes lines reached
by the six failing tests.

The exact repository collection reports 1,560 tests plus four collection
errors. A broad run with the four import-broken modules and the aborting Qt
module excluded still reached the `--maxfail=50` limit after only 727 passes.

## Verdict

This is not currently a TDD-grade suite.

It is a large and often technically sophisticated regression corpus. It has
strong pockets of negative testing, explicit diagnostic contracts, numerical
assertions, parameterized cases, and reference comparisons. Most test modules
also pass quickly when run by themselves.

However, the suite cannot act as one executable specification:

- repository collection is broken;
- integrated execution is order-dependent;
- the CI ordering hangs;
- the broad run can abort inside Qt;
- no enforced coverage or green-suite gate exists;
- many tests are coupled to private functions and implementation records;
- numerous tests are explicitly shaped around covering remaining branches;
- entire production modules remain at zero coverage.

The suite appears to have been accumulated around implementation surfaces more
than curated around a stable hierarchy of user behaviors. That does not make
the tests worthless. It does mean their volume substantially overstates their
usefulness during red-green-refactor work.

## Findings

### P0: There Is No Green Executable Specification

The full suite cannot collect:

- `tests/test_hinges.py` cannot import `make_bistable_hinge`;
- `tests/test_no_hidden_mesh_fallback.py` cannot import
  `HingeSurfaceAssembly`;
- `tests/test_reference_images.py` cannot import
  `handoff_hinge_surface`;
- `tests/test_surface_hinges.py` cannot import
  `HingeSurfaceAssembly`.

Eight of 77 test modules fail even when run in isolated processes. The
collectable portion contains six assertion failures covering stale
documentation paths, changed CSG behavior, hinge documentation, and cone
connectivity.

TDD depends on a binary, trusted baseline: a new red test must be distinguishable
from pre-existing red. That is impossible while collection and known failures
are part of the default state.

Required correction:

- restore clean collection before adding feature tests;
- make every known failure green, remove a stale test with a recorded decision,
  or quarantine it under an explicit expiring defect identifier;
- fail CI when collection differs from the declared suite manifest;
- prohibit merging while the default test command is red.

### P0: Test Outcomes Depend On Execution Order And Shared Runtime State

The same modules that pass independently fail in aggregate. The broad
non-Qt run stopped after 50 failures and 727 passes. Failures included
impossible-looking results such as an instance not satisfying `isinstance`
against the identically named class and `SurfaceBody` rejecting
`SurfaceShell` values.

The test sequence beginning with `tests/test_cli_preview.py` unloads and
reimports Impression modeling modules. Later tests retain older class objects,
creating split class identities. Heightmap, drafting, CSG, reference payload,
and STL tests then fail for reasons unrelated to the behavior they name.

The CI-selected process-controller and UI-shell modules also pass independently
but hang when run in CI order at
`test_live_preview_process_builder_returns_mesh_dataset`.

An isolated unit test is not isolated if running an earlier test changes the
meaning of its types or prevents process shutdown.

Required correction:

- remove production-package unloading from model reload behavior;
- give every process, executor, Qt object, and module cache an owned teardown;
- add an order-randomized gate after the deterministic suite is repaired;
- add a repeated-run gate to detect leaked state;
- make test order irrelevant instead of encoding a preferred order.

### P1: The Suite Does Not Provide A Usable TDD Feedback Ladder

There is no functioning smoke tier:

- the declared `smoke` marker selects zero tests;
- the `preview` marker selects only four;
- the `loft` marker selects 27 despite many additional loft test modules;
- the `reference_image` marker cannot collect because its only marked module
  imports missing hinge APIs.

The isolated coverage run took 289 seconds. The slowest modules were:

| Test module | Isolated duration |
| --- | ---: |
| `test_reference_stl_expansion.py` | 101.1 s |
| `test_loft.py` | 43.0 s |
| `test_loft_showcase.py` | 31.5 s |
| `test_loft_correspondence.py` | 26.1 s |
| `test_surface_csg.py` | 17.3 s |

A developer has no supported command for a sub-second contract loop, a
few-second subsystem loop, and a deterministic pre-merge loop.

Required correction:

- define `unit`, `contract`, `integration`, `reference`, `slow`, and `smoke`
  tiers by behavior and runtime;
- require every product capability to have at least one smoke acceptance;
- publish expected runtime budgets for each tier;
- keep the default local unit tier below ten seconds;
- run slow geometry and visual tests independently without weakening their
  release authority.

### P1: Many Tests Specify Private Implementation Instead Of Stable Behavior

The suite imports at least 46 private production symbols. For example,
`tests/test_loft.py:55-63` imports private planners, executors, assignment
algorithms, and handoff helpers directly.

The suite also contains:

- 122 references to `canonical_payload`;
- 114 test names containing `report`;
- 64 test names containing `evidence`;
- 49 test names containing `cover`;
- 41 test names containing `policy`;
- 35 test names containing `matrix`.

These concepts can be valid public contracts, but here many are intermediate
records and planning mechanisms. Tests that pin each internal record make
refactoring expensive without necessarily increasing user-visible confidence.

Required correction:

- classify every test as public behavior, stable subsystem contract, or
  implementation characterization;
- place characterization tests beside the internal owner and keep them out of
  the public acceptance count;
- prefer output geometry, errors, invariants, persistence, and workflow results
  over intermediate record shape;
- test private helpers directly only when the helper is a deliberately stable
  numerical kernel.

### P1: Coverage-Chasing Tests Bundle Unrelated Behaviors

Several test names state that their purpose is to cover implementation:

- `test_loft_candidate_enumerators_and_prediction_helpers_cover_remaining_paths`;
- `test_private_surface_builders_and_ops_cover_validation_branches`;
- other `cover_*`, `*_coverage_*`, and `*_remaining_paths` tests.

There are nine test functions longer than 80 lines and 18 with at least 15
direct assertions. One CSG test contains 74 assertions. The loft metadata
contract test contains 48.

These tests may increase line and branch percentages, but one failure can
represent dozens of unrelated causes. They do not provide the narrow red signal
needed for TDD.

Required correction:

- split branch-coverage aggregators into one behavior per test;
- use parameterization only when all rows express the same invariant;
- name tests for domain behavior, not for coverage bookkeeping;
- allow a small numerical helper test to have several related assertions, but
  keep orchestration and policy outcomes separate.

### P1: Whole Product Modules Have No Test-Driven Contract

The isolated coverage report shows zero line coverage for:

- `src/impression/cad.py`;
- `src/impression/cad/__init__.py`;
- `src/impression/modeling/extrude.py`;
- `src/impression/modeling/hinges.py`;
- `src/impression/printability.py`;
- `src/impression/validation.py`.

Other user-facing boundaries are weak:

| Module | Line coverage | Branch coverage |
| --- | ---: | ---: |
| `_vtk_runtime.py` | 9.8% | 0.0% |
| `modeling/_ops_planar.py` | 13.3% | 0.0% |
| reference artifact preview | 34.0% | 5.9% |
| `cli.py` | 40.1% | 27.5% |
| `preview.py` | 43.2% | 26.8% |

The dense center of CSG, loft, and surface code is highly executed, while
complete product paths and smaller user-facing modules remain unowned.

Required correction:

- assign a behavior-level test owner to every production module;
- begin with public extrude, hinge, validation, printability, CLI, preview, and
  CAD acceptance contracts;
- remove dead modules instead of treating zero coverage as a test backlog;
- set subsystem-specific minimums only after deciding which code is product.

### P1: Public Workflows Are Under-Tested Relative To Internal Machinery

`tests/test_cli_preview.py` contains four tests of the private
`_scene_factory_from_path` helper. It does not execute the installed
`impression` command, preview a canonical public `SurfaceBody`, export a real
model, download docs, or validate exit status and user-facing diagnostics.

The docs archive extraction and download paths in `cli.py:148-235` are entirely
uncovered. The built wheel is not installed for tests. Documented examples are
mostly checked as source text rather than executed from an installed
distribution.

Strict TDD needs outside-in acceptance tests to shape the API before internal
records proliferate.

Required correction:

- write executable acceptance tests for the README's first model;
- run the real CLI in a fresh subprocess against an installed wheel;
- exercise preview and export with canonical public values;
- execute documentation examples;
- test docs download success, failure, and malicious archives through the
  command boundary.

### P1: Coverage Is Configured But Not A Development Gate

`pyproject.toml:58-76` enables branch measurement and declares XML and HTML
paths, but it defines no `fail_under` threshold. CI does not invoke coverage,
upload an artifact, or compare changed-line coverage.

Before this review, `project/coverage/` contained no XML or HTML artifacts.
The actual CI selection reaches only 37.5% line and 9.0% branch coverage when
its files are run in isolated processes. The combined report is 30.7%.

Coverage should not drive test design, but unmeasured regressions leave no
feedback that production paths have become unexercised.

Required correction:

- make coverage collection part of the green test command;
- establish a ratcheting repository baseline rather than selecting an
  arbitrary aspirational percentage;
- require high changed-line and changed-branch coverage;
- fail when a production module unexpectedly drops to zero;
- preserve XML and HTML artifacts for every CI run.

### P2: Test Ownership Is Concentrated In Giant Files

Three files contain 564 of the 1,382 declared test functions:

| File | Test functions | Lines |
| --- | ---: | ---: |
| `test_surface_csg.py` | 217 | 5,954 |
| `test_surface.py` | 214 | 5,058 |
| `test_loft.py` | 133 | 3,629 |

This concentration mirrors the oversized production modules. It makes focused
selection, fixture ownership, review, and refactoring harder. A change to one
concept often requires navigating thousands of lines of unrelated tests.

Required correction:

- organize tests by behavior and owning subsystem;
- separate public API contracts, planner contracts, numerical kernels,
  refusals, persistence, and integration tests;
- keep reusable builders in explicitly scoped fixture modules;
- avoid a new test file for every record type; organize around user and domain
  capabilities.

### P2: Numerical And Topological Invariants Lack Generative Verification

The suite contains many hand-authored geometry matrices, but no Hypothesis or
other property-based test framework, no fuzzing harness, and no mutation test
gate.

This matters for:

- cyclic loft correspondence and point phase;
- station permutation and reversal;
- degenerate and near-degenerate geometry;
- transform composition and inverse properties;
- serialization round trips;
- CSG commutativity or containment where mathematically applicable;
- tessellation manifold and orientation invariants.

Enumerating more named fixtures is not a scalable substitute for exploring
input space.

Required correction:

- add bounded property generators for profiles, transforms, and serializable
  geometry;
- preserve every discovered failure as a minimal regression example;
- add mutation testing first to small numerical and validation modules;
- define tolerances and shrinking constraints as part of each property.

### P2: Markers Are Assigned By Filename Instead Of Test Contract

`tests/conftest.py:15-62` adds major markers based primarily on whole filenames.
This produces misleading selections. For example, only the five filenames in
`_LOFT_FILES` receive `loft`, while numerous loft-specific files do not.

Filename assignment also makes moving a test change its test tier without any
review of its behavior.

Required correction:

- declare test level and capability explicitly at module or test definition;
- add a collection check that every test has exactly one level marker and one
  capability owner;
- keep marker membership independent of filenames;
- publish collection counts so an accidentally empty tier fails.

### P2: Golden Tests Protect Existing Output, Including Known-Bad Output

Reference image and STL helpers perform real comparisons. STL output is
canonicalized and compared exactly; images use a mean-delta threshold and
additional silhouette comparisons.

However, the review fixture inventory contains:

- 116 fixture records;
- 112 unreviewed records;
- 98 dirty STL artifacts;
- four approved/gold artifacts;
- 14 diagnostic records with no artifact.

A dirty baseline is useful characterization, but it is not an approved behavior
specification. Exact agreement with a dirty STL can preserve a correspondence,
endcap, topology, or orientation defect indefinitely.

Required correction:

- classify dirty references as characterization only;
- exclude unreviewed references from passing release coverage;
- require approved references to state geometric invariants in addition to
  byte or image agreement;
- keep an explicit count of approved, dirty, missing, and invalidated baselines
  in the test result.

## Positive Foundations

- Deprecation warnings are errors by default.
- The suite includes 360 `pytest.raises` usages and substantial negative-path
  validation.
- Many numerical assertions use tolerances rather than fragile exact floats.
- Parameterized matrices cover meaningful geometry families.
- `.impress` serialization has broad round-trip and invalid-payload coverage.
- Reference helpers compare canonical STL text and provide silhouette-oriented
  image analysis rather than relying only on file existence.
- Sixty-nine test modules pass independently, and most complete in under one
  second.

These are valuable foundations. The repair should preserve domain assertions
while changing suite ownership, isolation, and feedback structure.

## TDD Recovery Sequence

1. Restore clean collection and remove known red from the default suite.
2. Repair module reload, executor, and Qt ownership so order cannot change
   outcomes.
3. Establish a sub-ten-second unit and contract tier.
4. Add outside-in acceptance tests for install, first model, preview, export,
   persistence, and CLI failure behavior.
5. Reclassify private-record tests as internal characterization or remove them.
6. Split coverage aggregators and giant test modules by behavior.
7. Add explicit markers with non-empty collection gates.
8. Add property tests for geometry and persistence invariants.
9. Ratchet line, branch, and changed-code coverage in CI.
10. Promote only reviewed references into release-authoritative coverage.

## Acceptance Criteria

The suite is TDD-ready when:

- the default command collects and passes from a clean checkout;
- two consecutive runs in different orders produce the same result;
- no test leaves executors, child processes, Qt objects, imports, or files for
  the next test;
- a supported fast tier completes in under ten seconds;
- every public modeling capability has an outside-in acceptance;
- production modules cannot silently fall to zero coverage;
- changed branches are covered or explicitly justified;
- private implementation can be reorganized without rewriting public behavior
  tests;
- slow and reference tiers are deterministic and independently visible;
- only approved reference artifacts contribute to a release claim.
