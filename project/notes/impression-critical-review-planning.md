---
created: 2026-07-23
---

# Discussion Notes: Impression Critical Review Planning

## Context

- The review posture for this discussion is deliberately critical: identify missing features, incorrect output, dirty reference STLs that should not be promoted, and places where user-visible behavior has drifted from earlier releases.
- Starting topics: dirty STL reference review fixtures, text support, and loft/endcap behavior.
- The user specifically called out a square/rectangle loft fixture where three corresponding corners look sharp but the same corner across stations looks soft, suggesting one corner may be shifted by one sample before loft.
- The three principal-engineer reviews must remain repository-wide and
  independent of those starting topics.
- For this work, code quality means the engineer-facing quality of the code
  itself: naming, identifier length, control-flow shape, call depth, API
  interfaces, module layout, abstraction quality, and whether the result reads
  like deliberate human code or accumulated generated code.
- Correctness, security, and release-integrity findings are valuable but are not
  code-quality findings merely because they were discovered during a code
  review. They are preserved in a separate supplemental review.
- The notes index tracks individual issues. Review-document links are not
  checkboxes unless a document contains exactly one issue.

## Current Local State

- The current dirty STL fixture file is `tests/reference_review_fixtures/dirty-stl-fixtures.json`.
- In that file, index `0` is `loft/anchor_shift_rectangle`; `loft/square_correspondence` is present at index `42`.
- `build_loft_square_correspondence` and `build_loft_anchor_shift_rectangle` both route through `_loft_from_profiles(...)`, which constructs `Loft(..., cap_ends=True)` with no authored correspondence names.
- `build_square_correspondence_profiles()` currently creates four rectangular stations, not three. The user's "three stations" description may refer to the visible review ordering, an older fixture version, or a neighboring square fixture.
- The current automatic correspondence stack has named/authored rail resolution and refusal logic, but the dirty square/rectangle review artifacts are not authored with named corner correspondence.

## Findings And Leanings

- The suspected visual defect is plausible as an automatic correspondence bug or policy gap. Unnamed rectangle-like sections rely on inferred ordering/anchoring, so a cyclic off-by-one can create a locally softened or blended corner while other corners still look correct.
- Requiring named correspondence for every loft would be robust but heavy-handed. A better first design target may be: automatic correspondence remains allowed when confidence is high, but ambiguity or unstable anchor evidence produces an explicit refusal/request for named rails.
- Mixed named and unnamed correspondence should probably be supported, but only if the named rails are treated as protected anchors and all inferred spans remain auditable. Partial naming should not silently imply that all unnamed points are safe.
- A focused regression should assert physical corner preservation for unnamed square/rectangle stations and separately assert refusal when corner phase is ambiguous.

## Text State

- Current `make_text(...)` is surface-first and returns `SurfaceBody`.
- `make_text_mesh(...)` remains as an explicit legacy/compatibility route.
- `text_profiles(...)` / `text_sections(...)` return topology-native `Section` values.
- Empty surface text returns a hidden placeholder body; empty mesh text returns an empty mesh.
- There is an approved reference fixture `surfacebody/text_surface` that builds `SURFACE` with a glyph-capable font.
- Critical review questions remain around font reproducibility, compound glyph/hole quality, complex glyph coverage, downstream boolean/CSG participation, and whether hidden placeholder output is the right empty-text behavior.

## Endcap State

- Current `loft(...)` supports `cap_ends=True` plus named `start_cap` / `end_cap` values: `flat`, `taper`, `dome`, and `slope`.
- Current `loft_endcaps(...)` still exists as an experimental comparison route with `FLAT`, `CHAMFER`, `ROUND`, and `COVE`, but it returns `Mesh`, not canonical `SurfaceBody`.
- `v0.0.3a1` already contained the experimental `loft_endcaps(...)` route and the same public cap option family, so the reuse question is likely not a code-copy-back task. It is more likely a surface-native promotion/adaptation task.
- Prior repo memory records a separate cone/frustum narrow-end cap fix: `make_surface_cone()` should emit planar caps for every non-zero end radius. That does not fully answer loft endcap parity.

## Current Six Assertion Failures

The six currently reproducible assertion failures reduce to four root causes.
They should not be treated as six independent production regressions.

### Documentation Policy Tests Target Retired Paths

Three failures in `tests/test_documentation_rules.py` still read
`agents/documentation.md`, `agents/specifications.md`, and
`project/agents/reference-images.md`. Those paths no longer exist. The current
rules live under `.agents/skills/documentation/SKILL.md` and
`.agents/skills/reference-artifact-lifecycle/SKILL.md`.

This is a governance-document migration mismatch. The tests need to target the
canonical managed skills and assert their current semantic contracts, unless
the old paths are deliberately retained as compatibility entry points.

### Reference Expansion CSG Test Expects A Retired Refusal

`test_reference_expansion_lofted_body_csg_refuses_without_fallback` expects an
intersecting loft/box difference to return `unsupported`. The current exact
surface CSG implementation succeeds through `surface-csg.loft-primitive`,
records `no_mesh_fallback=True`, and produces a watertight tessellation for this
fixture. Newer positive CSG tests explicitly cover the intersecting
trim-fragment route.

The negative test was not converted when the capability landed. It should now
assert successful exact surfaced CSG, or use a genuinely unsupported operand
family if refusal behavior is what the test is meant to preserve.

### Hinge Documentation Omits The Required Surface Selector

`test_hinges_docs_define_surface_public_handoff` correctly detects that the
public example omits `backend="surface"`. The extracted
`impression-hinges` implementation still declares
`make_traditional_hinge_pair(..., backend="mesh")`, while the documentation
claims the helper defaults to surfaced output and immediately passes the
result to `handoff_hinge_surface(...)`.

This is a current documentation/API contradiction, not merely a brittle test.
Either the implementation default must become `surface`, or the docs must show
the explicit selector and stop claiming that surfaced output is the default.

### Cone Connectivity Assertion Predates The Top-Cap Fix

`test_private_surface_builders_and_ops_cover_validation_branches` expects a
top-wide, bottom-apex cone shell to report `connected=True`. That assertion
predates the change that added a planar cap for every non-zero cone end. Once
the top cap was added, the constructor correctly stopped claiming that the
unstitched cap and sidewall form a connected shell.

The immediate test expectation is stale, but it points at a deeper completion
gap: cone and frustum cap patches are present without cap-to-side seam
topology. Current tessellation classifies them as open and reports boundary
edges. The right production fix is to model and weld those seams; simply
setting `connected=True` would make the metadata dishonest.

## Open Questions

- Which exact review app ordering or fixture is the user's "very first item" referring to: `loft/anchor_shift_rectangle`, `loft/square_correspondence`, or another current/older fixture?
- Should automatic correspondence be permitted for unannotated equal-count polygon stations, or should equal-count sharp-corner loops require at least two named anchors?
- What is the desired mixed-mode policy when only some points are named: accept with protected anchors, refuse if gaps are ambiguous, or require all corner names for sharp polygons?
- Which endcap family is the target for "great endcaps" parity: public `start_cap`/`end_cap` (`dome`, `taper`, `slope`) or experimental `loft_endcaps(...)` (`CHAMFER`, `ROUND`, `COVE`)?

## Follow-Up Lanes

### Dirty STL Triage

Inspect `loft/anchor_shift_rectangle` and `loft/square_correspondence` in the
reference review app, record which station/corner is wrong, and decide whether
the artifact should remain dirty, be regenerated, or become a regression
fixture.

### Correspondence Policy

Add targeted tests for unnamed rectangle corner preservation, named corner
protection, partial naming behavior, and ambiguity refusal.

### Text Review

Build a text quality matrix for simple Latin, holes, multiline alignment,
emoji/symbol glyphs, empty text, and text in downstream operations.

### Endcap Parity

Compare HEAD against `v0.0.3a1`/`v0.0.3a2` examples, then decide whether to
promote `loft_endcaps(...)` to surface-native output or keep it quarantined as
mesh compatibility.

## Principal Engineer Review Record

Three independent, repository-wide review passes preserve the broader
engineering assessment. They use separate evaluation frames so the starting
discussion of dirty STLs, correspondence, text, and endcaps does not determine
review priority:

- [Code Quality Review](code-quality-principal-engineer-review.md)
- [Efficiency And Reuse Review](efficiency-and-reuse-principal-engineer-review.md)
- [Technical And Industry Completeness Review](technical-and-industry-completeness-principal-engineer-review.md)

A supplemental
[Correctness And Release Integrity Review](correctness-and-release-integrity-principal-engineer-review.md)
preserves the runtime, security, contract, test, and release findings that were
initially misclassified as code quality.

The highest-priority cross-review conclusions are:

- loading a Python model unloads Impression's internal modeling modules and
  creates order-dependent class-identity failures across the kernel;
- the documented first `SurfaceBody` preview/export path is broken because the
  primary scene adapter does not consume surface values;
- the docs downloader permits archive path traversal;
- public CSG examples, common solid primitives, units, and watertight export do
  not meet their documented contracts;
- release, installer, worker shutdown, test isolation, and reference promotion
  are not reliable enough to establish release truth;
- the codebase has clear, human-readable numerical modules, but its core has a
  high generated-code smell: extremely broad exports, long context-encoded
  names, repeated records and wrappers, large parameter trains, and long call
  tails for simple operations;
- the codebase duplicates scene consumers, numerical kernels, transforms, path
  abstractions, and runtime policy, while large geometry values amplify identity
  and tessellation cost;
- text, correspondence, and endcaps remain important completion lanes, but they
  are part of a wider install-to-manufacture product gap.

## Test Coverage Review Record

Two independent test-coverage passes use different standards:

- [Strict TDD Review](test-coverage-tdd-review.md) evaluates whether
  the suite is a fast, deterministic, behavior-level executable specification
  suitable for red-green-refactor work.
- [Principal QA Review](test-coverage-principal-qa-review.md)
  evaluates product-risk coverage, environment qualification, installed
  artifacts, release gates, geometry evidence, security, lifecycle behavior,
  and nonfunctional quality.

Current measured facts:

- 1,560 tests collect, while four additional test modules fail during
  collection because hinge symbols are missing;
- 69 of 77 test modules pass when each module receives a fresh process;
- eight modules fail in isolation, representing six assertion failures and four
  collection-error modules;
- the broad integrated run excluding the Qt shell and import-broken modules
  stops at 50 failures after 727 passes;
- the monolithic run with the Qt shell aborts Python at 46%;
- the exact CI test order hangs after 79 passes;
- isolated execution reaches 84.3% line and 68.4% branch coverage;
- the 94-test CI selection reaches only 37.5% line and 9.0% branch coverage
  when isolated so it can complete;
- 112 of 116 reference-review fixture records are unreviewed.

Coverage XML, HTML, isolated test results, and CI-selection XML are preserved
under `project/coverage/`.
