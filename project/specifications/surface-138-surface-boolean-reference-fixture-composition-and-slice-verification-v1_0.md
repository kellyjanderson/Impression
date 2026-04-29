# Surface Spec 138: Surface Boolean Reference Fixture Composition and Slice Verification (v1.0)

## Overview

This specification defines how surfaced CSG reference fixtures should be
constructed so they prove boolean correctness instead of only reference-image
plumbing.

## Backlink

- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)](surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)

## Scope

This specification covers:

- composed surfaced CSG fixtures that show operand A, result, and operand B in
  one deterministic presentation
- rules for choosing non-trivial boolean operands instead of disjoint, no-op,
  or exact-containment stand-ins when the fixture is meant to prove overlap
  behavior
- canonical slice definitions for surfaced CSG fixtures whose truth is not well
  captured by a beauty render alone
- orientation-sensitive fixture cues such as outward notches

## Behavior

This leaf must define:

- how a surfaced CSG reference fixture can pair a triptych-style operand/result
  render with one or more canonical section checks
- how orientation-sensitive fixtures declare whether `same_shape_rotated` is a
  failure or an allowed equivalence
- how expected slice truth is defined in a fixture-local frame so expected and
  recovered sections can be compared meaningfully
- how no-cut or containment fixtures are named and scoped when they exist as
  harness baselines rather than overlap-evidence fixtures

## Constraints

- fixtures intended as boolean-correctness evidence must prefer overlapping
  operands whose result is visibly different from both inputs
- union fixtures must not rely on one operand fully containing the other when
  they are meant to prove surfaced overlap behavior
- orientation-sensitive slice fixtures should prefer asymmetric cues such as an
  outward notch so misalignment is visually and programmatically detectable
- fixture truth must remain bounded to the initial surfaced executable scope and
  must not imply general surfaced CSG completion

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- surfaced CSG fixture composition rules are explicit enough to reject weak
  no-cut stand-ins as overlap evidence
- operand/result presentation is explicit enough to make the intended boolean
  outcome visible at a glance
- canonical slice truth and orientation policy are explicit enough to classify
  `same_shape_same_orientation`, `same_shape_rotated`, and `different_shape`
- the paired test specification defines how those fixture rules are enforced
