---
created: 2026-07-29
status: active
---

# CSG One-To-One Surface Patch Completeness Matrix

Path: ad-hoc-path work

## Purpose

Track the implementation and verification state of every ordered one-to-one
surface patch pairing for every CSG operation.

The matrix contains:

- 10 left-operand patch families
- 10 right-operand patch families
- 3 operations: union, difference, and intersection
- 100 ordered family pairs
- 300 operation/family-pair cells

The order of the operands is intentional. Difference is directional, and union
and intersection remain ordered here so future tests prove both caller orders
instead of assuming symmetry.

## Authority Boundary

This document is the completion and evidence ledger. It does not replace:

- `PATCH_FAMILY_CAPABILITY_MATRIX`, which owns the available patch-family
  inventory
- `SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX`, which owns declared runtime
  support policy
- executable tests and reference evidence, which prove that a declaration is
  true

An existing support declaration or checked reference-plan item does not make a
cell complete by itself.

## Patch Families

The matrix covers the current families in `PATCH_FAMILY_CAPABILITY_MATRIX`:

1. `planar`
2. `ruled`
3. `revolution`
4. `bspline`
5. `nurbs`
6. `sweep`
7. `subdivision`
8. `implicit`
9. `heightmap`
10. `displacement`

Adding or removing a patch family requires updating this ledger and its total
cell count in the same change.

## Cell States

Use exactly one of these values in each operation cell:

| State | Meaning |
| --- | --- |
| `UNASSESSED` | No current completion claim. This is the initial state. |
| `PARTIAL` | Some route, test, or evidence exists, but the completion gate is not satisfied. |
| `EXECUTABLE-COMPLETE` | The operation executes as surfaced CSG and satisfies the full completion gate. |
| `REFUSAL-COMPLETE` | The pair deterministically refuses according to current policy and satisfies the refusal gate. This is test-complete but not executable support. |
| `BLOCKED` | Completion is prevented by a named missing dependency or unresolved policy decision. |

`EXECUTABLE-COMPLETE` and `REFUSAL-COMPLETE` are the only test-complete states.
Only `EXECUTABLE-COMPLETE` counts as implemented CSG capability.

## Completion Gates

### Executable Cell

A cell may become `EXECUTABLE-COMPLETE` only when all of the following are
linked from a cell evidence record:

- deterministic left- and right-family fixture builders exist
- the named operation reaches a surface-native execution route
- the result preserves valid `SurfaceBody` source truth
- topology and geometry assertions pass for the expected result
- provenance and operation ordering assertions pass
- no hidden mesh boolean fallback is attempted
- a focused automated test passes
- reviewable reference evidence is clean when the result is visually geometric

### Refusal Cell

A cell may become `REFUSAL-COMPLETE` only when all of the following are linked
from a cell evidence record:

- deterministic left- and right-family fixture builders exist
- the refusal is explicit and stable
- the diagnostic identifies the operation, ordered family pair, phase, and
  required future capability
- no hidden mesh boolean fallback is attempted
- a focused automated test passes

### Evidence Records

Evidence records should be added below the matrix rather than packed into table
cells. Use the operation-specific ID:

`<pair-id>/<operation>`

For example: `CSG-PAIR-012/difference`.

Each record must link the focused test, fixture, and reference or diagnostic
evidence supporting the chosen state.

## Progress Summary

| Measure | Count |
| --- | ---: |
| Total ordered family pairs | 100 |
| Total operation cells | 300 |
| `EXECUTABLE-COMPLETE` | 0 |
| `REFUSAL-COMPLETE` | 0 |
| `PARTIAL` | 0 |
| `BLOCKED` | 0 |
| `UNASSESSED` | 300 |

Update this summary in the same patch that changes any matrix cell.

## Matrix

| Pair ID | Left operand | Right operand | Union | Difference | Intersection |
| --- | --- | --- | --- | --- | --- |
| CSG-PAIR-001 | `planar` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-002 | `planar` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-003 | `planar` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-004 | `planar` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-005 | `planar` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-006 | `planar` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-007 | `planar` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-008 | `planar` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-009 | `planar` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-010 | `planar` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-011 | `ruled` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-012 | `ruled` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-013 | `ruled` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-014 | `ruled` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-015 | `ruled` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-016 | `ruled` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-017 | `ruled` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-018 | `ruled` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-019 | `ruled` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-020 | `ruled` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-021 | `revolution` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-022 | `revolution` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-023 | `revolution` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-024 | `revolution` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-025 | `revolution` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-026 | `revolution` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-027 | `revolution` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-028 | `revolution` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-029 | `revolution` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-030 | `revolution` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-031 | `bspline` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-032 | `bspline` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-033 | `bspline` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-034 | `bspline` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-035 | `bspline` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-036 | `bspline` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-037 | `bspline` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-038 | `bspline` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-039 | `bspline` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-040 | `bspline` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-041 | `nurbs` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-042 | `nurbs` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-043 | `nurbs` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-044 | `nurbs` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-045 | `nurbs` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-046 | `nurbs` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-047 | `nurbs` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-048 | `nurbs` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-049 | `nurbs` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-050 | `nurbs` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-051 | `sweep` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-052 | `sweep` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-053 | `sweep` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-054 | `sweep` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-055 | `sweep` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-056 | `sweep` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-057 | `sweep` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-058 | `sweep` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-059 | `sweep` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-060 | `sweep` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-061 | `subdivision` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-062 | `subdivision` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-063 | `subdivision` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-064 | `subdivision` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-065 | `subdivision` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-066 | `subdivision` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-067 | `subdivision` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-068 | `subdivision` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-069 | `subdivision` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-070 | `subdivision` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-071 | `implicit` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-072 | `implicit` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-073 | `implicit` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-074 | `implicit` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-075 | `implicit` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-076 | `implicit` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-077 | `implicit` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-078 | `implicit` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-079 | `implicit` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-080 | `implicit` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-081 | `heightmap` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-082 | `heightmap` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-083 | `heightmap` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-084 | `heightmap` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-085 | `heightmap` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-086 | `heightmap` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-087 | `heightmap` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-088 | `heightmap` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-089 | `heightmap` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-090 | `heightmap` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-091 | `displacement` | `planar` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-092 | `displacement` | `ruled` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-093 | `displacement` | `revolution` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-094 | `displacement` | `bspline` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-095 | `displacement` | `nurbs` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-096 | `displacement` | `sweep` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-097 | `displacement` | `subdivision` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-098 | `displacement` | `implicit` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-099 | `displacement` | `heightmap` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |
| CSG-PAIR-100 | `displacement` | `displacement` | `UNASSESSED` | `UNASSESSED` | `UNASSESSED` |

## Evidence Records

No cells have been assessed yet.

## Automated Verification Architecture

The matrix has 300 coverage cells. It does not require 300 separately authored
or manually reviewed models.

### Generated Interaction Atlases

Tests should generate two multi-shell `SurfaceBody` operands for each family
block. Every ordered family pair in the block receives its own spatially
isolated interaction cell:

- the left body contains one representative component for the cell's left
  family
- the right body contains one representative component for the cell's right
  family
- the two components overlap only inside that cell
- large deterministic spacing prevents geometry in one cell from interacting
  with another cell
- every authored input patch in both bodies receives a terminal CSG
  participation record

For example, a left block containing three distinct target families and a
right block containing four distinct target families covers 12 ordered matrix
cells in one CSG call. The families may repeat across isolated shells so every
Cartesian pair gets a dedicated interaction cell. To claim all 12 cells, the
observed target-family interaction graph must contain the complete intended
`3×4` edge set. Simply finding three family names in one operand and four in
the other is not coverage.

Each representative closed component may use auxiliary closure patches, but
those patches are not allowed to be inert. Every source patch must be
intersected, split, classified as retained or discarded, resolved as
coincident/contacting, or named in an explicit refusal. Auxiliary cap or seam
families do not create extra matrix-cell coverage unless they are themselves
the intended family pair for that cell.

### Initial Family Blocks

Use three bounded family groups:

| Group | Families |
| --- | --- |
| A | `planar`, `ruled`, `revolution` |
| B | `bspline`, `nurbs`, `sweep` |
| C | `subdivision`, `implicit`, `heightmap`, `displacement` |

The ordered group Cartesian product produces nine atlas scenarios per
operation:

`A×A`, `A×B`, `A×C`, `B×A`, `B×B`, `B×C`, `C×A`, `C×B`, and `C×C`.

Across union, difference, and intersection, the base suite therefore needs 27
generated CSG invocations while still accounting for all 300 matrix cells.
Smaller blocks may be used when a failure needs better isolation; splitting a
block must not change the required cell set.

### Per-Cell Execution Witness

Passing the whole atlas body is insufficient. The CSG pipeline must expose or
allow the test harness to assemble a structured witness for every intended
cell:

- operation
- ordered left/right family
- left/right body, shell, and target-patch identities
- spatial interaction-cell ID and bounds
- contact classification
- selected solver route and support state
- intersection, fragment, or refusal record
- result-patch provenance for surviving and generated patches
- validity-gate outcome
- no-hidden-mesh-fallback evidence

A family pair appearing in the operand inventory does not count as exercised.
The intended target patch pair must appear in execution, refusal, or result
provenance.

### Whole-Body Patch Participation Gate

Before each operation, the harness must inventory every source patch as a
stable `(operand, shell, patch)` reference. After execution, it must reconcile
that inventory against pair interactions, fragment classification, result
provenance, and refusal records.

Every input patch must have at least one terminal disposition:

| Disposition | Required evidence |
| --- | --- |
| `intersected` | The patch appears in an executed patch-pair intersection or contact record. |
| `retained` | One or more result fragments trace back to the source patch. |
| `discarded` | Operation selection explicitly classifies all source fragments out of the result. |
| `coincident-resolved` | Coincident ownership or contact policy names the source patch and resolution. |
| `refused` | A structured refusal names the source patch or its containing family route. |

The scenario fails if any input patch is unvisited, silently dropped,
unclassified, or present only in the pre-operation family inventory.

Two independent coverage gates therefore apply:

1. **Pair-edge coverage:** every intended ordered family-pair cell has an
   actual interaction/refusal witness.
2. **Patch-node coverage:** every authored patch instance in both operand
   bodies has a terminal disposition.

An atlas invocation counts only when both gates pass. In graph terms, the
intended family-pair edges must be present and every concrete source-patch node
must be accounted for.

### Automatic Result Oracles

Each interaction cell is validated independently using its known spatial
bounds and deterministic fixture oracle. An executable result must pass:

- surface-native runtime validity
- complete whole-body source-patch participation
- closed-shell, seam, adjacency, trim, and non-manifold checks
- result provenance completeness
- deterministic repeatability
- expected local component count and bounds
- predefined inside/outside witness points for the operation's membership rule
- derived watertight tessellation and volume checks at the verification
  boundary only
- operation identities within declared tolerance:
  - union membership equals `left or right`
  - intersection membership equals `left and right`
  - difference membership equals `left and not right`
- no mesh boolean execution

Union and intersection results should also agree under operand reversal.
Difference must preserve direction and is checked against its own reversed
matrix cell.

A refused result must pass the refusal completion gate already defined above.
No visual approval is required for each generated cell. A small separate
human-reviewed showcase set may remain useful for presentation quality, but it
is not the exhaustive correctness oracle.

### Coverage Aggregation

The test run should produce one machine-readable coverage report keyed by
`<pair-id>/<operation>`. The report must:

- list the atlas scenario that exercised each cell
- record executable, refusal, partial, blocked, or missing status
- reject duplicate claims that disagree
- reject intended cells with no execution witness
- reject executed patch-pair interactions that are absent from this matrix
- summarize all 300 cells without editing source files during ordinary tests

The Markdown ledger can then be refreshed from a successful report in an
explicit maintenance command. Tests should fail on ledger/code drift rather
than silently rewriting the ledger.

## Parametrized Test Contract

The eventual exhaustive test suite should:

1. derive the current family list from `PATCH_FAMILY_CAPABILITY_MATRIX`
2. generate the Cartesian product of left family, right family, and operation
3. partition that Cartesian product into generated atlas blocks
4. require exactly one agreeing coverage result for every matrix cell
5. use the ID `<pair-id>/<operation>` in coverage and failure output
6. fail if the code family inventory and this matrix diverge
7. distinguish executable success from policy-complete refusal
8. prove that no case reaches a mesh boolean fallback

The initial base shape is 27 atlas invocations covering 300 cells, not 300
individually authored models. Geometry variations such as disjoint,
overlapping, contained, tangent, and coincident operands are additional
coverage dimensions and must not be mistaken for replacement cells in this
one-to-one family matrix.

## Scope Boundary

This matrix tracks family-pair operation completeness only. It does not claim
coverage of:

- multi-operand CSG
- every geometric relationship or degeneracy
- numerical tolerance sweeps
- performance or stress limits
- UI behavior

Those concerns require separate matrices or test layers after this base
one-to-one matrix is covered.

## Acceptance

This ledger is structurally complete when:

- every current patch family appears once as each left/right ordered pair
- every ordered pair has union, difference, and intersection cells
- the total remains synchronized with the code-owned patch-family inventory

The CSG one-to-one program is complete when all 300 cells are either
`EXECUTABLE-COMPLETE` or `REFUSAL-COMPLETE`, every state has linked evidence,
and the exhaustive parametrized test contract passes.
