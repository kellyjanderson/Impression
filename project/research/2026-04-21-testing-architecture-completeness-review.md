# Testing Architecture Completeness Review

## Topic

Pre-implementation completeness review of the top-level testing architecture
branch, with emphasis on hidden work that is still unrepresented or bundled too
coarsely for honest execution.

Related documents:

- [Testing Architecture](../architecture/testing-architecture.md)
- [Model Output Reference Verification](../architecture/model-output-reference-verification.md)
- [Computer Vision Verification Architecture](../architecture/computer-vision-verification-architecture.md)
- [Testing Spec 01: Testing Tooling and Verification Program](../specifications/testing-01-testing-tooling-and-verification-program-v1_0.md)

## Findings

### 1. The branch structure is materially better than the earlier feature-owned layout

The current testing branch now captures an important architectural truth:

- testing tooling is a top-level concern
- CV verification is a testing subsystem
- feature trunks should consume testing tools rather than own testing-tool
  structure

That structural correction should remain.

### 2. The testing branch is still incomplete because reusable reference-artifact tooling is not yet owned by testing

The top-level testing architecture explicitly claims:

- reusable verification harnesses
- baseline/reference lifecycle rules
- cross-feature diagnostic lanes

But the actual reusable reference-artifact implementation/spec branch still
remains under:

- [Surface Spec 106](../specifications/surface-106-reference-artifact-regression-suite-v1_0.md)

with a feature-program backlink through the surface replacement branch.

That means the testing architecture is still missing one of its own major child
tooling branches.

The right long-term state is:

- top-level testing tooling owns reusable reference-artifact infrastructure
- feature trunks depend on that infrastructure as a tool they use

### 3. The current testing leaves still hide multiple implementation rounds

Several of the `testing-*` leaves are valuable as architectural buckets, but
they are not yet truly implementation-sized.

The main cases are:

- `testing-02`
  It still bundles fixture schema, full harness layering, grouped completeness,
  decision taxonomy, uncertainty posture, and invalidation.
- `testing-03`
  It still hides the initial executable scope of OCR/glyph tooling.
- `testing-04`
  It still lacks the actual comparison algorithm contract for slice
  equivalence.
- `testing-06`
  It still does not define what object-view truth comparison actually is.
- `testing-07`
  It still assumes a cross-space anchoring contract that is not yet explicit.
- `testing-08`
  It still leaves the "honest triptych" problem unresolved.

Only the camera/framing lane currently reads close to a true first executable
leaf.

### 4. Reference-artifact and CV lanes are related, but they are not the same concern

The branch should preserve a clean split:

- reference artifacts are baseline/change-detection infrastructure
- CV lanes are interpretation and semantic-truth infrastructure

Those systems interact closely, but they should not collapse into one
undifferentiated branch.

Reference-artifact ownership belongs with top-level testing infrastructure.
CV lanes should layer on top of that infrastructure rather than redefine it.

### 5. The most important hidden work is contract-definition work, not implementation polish

The missing work is mostly not "write more code."
It is:

- moving remaining top-level testing infrastructure under the testing trunk
- separating shared tooling contracts from lane-specific algorithms
- defining honest first executable scope for each lane
- defining the comparison semantics and tolerances that convert a lane from an
  idea into an implementable tool

Without that work, implementation would be forced to make architecture-level
decisions on the fly.

## Recommended Next Refinement Moves

### 1. Move reusable reference-artifact tooling into the testing branch

The top-level testing branch should absorb the reusable reference-artifact
tooling/spec work that is still represented by `surface-106`.

That does not mean feature trunks stop using it.
It means the tool ownership becomes structurally correct.

### 2. Split the shared CV tooling leaf

`testing-02` should likely split into at least:

- shared fixture/result contract
- shared harness pipeline and artifact-lifecycle integration

That would separate declarative test contracts from executable harness plumbing.

### 3. Add initial executable-scope leaves for each CV lane

The current CV leaves are still too broad.
They need bounded first executable scope, for example:

- text OCR first supported script/fixture family
- slice classifier first supported normalization/comparison method
- object-view first supported authoritative derived product
- handedness first supported witness family
- triptych first honest panel/framing contract

### 4. Make cross-space anchoring explicit for handedness work

If handedness is supposed to compare modeling, export, and viewer space, then
the testing branch needs an explicit contract for how those spaces are aligned
before classification runs.

### 5. Define honesty rules for diagnostic presentation

Triptych and multi-panel tooling must explicitly state:

- when panels may be independently rendered
- when shared scene/camera/scale is required
- what diagnostic panels may prove
- what they may never be treated as proving

That honesty boundary should be explicit before implementation.

## Implications

- The testing architecture is now structurally credible, but not yet complete
  enough for broad implementation.
- The next work should be another refinement pass, not direct implementation of
  the whole branch.
- The main risk is not coding difficulty alone; it is allowing tooling
  boundaries and truth contracts to remain implicit.

