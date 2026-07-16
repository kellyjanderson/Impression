# Loft Primitive Cut-Shell Spec Promotion And Sizing Ledger

Scope: Surface Specs 423-431 and paired test specifications promoted from the
loft primitive cut-shell child ACD manifests.

## Iteration 1 - 2026-07-16

Spec files reviewed this iteration:

- `project/release-0.1.0a/specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- `project/release-0.1.0a/specifications/surface-429-loft-primitive-public-cut-executor-integration-v1_0.md`
- `project/release-0.1.0a/specifications/surface-430-loft-csg-reference-geometry-handoff-proof-v1_0.md`
- `project/release-0.1.0a/specifications/surface-431-loft-csg-section-evidence-readiness-handoff-v1_0.md`

Paired test spec files reviewed this iteration:

- `project/release-0.1.0a/test-specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-429-loft-primitive-public-cut-executor-integration-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-430-loft-csg-reference-geometry-handoff-proof-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-431-loft-csg-section-evidence-readiness-handoff-v1_0.md`

Files written before barrier:

- the 18 spec and paired test spec files listed above
- `project/release-0.1.0a/specifications/README.md`
- `project/release-0.1.0a/test-specifications/README.md`
- `project/release-0.1.0a/planning/progression.md`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

IWU values before and after update:

- Surface Spec 423: manifest score 19 -> 1 IWU.
- Surface Spec 424: manifest score 19 -> 1 IWU.
- Surface Spec 425: manifest score 19 -> 1 IWU.
- Surface Spec 426: manifest score 19 -> 1 IWU.
- Surface Spec 427: manifest score 19 -> 1 IWU.
- Surface Spec 428: manifest score 19.5 -> 1 IWU.
- Surface Spec 429: manifest score 19.5 -> 1 IWU.
- Surface Spec 430: manifest score 22 -> 1 IWU.
- Surface Spec 431: manifest score 22 -> 1 IWU.

Split decision and artifacts:

- No implementation spec requires a child split.
- Each final implementation leaf is 1 IWU.
- Parent Surface Spec 422 is superseded for active implementation sequencing
  by Surface Specs 423-429.
- Parent Surface Spec 407 is superseded for active implementation sequencing
  by Surface Spec 430.
- Parent Surface Spec 418 is superseded for active implementation sequencing
  by Surface Spec 431.

Child re-review status:

- Not applicable. No child specs were created by a split during this sizing
  iteration.

Next scope after readback:

- Re-read this ledger and the 18 written spec/test-spec artifacts.
- Run one post-readback fixed-point sizing iteration.

## Iteration 2 - 2026-07-16

Spec files reviewed this iteration:

- Surface Specs 423-431 in `project/release-0.1.0a/specifications/`
- paired Surface Spec 423-431 test specifications in
  `project/release-0.1.0a/test-specifications/`
- `project/release-0.1.0a/specifications/README.md`
- `project/release-0.1.0a/test-specifications/README.md`
- `project/release-0.1.0a/planning/progression.md`

Files written before barrier:

- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

IWU values before and after update:

- Surface Specs 423-431: 1 IWU each -> 1 IWU each.
- Paired test specs 423-431: 1 verification leaf each -> 1 verification leaf each.

Split decision and artifacts:

- No implementation spec requires a child split.
- No paired test specification requires a child split.
- No unresolved parent coverage gaps were found for the promoted leaf set.
- Split artifacts created: none.

Child re-review status:

- Not applicable. No child specs were created by this fixed-point iteration.

Next scope after readback:

- Fixed point reached for this spec promotion scope.
- Next workflow step is implementation in progression order, starting with
  Surface Spec 423.

## Iteration 3 - 2026-07-16

Spec files reviewed this iteration:

- `project/release-0.1.0a/specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`

Files written before barrier:

- `project/release-0.1.0a/specifications/surface-424a-loft-patch-local-source-curve-inversion-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425a-loft-primitive-cap-support-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426a-loft-primitive-operation-fragment-retention-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426b-loft-primitive-result-topology-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427a-loft-primitive-seam-use-pairing-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428a-loft-primitive-runtime-validity-checker-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428c-loft-primitive-no-hidden-mesh-acceptance-proof-v1_0.md`
- paired test specifications for Surface Specs 424a-428c in `project/release-0.1.0a/test-specifications/`
- `project/release-0.1.0a/specifications/README.md`
- `project/release-0.1.0a/test-specifications/README.md`
- `project/release-0.1.0a/planning/progression.md`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

IWU values before and after update:

- Surface Spec 424: 1 IWU claimed leaf -> 3 IWU branch rollup, superseded by three 1-IWU leaves.
- Surface Spec 425: 1 IWU claimed leaf -> 3 IWU branch rollup, superseded by three 1-IWU leaves.
- Surface Spec 426: 1 IWU claimed leaf -> 3 IWU branch rollup, superseded by three 1-IWU leaves.
- Surface Spec 427: 1 IWU claimed leaf -> 3 IWU branch rollup, superseded by three 1-IWU leaves.
- Surface Spec 428: 1 IWU claimed leaf -> 3 IWU branch rollup, superseded by three 1-IWU leaves.

Split decision and artifacts:

- Surface Spec 424 split into 424a source curve inversion, 424b loop closure and boundary participation, and 424c degeneracy diagnostics.
- Surface Spec 425 split into 425a cap support classification, 425b generated cap record construction, and 425c cap loop pairing and diagnostics.
- Surface Spec 426 split into 426a operation fragment retention, 426b result topology classification, and 426c topology orientation and refusal diagnostics.
- Surface Spec 427 split into 427a seam/use pairing, 427b candidate shell assembly, and 427c adjacency rebuild diagnostics.
- Surface Spec 428 split into 428a runtime validity checker, 428b persistence and tessellation readiness, and 428c no-hidden-mesh acceptance proof.

Child re-review status:

- Pending. The next step is a closed-file readback and critical review of Surface Specs 424a-428c and their paired test specifications.

Next scope after readback:

- Re-read the 15 new child implementation specs, 15 paired test specifications, the two README indexes, progression, and this ledger.
- If any child still contains more than one IWU or has incomplete parent coverage, split it and add another ledger entry.

## Iteration 4 - 2026-07-16

Spec files reviewed this iteration:

- Surface Specs 424a-428c in `project/release-0.1.0a/specifications/`
- paired Surface Spec 424a-428c test specifications in `project/release-0.1.0a/test-specifications/`
- superseded parent Surface Spec 424-428 test specifications in `project/release-0.1.0a/test-specifications/`
- `project/release-0.1.0a/specifications/README.md`
- `project/release-0.1.0a/test-specifications/README.md`
- `project/release-0.1.0a/planning/progression.md`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

Files written before barrier:

- `project/release-0.1.0a/test-specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- `project/release-0.1.0a/test-specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

IWU values before and after update:

- Surface Specs 424a-428c: 1 IWU each -> 1 IWU each.
- Paired test specs 424a-428c: one verification leaf each -> one verification leaf each.
- Superseded parent Surface Spec 424-428 test specifications: parent-test status corrected from proposed/canonical to superseded parent.

Split decision and artifacts:

- No split specifications were created in this iteration.
- No child implementation spec requires a further child split.
- No paired child test specification requires a further child split.
- No unresolved parent coverage gaps were found in Surface Specs 424a-428c.

Child re-review status:

- Complete. Surface Specs 424a-428c remain final active implementation leaves.

Next scope after readback:

- Fixed point reached for the Surface Spec 424-428 split family.
- Active implementation sequencing now proceeds through Surface Spec 423, Surface Specs 424a-428c, then Surface Specs 429-431.

## Iteration 5 - 2026-07-16

Spec files reviewed this iteration:

- `project/release-0.1.0a/specifications/surface-424a-loft-patch-local-source-curve-inversion-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425a-loft-primitive-cap-support-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426a-loft-primitive-operation-fragment-retention-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426b-loft-primitive-result-topology-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427a-loft-primitive-seam-use-pairing-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428a-loft-primitive-runtime-validity-checker-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428c-loft-primitive-no-hidden-mesh-acceptance-proof-v1_0.md`
- paired Surface Spec 424a-428c test specifications in `project/release-0.1.0a/test-specifications/`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

Files written before barrier:

- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

Content-based IWU verification:

- Surface Spec 424a: 1 IWU verified from one outcome, patch-local source curve inversion; loop closure, degeneracy, and cap construction are explicitly out of scope.
- Surface Spec 424b: 1 IWU verified from one outcome, closed cut-loop records with boundary participation; inversion, degeneracy policy, and cap construction are explicitly out of scope.
- Surface Spec 424c: 1 IWU verified from one outcome, degeneracy/refusal diagnostics; construction stages are explicitly out of scope.
- Surface Spec 425a: 1 IWU verified from one outcome, cap support/refusal classification; generated record construction and pairing are explicitly out of scope.
- Surface Spec 425b: 1 IWU verified from one outcome, generated cap records for already-supported caps; support decisions and pairing are explicitly out of scope.
- Surface Spec 425c: 1 IWU verified from one outcome, exact cap-loop pairing/readiness; support classification, cap construction, and shell assembly are explicitly out of scope.
- Surface Spec 426a: 1 IWU verified from one outcome, operation-specific fragment retention; topology classification and assembly are explicitly out of scope.
- Surface Spec 426b: 1 IWU verified from one outcome, result topology classification; retention, orientation, and assembly are explicitly out of scope.
- Surface Spec 426c: 1 IWU verified from one outcome, orientation readiness/refusal diagnostics; topology creation and seam/use pairing are explicitly out of scope.
- Surface Spec 427a: 1 IWU verified from one outcome, seam/use pairing; candidate shell assembly and adjacency rebuild are explicitly out of scope.
- Surface Spec 427b: 1 IWU verified from one outcome, candidate shell assembly; pairing, adjacency rebuild, and validity are explicitly out of scope.
- Surface Spec 427c: 1 IWU verified from one outcome, adjacency rebuild diagnostics; candidate shell assembly, validity, and persistence are explicitly out of scope.
- Surface Spec 428a: 1 IWU verified from one outcome, runtime validity checking; adjacency rebuild, persistence, and no-hidden-mesh proof are explicitly out of scope.
- Surface Spec 428b: 1 IWU verified with caution from one outcome, accepted-result persistence gate. Tessellation readiness remains metadata on that same gate and does not introduce tessellation execution or a second artifact.
- Surface Spec 428c: 1 IWU verified from one outcome, no-hidden-mesh acceptance proof; validity, persistence, and fixture generation are explicitly out of scope.

Split decision and artifacts:

- No split specifications were created in this iteration.
- No child implementation spec requires a further child split after content-based IWU verification.
- No paired child test specification requires a further child split.
- No unresolved parent coverage gaps were found in Surface Specs 424a-428c.

Child re-review status:

- Complete after content-based scoring. Surface Specs 424a-428c remain final active implementation leaves.

Next scope after readback:

- Fixed point remains reached for the Surface Spec 424-428 split family.
- Active implementation sequencing remains Surface Spec 423, Surface Specs 424a-428c, then Surface Specs 429-431.

## Iteration 6 - 2026-07-16

Spec files reviewed this iteration:

- `project/release-0.1.0a/specifications/surface-424a-loft-patch-local-source-curve-inversion-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md`
- `project/release-0.1.0a/specifications/surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425a-loft-primitive-cap-support-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md`
- `project/release-0.1.0a/specifications/surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426a-loft-primitive-operation-fragment-retention-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426b-loft-primitive-result-topology-classification-v1_0.md`
- `project/release-0.1.0a/specifications/surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427a-loft-primitive-seam-use-pairing-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md`
- `project/release-0.1.0a/specifications/surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428a-loft-primitive-runtime-validity-checker-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md`
- `project/release-0.1.0a/specifications/surface-428c-loft-primitive-no-hidden-mesh-acceptance-proof-v1_0.md`
- paired Surface Spec 424a-428c test specifications in `project/release-0.1.0a/test-specifications/`

Files written before barrier:

- Surface Spec 424a-428c implementation specs in `project/release-0.1.0a/specifications/`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

ACD feedback artifacts created or updated:

- None. Existing ACD ancestors were present and linked:
  - `project/release-0.1.0a/architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
  - `project/release-0.1.0a/architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
  - `project/release-0.1.0a/architecture/acd-loft-primitive-seam-shell-validity-execution.md`

Prior recorded IWU/Review Score challenged:

- Prior IWU claims from Iteration 5 were challenged rather than copied.
- Prior Review Score values were missing from Surface Specs 424a-428c and were treated as a readiness defect.
- Checks performed: undercounted responsibilities, hidden coupling, missing readiness fields, omitted UI/control inventory, uncounted reuse work, unrecorded routes, unresolved architecture prerequisites, ambiguous verification, and parent-only responsibility leakage.

Fresh adversarial IWU and Review Score:

- Surface Spec 424a: 1 IWU, Review Score 12.5.
- Surface Spec 424b: 1 IWU, Review Score 14.5.
- Surface Spec 424c: 1 IWU, Review Score 12.5.
- Surface Spec 425a: 1 IWU, Review Score 12.5.
- Surface Spec 425b: 1 IWU, Review Score 12.5.
- Surface Spec 425c: 1 IWU, Review Score 12.5.
- Surface Spec 426a: 1 IWU, Review Score 12.5.
- Surface Spec 426b: 1 IWU, Review Score 13.5.
- Surface Spec 426c: 1 IWU, Review Score 13.5.
- Surface Spec 427a: 1 IWU, Review Score 14.5.
- Surface Spec 427b: 1 IWU, Review Score 13.5.
- Surface Spec 427c: 1 IWU, Review Score 14.5.
- Surface Spec 428a: 1 IWU, Review Score 14.5.
- Surface Spec 428b: 1 IWU, Review Score 15.5, with implementation caution that tessellation readiness must remain metadata-only.
- Surface Spec 428c: 1 IWU, Review Score 11.5.

Split decision and child spec paths:

- No split specifications were created in this iteration.
- No child implementation spec requires a further child split.
- No paired child test specification requires a further child split.

Parent coverage status:

- Surface Spec 424 is 100% covered by Surface Specs 424a-424c.
- Surface Spec 425 is 100% covered by Surface Specs 425a-425c.
- Surface Spec 426 is 100% covered by Surface Specs 426a-426c.
- Surface Spec 427 is 100% covered by Surface Specs 427a-427c.
- Surface Spec 428 is 100% covered by Surface Specs 428a-428c.
- Parent coverage status was added to Surface Specs 426a-428c so the coverage proof is visible in every child spec.

Readiness defects resolved or still blocking:

- Resolved: missing Review Score sections in Surface Specs 424a-428c.
- Resolved: missing explicit parent coverage status in Surface Specs 426a-428c.
- Remaining blockers: none found in this review scope.

Child re-review status:

- Complete. Surface Specs 424a-428c remain final active implementation leaves after adversarial score and IWU recount.

Next scope after readback:

- Re-read this ledger entry and the changed Surface Spec 424a-428c files.
- If the readback finds no new split pressure, no missing Review Score, and no coverage gap, the fixed point remains reached.

## Iteration 7 - 2026-07-16

Spec files reviewed this iteration:

- Surface Specs 424a-428c in `project/release-0.1.0a/specifications/`
- paired Surface Spec 424a-428c test specifications in `project/release-0.1.0a/test-specifications/`
- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

Files written before barrier:

- `project/release-0.1.0a/spec-refinement-history/loft-primitive-cut-shell-specs.md`

ACD feedback artifacts created or updated:

- None.

Prior recorded IWU/Review Score challenged:

- Iteration 6 IWU and Review Score values were challenged by readback of every active child spec's scope, exclusions, parent coverage, and Review Score section.
- Undercount risks checked: multi-case classifiers, hidden tessellation readiness expansion, two-module helper use, route-level verification, and parent-only responsibility leakage.

Fresh adversarial IWU and Review Score:

- Surface Spec 424a: 1 IWU, Review Score 12.5.
- Surface Spec 424b: 1 IWU, Review Score 14.5.
- Surface Spec 424c: 1 IWU, Review Score 12.5.
- Surface Spec 425a: 1 IWU, Review Score 12.5.
- Surface Spec 425b: 1 IWU, Review Score 12.5.
- Surface Spec 425c: 1 IWU, Review Score 12.5.
- Surface Spec 426a: 1 IWU, Review Score 12.5.
- Surface Spec 426b: 1 IWU, Review Score 13.5.
- Surface Spec 426c: 1 IWU, Review Score 13.5.
- Surface Spec 427a: 1 IWU, Review Score 14.5.
- Surface Spec 427b: 1 IWU, Review Score 13.5.
- Surface Spec 427c: 1 IWU, Review Score 14.5.
- Surface Spec 428a: 1 IWU, Review Score 14.5.
- Surface Spec 428b: 1 IWU, Review Score 15.5.
- Surface Spec 428c: 1 IWU, Review Score 11.5.

Split decision and child spec paths:

- No split specifications were created in this iteration.
- No child implementation spec requires a further child split.
- No paired child test specification requires a further child split.

Parent coverage status:

- Surface Specs 424-428 remain 100% covered by Surface Specs 424a-428c.
- No parent-only, partial, or missing responsibilities were found.

Readiness defects resolved or still blocking:

- Remaining blockers: none found in this review scope.

Child re-review status:

- Complete. Fixed point reached after readback.

Next scope after readback:

- Active implementation sequencing remains Surface Spec 423, Surface Specs 424a-428c, then Surface Specs 429-431.
