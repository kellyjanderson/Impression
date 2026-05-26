# 0.1.0.a Progression

This progression orders the `0.1.0.a` final leaf specifications in the
recommended implementation sequence.

Only final leaf specifications appear here.

Each implementation prompt should normally target one tranche of five specs.
Paired test-spec work is implied for each leaf and should be completed together
with the implementation.

## Tranche 01: B-Spline and Fit Foundations

- [x] [Feature Spec 01A: B-Spline Curve Primitive Representation and Ownership](specifications/feature-01a-b-spline-curve-primitive-representation-and-ownership-v1_0.md)
- [x] [Feature Spec 01B: B-Spline Curve Evaluation, Sampling, and Closure Contract](specifications/feature-01b-b-spline-curve-evaluation-sampling-and-closure-contract-v1_0.md)
- [x] [Feature Spec 02A1: Parameterization Policy Records](specifications/feature-02a1-parameterization-policy-records-v1_0.md)
- [x] [Feature Spec 02A2: Knot Count and Knot Placement Policy Records](specifications/feature-02a2-knot-count-and-knot-placement-policy-records-v1_0.md)
- [x] [Feature Spec 02A3: Fit Configuration Record Contract](specifications/feature-02a3-fit-configuration-record-contract-v1_0.md)

## Tranche 02: Progression Core and Fit Outcomes

- [x] [Feature Spec 02B: Fit Residual, Acceptance, and Refusal Reporting](specifications/feature-02b-fit-residual-acceptance-and-refusal-reporting-v1_0.md)
- [x] [Feature Spec 08A: Path-Backed Progression Object and Provenance Contract](specifications/feature-08a-path-backed-progression-object-and-provenance-contract-v1_0.md)
- [x] [Feature Spec 08B1: Station Attachment to Path-Backed Progression](specifications/feature-08b1-station-attachment-to-path-backed-progression-v1_0.md)
- [x] [Feature Spec 08B2: Progression Transport Semantics Contract](specifications/feature-08b2-progression-transport-semantics-contract-v1_0.md)
- [x] [Feature Spec 08B3: Progression Twist and Scale Semantic Slots](specifications/feature-08b3-progression-twist-and-scale-semantic-slots-v1_0.md)

## Tranche 03: Hidden Control Structure and Evidence Inputs

- [x] [Feature Spec 04A: Internal Control-Station Representation and Provenance](specifications/feature-04a-internal-control-station-representation-and-provenance-v1_0.md)
- [x] [Feature Spec 04B: Planner Consumption Boundary for Hidden Control Stations](specifications/feature-04b-planner-consumption-boundary-for-hidden-control-stations-v1_0.md)
- [x] [Feature Spec 03A: Dense Loft Evidence Descriptor Preparation for Curve Fitting](specifications/feature-03a-dense-loft-evidence-descriptor-preparation-for-curve-fitting-v1_0.md)
- [x] [Feature Spec 06A1: Descriptor Record Families for Curve-Intent Inference](specifications/feature-06a1-descriptor-record-families-for-curve-intent-inference-v1_0.md)
- [x] [Feature Spec 06A2: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference](specifications/feature-06a2-span-local-evidence-assembly-and-ordering-for-curve-intent-inference-v1_0.md)

## Tranche 04: Candidate Fitting and Shared-Trajectory Inference

- [x] [Feature Spec 03B: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture](specifications/feature-03b-station-derived-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)
- [x] [Feature Spec 03C: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture](specifications/feature-03c-shared-trajectory-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)
- [x] [Feature Spec 06B: Curve-Intent Candidate Classification and Confidence Posture](specifications/feature-06b-curve-intent-candidate-classification-and-confidence-posture-v1_0.md)
- [x] [Feature Spec 07A1: Shared Whole-Loft Trajectory Candidate Generation](specifications/feature-07a1-shared-whole-loft-trajectory-candidate-generation-v1_0.md)
- [x] [Feature Spec 07A2: Shared Whole-Loft Trajectory Confidence and Refusal Posture](specifications/feature-07a2-shared-whole-loft-trajectory-confidence-and-refusal-posture-v1_0.md)

## Tranche 05: Guidance and Control-Station Inference Results

- [x] [Feature Spec 07B1: Explicit Shared Guidance Attachment Record Contract](specifications/feature-07b1-explicit-shared-guidance-attachment-record-contract-v1_0.md)
- [x] [Feature Spec 07B2: Planner Consumption Boundaries for Explicit Shared Guidance](specifications/feature-07b2-planner-consumption-boundaries-for-explicit-shared-guidance-v1_0.md)
- [x] [Feature Spec 05A1: Reduced Progression Bundle Shape and Replay Contract](specifications/feature-05a1-reduced-progression-bundle-shape-and-replay-contract-v1_0.md)
- [x] [Feature Spec 05A2: Retained Station Classification and Diagnostic Association Contract](specifications/feature-05a2-retained-station-classification-and-diagnostic-association-contract-v1_0.md)
- [x] [Feature Spec 05B: Structural Preservation and Inference Refusal Posture](specifications/feature-05b-structural-preservation-and-inference-refusal-posture-v1_0.md)

## Tranche 06: Diagnostics and Explainability

- [x] [Feature Spec 09A1: Shared Inference Diagnostic Bundle Schema](specifications/feature-09a1-shared-inference-diagnostic-bundle-schema-v1_0.md)
- [x] [Feature Spec 09A2: Inference Diagnostic Bundle Population and Reuse Posture](specifications/feature-09a2-inference-diagnostic-bundle-population-and-reuse-posture-v1_0.md)
- [x] [Feature Spec 09B1: Developer-Facing Inference Explainability and Inspection Contract](specifications/feature-09b1-developer-facing-inference-explainability-and-inspection-contract-v1_0.md)
- [x] [Feature Spec 09B2: Downstream Inference Reporting and Uncertainty Communication Contract](specifications/feature-09b2-downstream-inference-reporting-and-uncertainty-communication-contract-v1_0.md)

## Ordering Rationale

The order follows the current dependency read:

1. build the B-spline and fit-policy substrate first
2. establish progression as the path-backed semantic home for later inference
3. add hidden control structure and durable evidence inputs
4. add fit generation and shared-trajectory inference
5. add explicit guidance and control-station inference result contracts
6. finish with cross-cutting diagnostics and explainability hardening

This keeps early prompts focused on foundational model objects and contracts,
and delays the more cross-cutting reporting work until the underlying inference
branches exist.
