# Surface and Loft Program Progression

This progression document sequences implementation of the surface-first
internal-model program together with the active loft work that depends on it.

Only final leaf specifications appear here.

## Core Functionality

### Surface Kernel Contracts

- [x] [Surface Spec 59: SurfaceBody Ownership and Containment Contract (v1.0)](../specifications/surface-59-surfacebody-ownership-containment-v1_0.md)
- [x] [Surface Spec 60: SurfaceShell Multiplicity and Connectivity Policy (v1.0)](../specifications/surface-60-surfaceshell-multiplicity-connectivity-v1_0.md)
- [x] [Surface Spec 61: Deterministic Body/Shell Traversal and Ordering Rules (v1.0)](../specifications/surface-61-body-shell-traversal-ordering-v1_0.md)
- [x] [Surface Spec 62: SurfacePatch Interface and Required Methods (v1.0)](../specifications/surface-62-surfacepatch-interface-required-methods-v1_0.md)
- [x] [Surface Spec 63: Patch Evaluation Semantics and Parameter Queries (v1.0)](../specifications/surface-63-patch-evaluation-semantics-v1_0.md)
- [x] [Surface Spec 64: Family-Agnostic Patch Properties and Capability Flags (v1.0)](../specifications/surface-64-family-agnostic-patch-properties-v1_0.md)
- [x] [Surface Spec 65: Required V1 Patch Families (v1.0)](../specifications/surface-65-required-v1-patch-families-v1_0.md)
- [x] [Surface Spec 66: Deferred Patch Families and Explicit Exclusions (v1.0)](../specifications/surface-66-deferred-patch-families-exclusions-v1_0.md)
- [x] [Surface Spec 67: Patch-Family to Feature Coverage Matrix (v1.0)](../specifications/surface-67-patch-family-feature-coverage-v1_0.md)

### Parameter Domains and Trims

- [x] [Surface Spec 68: Patch Domain Existence and Shape Contract (v1.0)](../specifications/surface-68-patch-domain-existence-shape-v1_0.md)
- [x] [Surface Spec 69: Parameter Domain Normalization Policy (v1.0)](../specifications/surface-69-parameter-domain-normalization-policy-v1_0.md)
- [x] [Surface Spec 70: Downstream Parameter-Space Assumptions Contract (v1.0)](../specifications/surface-70-downstream-parameter-space-assumptions-v1_0.md)
- [x] [Surface Spec 71: Trim Loop Data Structure Contract (v1.0)](../specifications/surface-71-trim-loop-data-structure-v1_0.md)
- [x] [Surface Spec 72: Trim Ownership and Attachment Policy (v1.0)](../specifications/surface-72-trim-ownership-attachment-policy-v1_0.md)
- [x] [Surface Spec 73: Outer and Inner Trim Categorization Rules (v1.0)](../specifications/surface-73-outer-inner-trim-categorization-v1_0.md)
- [x] [Surface Spec 74: Trim Validity Conditions and Failure Modes (v1.0)](../specifications/surface-74-trim-validity-failure-modes-v1_0.md)
- [x] [Surface Spec 75: Trim Orientation Semantics (v1.0)](../specifications/surface-75-trim-orientation-semantics-v1_0.md)
- [x] [Surface Spec 76: Boundary Inclusion and Interior Meaning Rules (v1.0)](../specifications/surface-76-boundary-inclusion-interior-meaning-v1_0.md)

### Adjacency and Seams

- [x] [Surface Spec 77: Patch Adjacency Record Structure (v1.0)](../specifications/surface-77-patch-adjacency-record-structure-v1_0.md)
- [x] [Surface Spec 78: Adjacency Lookup and Navigation Semantics (v1.0)](../specifications/surface-78-adjacency-lookup-navigation-v1_0.md)
- [x] [Surface Spec 79: Adjacency Identity and Index Stability Rules (v1.0)](../specifications/surface-79-adjacency-identity-index-stability-v1_0.md)
- [x] [Surface Spec 80: Explicit Versus Implicit Seam Representation Policy (v1.0)](../specifications/surface-80-explicit-vs-implicit-seam-policy-v1_0.md)
- [x] [Surface Spec 81: Seam Identity Contract (v1.0)](../specifications/surface-81-seam-identity-contract-v1_0.md)
- [x] [Surface Spec 82: Seam Ownership and Source-of-Truth Policy (v1.0)](../specifications/surface-82-seam-ownership-source-of-truth-v1_0.md)
- [x] [Surface Spec 83: Shared-Boundary Validity Rules (v1.0)](../specifications/surface-83-shared-boundary-validity-rules-v1_0.md)
- [x] [Surface Spec 84: Open Boundary Versus Shared Boundary Distinction (v1.0)](../specifications/surface-84-open-vs-shared-boundary-v1_0.md)
- [x] [Surface Spec 85: Surface Continuity Metadata Contract (v1.0)](../specifications/surface-85-surface-continuity-metadata-v1_0.md)

### Transforms, Metadata, and Identity

- [x] [Surface Spec 86: Default Attached-Transform Policy (v1.0)](../specifications/surface-86-default-attached-transform-policy-v1_0.md)
- [x] [Surface Spec 87: Geometry Baking Triggers and Required Cases (v1.0)](../specifications/surface-87-geometry-baking-triggers-v1_0.md)
- [x] [Surface Spec 88: Downstream Transformed-Object Assumptions (v1.0)](../specifications/surface-88-downstream-transformed-object-assumptions-v1_0.md)
- [x] [Surface Spec 89: Metadata Placement by Body, Shell, and Patch Level (v1.0)](../specifications/surface-89-metadata-placement-by-level-v1_0.md)
- [x] [Surface Spec 90: Metadata Inheritance and Override Rules (v1.0)](../specifications/surface-90-metadata-inheritance-override-rules-v1_0.md)
- [x] [Surface Spec 91: Kernel-Native Versus Consumer Metadata Boundary (v1.0)](../specifications/surface-91-kernel-vs-consumer-metadata-boundary-v1_0.md)
- [x] [Surface Spec 92: Stable Surface Identity Contract (v1.0)](../specifications/surface-92-stable-surface-identity-contract-v1_0.md)
- [x] [Surface Spec 93: Identity Preservation Through Transform and Composition (v1.0)](../specifications/surface-93-identity-preservation-transform-composition-v1_0.md)
- [x] [Surface Spec 94: Cache-Key Dependency and Identity Usage Rules (v1.0)](../specifications/surface-94-cache-key-identity-usage-rules-v1_0.md)

## Obligate Specifications

### Tessellation Boundary

- [x] [Surface Spec 32: Tessellation Request Object and Field Contract (v1.0)](../specifications/surface-32-tessellation-request-object-contract-v1_0.md)
- [x] [Surface Spec 33: Quality Presets and Explicit Tolerance Normalization (v1.0)](../specifications/surface-33-quality-preset-tolerance-normalization-v1_0.md)
- [x] [Surface Spec 34: Canonical Executor Input and Request Normalization (v1.0)](../specifications/surface-34-executor-input-request-canonicalization-v1_0.md)
- [x] [Surface Spec 35: Preview Tessellation Policy Contract (v1.0)](../specifications/surface-35-preview-tessellation-policy-v1_0.md)
- [x] [Surface Spec 36: Export and Analysis Tessellation Policy Contract (v1.0)](../specifications/surface-36-export-analysis-tessellation-policy-v1_0.md)
- [x] [Surface Spec 37: Cross-Mode Equivalence and Drift Bounds (v1.0)](../specifications/surface-37-cross-mode-equivalence-drift-bounds-v1_0.md)
- [x] [Surface Spec 38: Shared-Boundary Sampling and Edge Agreement Rules (v1.0)](../specifications/surface-38-shared-boundary-sampling-edge-agreement-v1_0.md)
- [x] [Surface Spec 38 Test: Shared-Boundary Sampling and Edge Agreement](../test-specifications/surface-38-shared-boundary-sampling-edge-agreement-v1_0.md)
- [x] [Surface Spec 39: Closed-Body Watertight Tessellation Contract (v1.0)](../specifications/surface-39-closed-body-watertight-tessellation-v1_0.md)
- [x] [Surface Spec 39 Test: Closed-Body Watertight Tessellation](../test-specifications/surface-39-closed-body-watertight-tessellation-v1_0.md)
- [x] [Surface Spec 40: Open-Surface Classification and Mesh QA Contract (v1.0)](../specifications/surface-40-open-surface-classification-mesh-qa-v1_0.md)

### Scene and Modeling Adoption

- [x] [Surface Spec 41: Scene Node Surface Payload Contract (v1.0)](../specifications/surface-41-scene-node-surface-payload-v1_0.md)
- [x] [Surface Spec 42: Group Traversal, Ordering, and Composition Rules (v1.0)](../specifications/surface-42-group-traversal-ordering-composition-v1_0.md)
- [x] [Surface Spec 43: Scene-to-Tessellation Consumer Handoff Contract (v1.0)](../specifications/surface-43-scene-to-tessellation-handoff-v1_0.md)
- [x] [Surface Spec 44: Primitive API Surface Return-Type Migration (v1.0)](../specifications/surface-44-primitive-api-surface-return-migration-v1_0.md)
- [x] [Surface Spec 44 Test: Primitive API Surface Return Migration](../test-specifications/surface-44-primitive-api-surface-return-migration-v1_0.md)
- [x] [Surface Spec 45: Modeling Operation Surface Return-Type Migration (v1.0)](../specifications/surface-45-modeling-op-surface-return-migration-v1_0.md)
- [x] [Surface Spec 45 Test: Modeling Operation Surface Return Migration](../test-specifications/surface-45-modeling-op-surface-return-migration-v1_0.md)
- [x] [Surface Spec 46: Public/Internal API Transition and Documentation Boundary (v1.0)](../specifications/surface-46-public-internal-api-transition-boundary-v1_0.md)
- [x] [Surface Spec 46 Test: Public/Internal API Transition Boundary](../test-specifications/surface-46-public-internal-api-transition-boundary-v1_0.md)
- [x] [Surface Spec 47: Surface Collection Consumer Interface (v1.0)](../specifications/surface-47-surface-collection-consumer-interface-v1_0.md)
- [x] [Surface Spec 48: Composition Flattening and Traversal Rules (v1.0)](../specifications/surface-48-composition-flattening-traversal-v1_0.md)
- [x] [Surface Spec 49: Composition-to-Tessellation Invocation Boundary (v1.0)](../specifications/surface-49-composition-to-tessellation-boundary-v1_0.md)

### Migration and Compatibility

- [x] [Surface Spec 50: Surface-to-Mesh Adapter Contract (v1.0)](../specifications/surface-50-surface-to-mesh-adapter-contract-v1_0.md)
- [x] [Surface Spec 51: Legacy Mesh Consumer Bridge Policy (v1.0)](../specifications/surface-51-legacy-mesh-consumer-bridge-policy-v1_0.md)
- [x] [Surface Spec 52: Adapter Lossiness and Lifecycle Rules (v1.0)](../specifications/surface-52-adapter-lossiness-lifecycle-rules-v1_0.md)
- [x] [Surface Spec 53: Surface Migration Phase Ordering (v1.0)](../specifications/surface-53-surface-migration-phase-ordering-v1_0.md)
- [x] [Surface Spec 53 Test: Surface Migration Phase Ordering](../test-specifications/surface-53-surface-migration-phase-ordering-v1_0.md)
- [x] [Surface Spec 54: Migration Phase Gates and Dependency Rules (v1.0)](../specifications/surface-54-migration-phase-gates-dependency-rules-v1_0.md)
- [x] [Surface Spec 54 Test: Migration Phase Gates and Dependency Rules](../test-specifications/surface-54-migration-phase-gates-dependency-rules-v1_0.md)
- [x] [Surface Spec 55: Surface-Foundation to Loft-Track Handoff Gate (v1.0)](../specifications/surface-55-surface-foundation-to-loft-handoff-v1_0.md)
- [x] [Surface Spec 55 Test: Surface-Foundation to Loft-Track Handoff Gate](../test-specifications/surface-55-surface-foundation-to-loft-handoff-v1_0.md)
- [x] [Surface Spec 56: Surface Canonical Promotion Criteria (v1.0)](../specifications/surface-56-surface-canonical-promotion-criteria-v1_0.md)
- [x] [Surface Spec 56 Test: Surface Canonical Promotion Criteria](../test-specifications/surface-56-surface-canonical-promotion-criteria-v1_0.md)
- [x] [Surface Spec 57: Mesh-First Decommission and Rollback Policy (v1.0)](../specifications/surface-57-mesh-first-decommission-rollback-v1_0.md)
- [x] [Surface Spec 57 Test: Mesh-First Decommission and Rollback Policy](../test-specifications/surface-57-mesh-first-decommission-rollback-v1_0.md)
- [x] [Surface Spec 58: Promotion Verification Matrix and Evidence Burden (v1.0)](../specifications/surface-58-promotion-verification-matrix-v1_0.md)
- [x] [Surface Spec 58 Test: Promotion Verification Matrix and Evidence Burden](../test-specifications/surface-58-promotion-verification-matrix-v1_0.md)

### Loft Surface Refactor Track

- [x] [Surface Spec 95: Loft Plan-to-Surface Executor Contract (v1.0)](../specifications/surface-95-loft-plan-to-surface-executor-contract-v1_0.md)
- [x] [Surface Spec 95 Test: Loft Plan-to-Surface Executor Contract](../test-specifications/surface-95-loft-plan-to-surface-executor-contract-v1_0.md)
- [x] [Surface Spec 96: Loft Surface-Native Cap Construction (v1.0)](../specifications/surface-96-loft-surface-native-cap-construction-v1_0.md)
- [x] [Surface Spec 96 Test: Loft Surface-Native Cap Construction](../test-specifications/surface-96-loft-surface-native-cap-construction-v1_0.md)
- [x] [Surface Spec 97: Loft Split/Merge Surface Patch Orchestration (v1.0)](../specifications/surface-97-loft-surface-patch-orchestration-v1_0.md)
- [x] [Surface Spec 97 Test: Loft Split/Merge Surface Patch Orchestration](../test-specifications/surface-97-loft-surface-patch-orchestration-v1_0.md)
- [x] [Surface Spec 98: Loft Surface Output Consumer Handoff (v1.0)](../specifications/surface-98-loft-surface-output-consumer-handoff-v1_0.md)
- [x] [Surface Spec 98 Test: Loft Surface Output Consumer Handoff](../test-specifications/surface-98-loft-surface-output-consumer-handoff-v1_0.md)

### Surface-First Replacement Program

- [x] [Surface Spec 100: Surface-Native Drafting Replacement (v1.0)](../specifications/surface-100-surface-native-drafting-replacement-v1_0.md)
- [x] [Surface Spec 100 Test: Surface-Native Drafting Replacement](../test-specifications/surface-100-surface-native-drafting-replacement-v1_0.md)
- [x] [Surface Spec 101: Surface-Native Text Replacement (v1.0)](../specifications/surface-101-surface-native-text-replacement-v1_0.md)
- [x] [Surface Spec 101 Test: Surface-Native Text Replacement](../test-specifications/surface-101-surface-native-text-replacement-v1_0.md)
- [x] [Surface Spec 108: Surface Boolean Input Eligibility and Canonicalization (v1.0)](../specifications/surface-108-surface-boolean-input-eligibility-and-canonicalization-v1_0.md)
- [x] [Surface Spec 108 Test: Surface Boolean Input Eligibility and Canonicalization](../test-specifications/surface-108-surface-boolean-input-eligibility-and-canonicalization-v1_0.md)
- [x] [Surface Spec 109: Surface Boolean Result Contract and Failure Modes (v1.0)](../specifications/surface-109-surface-boolean-result-contract-and-failure-modes-v1_0.md)
- [x] [Surface Spec 109 Test: Surface Boolean Result Contract and Failure Modes](../test-specifications/surface-109-surface-boolean-result-contract-and-failure-modes-v1_0.md)
- [x] [Surface Spec 110: Surface Boolean Public API Migration and Reference Verification (v1.0)](../specifications/surface-110-surface-boolean-public-api-migration-and-reference-verification-v1_0.md)
- [x] [Surface Spec 110 Test: Surface Boolean Public API Migration and Reference Verification](../test-specifications/surface-110-surface-boolean-public-api-migration-and-reference-verification-v1_0.md)
- [x] [Surface Spec 126: Surface Boolean Exact Body-Relation Classification and No-Cut Gate (v1.0)](../specifications/surface-126-surface-boolean-exact-body-relation-classification-and-no-cut-gate-v1_0.md)
- [x] [Surface Spec 126 Test: Surface Boolean Exact Body-Relation Classification and No-Cut Gate](../test-specifications/surface-126-surface-boolean-exact-body-relation-classification-and-no-cut-gate-v1_0.md)
- [x] [Surface Spec 127: Surface Boolean Initial Box/Box Cut-Curve Discovery (v1.0)](../specifications/surface-127-surface-boolean-initial-box-box-cut-curve-discovery-v1_0.md)
- [x] [Surface Spec 127 Test: Surface Boolean Initial Box/Box Cut-Curve Discovery](../test-specifications/surface-127-surface-boolean-initial-box-box-cut-curve-discovery-v1_0.md)
- [x] [Surface Spec 128: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice (v1.0)](../specifications/surface-128-surface-boolean-patch-local-trim-fragment-mapping-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 128 Test: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice](../test-specifications/surface-128-surface-boolean-patch-local-trim-fragment-mapping-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 129: Surface Boolean Initial Box Slice Fragment Classification and Split Records (v1.0)](../specifications/surface-129-surface-boolean-initial-box-slice-fragment-classification-and-split-records-v1_0.md)
- [x] [Surface Spec 129 Test: Surface Boolean Initial Box Slice Fragment Classification and Split Records](../test-specifications/surface-129-surface-boolean-initial-box-slice-fragment-classification-and-split-records-v1_0.md)
- [x] [Surface Spec 130: Surface Boolean No-Cut and Exact-Reuse Result Reconstruction (v1.0)](../specifications/surface-130-surface-boolean-no-cut-and-exact-reuse-result-reconstruction-v1_0.md)
- [x] [Surface Spec 130 Test: Surface Boolean No-Cut and Exact-Reuse Result Reconstruction](../test-specifications/surface-130-surface-boolean-no-cut-and-exact-reuse-result-reconstruction-v1_0.md)
- [x] [Surface Spec 131: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice (v1.0)](../specifications/surface-131-surface-boolean-overlap-cut-boundary-trim-loop-reconstruction-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 131 Test: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice](../test-specifications/surface-131-surface-boolean-overlap-cut-boundary-trim-loop-reconstruction-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 132: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice (v1.0)](../specifications/surface-132-surface-boolean-overlap-shell-and-seam-assembly-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 132 Test: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice](../test-specifications/surface-132-surface-boolean-overlap-shell-and-seam-assembly-for-initial-box-slice-v1_0.md)
- [x] [Surface Spec 133: Surface Boolean Result Outcome Classification for the Initial Slice (v1.0)](../specifications/surface-133-surface-boolean-result-outcome-classification-for-the-initial-slice-v1_0.md)
- [x] [Surface Spec 133 Test: Surface Boolean Result Outcome Classification for the Initial Slice](../test-specifications/surface-133-surface-boolean-result-outcome-classification-for-the-initial-slice-v1_0.md)
- [x] [Surface Spec 134: Surface Boolean Deterministic Validity Gate and Bounded Cleanup (v1.0)](../specifications/surface-134-surface-boolean-deterministic-validity-gate-and-bounded-cleanup-v1_0.md)
- [x] [Surface Spec 134 Test: Surface Boolean Deterministic Validity Gate and Bounded Cleanup](../test-specifications/surface-134-surface-boolean-deterministic-validity-gate-and-bounded-cleanup-v1_0.md)
- [x] [Surface Spec 135: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture (v1.0)](../specifications/surface-135-surface-boolean-metadata-provenance-and-explicit-invalid-result-posture-v1_0.md)
- [x] [Surface Spec 135 Test: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture](../test-specifications/surface-135-surface-boolean-metadata-provenance-and-explicit-invalid-result-posture-v1_0.md)
- [x] [Surface Spec 136: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix (v1.0)](../specifications/surface-136-surface-boolean-initial-executable-scope-and-unsupported-case-matrix-v1_0.md)
- [x] [Surface Spec 136 Test: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix](../test-specifications/surface-136-surface-boolean-initial-executable-scope-and-unsupported-case-matrix-v1_0.md)
- [x] [Surface Spec 138: Surface Boolean Reference Fixture Composition and Slice Verification (v1.0)](../specifications/surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md)
- [x] [Surface Spec 138 Test: Surface Boolean Reference Fixture Composition and Slice Verification](../test-specifications/surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md)
- [x] [Surface Spec 137: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates (v1.0)](../specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)
- [x] [Surface Spec 137 Test: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates](../test-specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)
- [x] [Surface Spec 114: Traditional Hinge Surface Assembly (v1.0)](../specifications/surface-114-traditional-hinge-surface-assembly-v1_0.md)
- [x] [Surface Spec 114 Test: Traditional Hinge Surface Assembly](../test-specifications/surface-114-traditional-hinge-surface-assembly-v1_0.md)
- [x] [Surface Spec 115: Living and Bistable Hinge Surface Assembly (v1.0)](../specifications/surface-115-living-and-bistable-hinge-surface-assembly-v1_0.md)
- [x] [Surface Spec 115 Test: Living and Bistable Hinge Surface Assembly](../test-specifications/surface-115-living-and-bistable-hinge-surface-assembly-v1_0.md)
- [x] [Surface Spec 116: Hinge Public Handoff and Showcase Verification (v1.0)](../specifications/surface-116-hinge-public-handoff-and-showcase-verification-v1_0.md)
- [x] [Surface Spec 116 Test: Hinge Public Handoff and Showcase Verification](../test-specifications/surface-116-hinge-public-handoff-and-showcase-verification-v1_0.md)
- [x] [Surface Spec 105: Surface-Native Heightfield and Displacement Replacement (v1.0)](../specifications/surface-105-surface-native-heightfield-displacement-v1_0.md)
- [x] [Surface Spec 105 Test: Surface-Native Heightfield and Displacement Replacement](../test-specifications/surface-105-surface-native-heightfield-displacement-v1_0.md)
- [x] [Surface Spec 122: Mesh Capability Retention and Deletion Matrix (v1.0)](../specifications/surface-122-mesh-capability-retention-and-deletion-matrix-v1_0.md)
- [x] [Surface Spec 122 Test: Mesh Capability Retention and Deletion Matrix](../test-specifications/surface-122-mesh-capability-retention-and-deletion-matrix-v1_0.md)
- [x] [Surface Spec 123: Mesh Analysis Toolchain Contract (v1.0)](../specifications/surface-123-mesh-analysis-toolchain-contract-v1_0.md)
- [x] [Surface Spec 123 Test: Mesh Analysis Toolchain Contract](../test-specifications/surface-123-mesh-analysis-toolchain-contract-v1_0.md)
- [x] [Surface Spec 124: Mesh Repair Toolchain Contract (v1.0)](../specifications/surface-124-mesh-repair-toolchain-contract-v1_0.md)
- [x] [Surface Spec 124 Test: Mesh Repair Toolchain Contract](../test-specifications/surface-124-mesh-repair-toolchain-contract-v1_0.md)
- [x] [Surface Spec 125: Standalone Mesh Utility Tool Contract (v1.0)](../specifications/surface-125-standalone-mesh-utility-tool-contract-v1_0.md)
- [x] [Surface Spec 125 Test: Standalone Mesh Utility Tool Contract](../test-specifications/surface-125-standalone-mesh-utility-tool-contract-v1_0.md)

### Reference Artifacts and Documentation

- [x] [Surface Spec 106: Reference Artifact Regression Suite (v1.0)](../specifications/surface-106-reference-artifact-regression-suite-v1_0.md)
- [x] [Surface Spec 106 Test: Reference Artifact Regression Suite](../test-specifications/surface-106-reference-artifact-regression-suite-v1_0.md)
- [x] [Surface Spec 107: Documentation-First Delivery Requirements (v1.0)](../specifications/surface-107-documentation-first-delivery-requirements-v1_0.md)
- [x] [Surface Spec 107 Test: Documentation-First Delivery Requirements](../test-specifications/surface-107-documentation-first-delivery-requirements-v1_0.md)

### Testing Architecture and Tooling

#### Reference Artifact Tooling

- [x] [Testing Spec 10: Reference Artifact Baseline Lifecycle and Invalidation Contract (v1.0)](../specifications/testing-10-reference-artifact-baseline-lifecycle-and-invalidation-contract-v1_0.md)
- [x] [Testing Spec 10 Test: Reference Artifact Baseline Lifecycle and Invalidation Contract](../test-specifications/testing-10-reference-artifact-baseline-lifecycle-and-invalidation-contract-v1_0.md)
- [x] [Testing Spec 11: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules (v1.0)](../specifications/testing-11-reference-artifact-grouped-model-output-completeness-and-bootstrap-rules-v1_0.md)
- [x] [Testing Spec 11 Test: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules](../test-specifications/testing-11-reference-artifact-grouped-model-output-completeness-and-bootstrap-rules-v1_0.md)

#### Shared CV Contracts

- [x] [Testing Spec 12: Computer Vision Shared Fixture Contract and Result Taxonomy (v1.0)](../specifications/testing-12-computer-vision-shared-fixture-contract-and-result-taxonomy-v1_0.md)
- [x] [Testing Spec 12 Test: Computer Vision Shared Fixture Contract and Result Taxonomy](../test-specifications/testing-12-computer-vision-shared-fixture-contract-and-result-taxonomy-v1_0.md)
- [x] [Testing Spec 13: Computer Vision Shared Harness Pipeline and Artifact Bundle Integration (v1.0)](../specifications/testing-13-computer-vision-shared-harness-pipeline-and-artifact-bundle-integration-v1_0.md)
- [x] [Testing Spec 13 Test: Computer Vision Shared Harness Pipeline and Artifact Bundle Integration](../test-specifications/testing-13-computer-vision-shared-harness-pipeline-and-artifact-bundle-integration-v1_0.md)

#### Text CV Lane

- [x] [Testing Spec 14: Computer Vision Text Canonical Artifact Set and Initial OCR Scope (v1.0)](../specifications/testing-14-computer-vision-text-canonical-artifact-set-and-initial-ocr-scope-v1_0.md)
- [x] [Testing Spec 14 Test: Computer Vision Text Canonical Artifact Set and Initial OCR Scope](../test-specifications/testing-14-computer-vision-text-canonical-artifact-set-and-initial-ocr-scope-v1_0.md)
- [x] [Testing Spec 15: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy (v1.0)](../specifications/testing-15-computer-vision-text-classification-confidence-and-fallback-glyph-policy-v1_0.md)
- [x] [Testing Spec 15 Test: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy](../test-specifications/testing-15-computer-vision-text-classification-confidence-and-fallback-glyph-policy-v1_0.md)

#### Slice CV Lane

- [x] [Testing Spec 16: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract (v1.0)](../specifications/testing-16-computer-vision-slice-artifact-frame-extraction-and-normalization-contract-v1_0.md)
- [x] [Testing Spec 16 Test: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract](../test-specifications/testing-16-computer-vision-slice-artifact-frame-extraction-and-normalization-contract-v1_0.md)
- [x] [Testing Spec 17: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy (v1.0)](../specifications/testing-17-computer-vision-slice-silhouette-comparison-method-and-orientation-class-taxonomy-v1_0.md)
- [x] [Testing Spec 17 Test: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy](../test-specifications/testing-17-computer-vision-slice-silhouette-comparison-method-and-orientation-class-taxonomy-v1_0.md)

#### Camera Contract

- [x] [Testing Spec 05: Computer Vision Camera and Framing Contract Compliance (v1.0)](../specifications/testing-05-computer-vision-camera-and-framing-contract-compliance-v1_0.md)
- [x] [Testing Spec 05 Test: Computer Vision Camera and Framing Contract Compliance](../test-specifications/testing-05-computer-vision-camera-and-framing-contract-compliance-v1_0.md)

#### Object-View CV Lane

- [x] [Testing Spec 18: Computer Vision Canonical Object-View Set and Authoritative Derived Products (v1.0)](../specifications/testing-18-computer-vision-canonical-object-view-set-and-authoritative-derived-products-v1_0.md)
- [x] [Testing Spec 18 Test: Computer Vision Canonical Object-View Set and Authoritative Derived Products](../test-specifications/testing-18-computer-vision-canonical-object-view-set-and-authoritative-derived-products-v1_0.md)
- [x] [Testing Spec 19: Computer Vision Object-View Interpretation Semantics and Product Comparison Posture (v1.0)](../specifications/testing-19-computer-vision-object-view-interpretation-semantics-and-product-comparison-posture-v1_0.md)
- [x] [Testing Spec 19 Test: Computer Vision Object-View Interpretation Semantics and Product Comparison Posture](../test-specifications/testing-19-computer-vision-object-view-interpretation-semantics-and-product-comparison-posture-v1_0.md)

#### Handedness CV Lane

- [x] [Testing Spec 20: Computer Vision Cross-Space Anchoring Contract for Handedness Verification (v1.0)](../specifications/testing-20-computer-vision-cross-space-anchoring-contract-for-handedness-verification-v1_0.md)
- [x] [Testing Spec 20 Test: Computer Vision Cross-Space Anchoring Contract for Handedness Verification](../test-specifications/testing-20-computer-vision-cross-space-anchoring-contract-for-handedness-verification-v1_0.md)
- [x] [Testing Spec 21: Computer Vision Handedness Witness Adequacy and Classification Taxonomy (v1.0)](../specifications/testing-21-computer-vision-handedness-witness-adequacy-and-classification-taxonomy-v1_0.md)
- [x] [Testing Spec 21 Test: Computer Vision Handedness Witness Adequacy and Classification Taxonomy](../test-specifications/testing-21-computer-vision-handedness-witness-adequacy-and-classification-taxonomy-v1_0.md)

#### Diagnostic Panels

- [x] [Testing Spec 22: Computer Vision Diagnostic Panel Layout, Ordering, and Region Extraction Contract (v1.0)](../specifications/testing-22-computer-vision-diagnostic-panel-layout-ordering-and-region-extraction-contract-v1_0.md)
- [x] [Testing Spec 22 Test: Computer Vision Diagnostic Panel Layout, Ordering, and Region Extraction Contract](../test-specifications/testing-22-computer-vision-diagnostic-panel-layout-ordering-and-region-extraction-contract-v1_0.md)
- [x] [Testing Spec 23: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules (v1.0)](../specifications/testing-23-computer-vision-diagnostic-panel-honesty-boundary-and-proof-delegation-rules-v1_0.md)
- [x] [Testing Spec 23 Test: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules](../test-specifications/testing-23-computer-vision-diagnostic-panel-honesty-boundary-and-proof-delegation-rules-v1_0.md)

## Loft Evolution

### Placed Topology Input

- [x] [Loft Spec 28: Placed Topology State Object Shape (v1.0)](../specifications/loft-28-placed-topology-state-object-shape-v1_0.md)
- [x] [Loft Spec 28 Test: Placed Topology State Object Shape](../test-specifications/loft-28-placed-topology-state-object-shape-v1_0.md)
- [x] [Loft Spec 29: Topology State Normalization Invariants (v1.0)](../specifications/loft-29-topology-state-normalization-invariants-v1_0.md)
- [x] [Loft Spec 29 Test: Topology State Normalization Invariants](../test-specifications/loft-29-topology-state-normalization-invariants-v1_0.md)
- [x] [Loft Spec 30: Directional Correspondence Field Contract (v1.0)](../specifications/loft-30-directional-correspondence-field-contract-v1_0.md)
- [x] [Loft Spec 30 Test: Directional Correspondence Field Contract](../test-specifications/loft-30-directional-correspondence-field-contract-v1_0.md)

### Plan Object

- [x] [Loft Spec 31: Plan Header and Sequence Metadata Contract (v1.0)](../specifications/loft-31-plan-header-and-sequence-metadata-contract-v1_0.md)
- [x] [Loft Spec 31 Test: Plan Header and Sequence Metadata Contract](../test-specifications/loft-31-plan-header-and-sequence-metadata-contract-v1_0.md)
- [x] [Loft Spec 32: Planned State and Interval Record Contract (v1.0)](../specifications/loft-32-planned-state-and-interval-record-contract-v1_0.md)
- [x] [Loft Spec 32 Test: Planned State and Interval Record Contract](../test-specifications/loft-32-planned-state-and-interval-record-contract-v1_0.md)
- [x] [Loft Spec 33: Branch and Closure Record Contract (v1.0)](../specifications/loft-33-branch-and-closure-record-contract-v1_0.md)
- [x] [Loft Spec 33 Test: Branch and Closure Record Contract](../test-specifications/loft-33-branch-and-closure-record-contract-v1_0.md)
- [x] [Loft Spec 34: Plan Diagnostics and Execution Eligibility Contract (v1.0)](../specifications/loft-34-plan-diagnostics-and-execution-eligibility-contract-v1_0.md)
- [x] [Loft Spec 34 Test: Plan Diagnostics and Execution Eligibility Contract](../test-specifications/loft-34-plan-diagnostics-and-execution-eligibility-contract-v1_0.md)

### Planner / Executor Boundary

- [x] [Loft Spec 35: Transition Operator Family Set (v1.0)](../specifications/loft-35-transition-operator-family-set-v1_0.md)
- [x] [Loft Spec 35 Test: Transition Operator Family Set](../test-specifications/loft-35-transition-operator-family-set-v1_0.md)
- [x] [Loft Spec 36: Transition Operator Payload Contract (v1.0)](../specifications/loft-36-transition-operator-payload-contract-v1_0.md)
- [x] [Loft Spec 36 Test: Transition Operator Payload Contract](../test-specifications/loft-36-transition-operator-payload-contract-v1_0.md)
- [x] [Loft Spec 37: Planner / Executor Execution-Boundary Rules (v1.0)](../specifications/loft-37-planner-executor-execution-boundary-rules-v1_0.md)
- [x] [Loft Spec 37 Test: Planner / Executor Execution-Boundary Rules](../test-specifications/loft-37-planner-executor-execution-boundary-rules-v1_0.md)

### Ambiguity and Constraint Requests

- [x] [Loft Spec 38: Ambiguity Record Minimal Locator Contract (v1.0)](../specifications/loft-38-ambiguity-record-minimal-locator-contract-v1_0.md)
- [x] [Loft Spec 38 Test: Ambiguity Record Minimal Locator Contract](../test-specifications/loft-38-ambiguity-record-minimal-locator-contract-v1_0.md)
- [x] [Loft Spec 39: Constraint Request Record Contract (v1.0)](../specifications/loft-39-constraint-request-record-contract-v1_0.md)
- [x] [Loft Spec 39 Test: Constraint Request Record Contract](../test-specifications/loft-39-constraint-request-record-contract-v1_0.md)
- [x] [Loft Spec 40: Invalid-Input Versus Underconstrained Taxonomy (v1.0)](../specifications/loft-40-invalid-input-versus-underconstrained-taxonomy-v1_0.md)
- [x] [Loft Spec 40 Test: Invalid-Input Versus Underconstrained Taxonomy](../test-specifications/loft-40-invalid-input-versus-underconstrained-taxonomy-v1_0.md)

### Tolerance and Validation

- [x] [Loft Spec 41: Input-Validity Tolerance Rules (v1.0)](../specifications/loft-41-input-validity-tolerance-rules-v1_0.md)
- [x] [Loft Spec 41 Test: Input-Validity Tolerance Rules](../test-specifications/loft-41-input-validity-tolerance-rules-v1_0.md)
- [x] [Loft Spec 42: Structural-Classification Tolerance Rules (v1.0)](../specifications/loft-42-structural-classification-tolerance-rules-v1_0.md)
- [x] [Loft Spec 42 Test: Structural-Classification Tolerance Rules](../test-specifications/loft-42-structural-classification-tolerance-rules-v1_0.md)
- [x] [Loft Spec 43: Decomposition-Resolution Tolerance Rules (v1.0)](../specifications/loft-43-decomposition-resolution-tolerance-rules-v1_0.md)
- [x] [Loft Spec 43 Test: Decomposition-Resolution Tolerance Rules](../test-specifications/loft-43-decomposition-resolution-tolerance-rules-v1_0.md)
- [x] [Loft Spec 44: Collapse and Degeneracy Tolerance Rules (v1.0)](../specifications/loft-44-collapse-and-degeneracy-tolerance-rules-v1_0.md)
- [x] [Loft Spec 44 Test: Collapse and Degeneracy Tolerance Rules](../test-specifications/loft-44-collapse-and-degeneracy-tolerance-rules-v1_0.md)
- [x] [Loft Spec 45: Plan-Validation Tolerance Rules (v1.0)](../specifications/loft-45-plan-validation-tolerance-rules-v1_0.md)
- [x] [Loft Spec 45 Test: Plan-Validation Tolerance Rules](../test-specifications/loft-45-plan-validation-tolerance-rules-v1_0.md)

### Many-to-Many Decomposition

- [x] [Loft Spec 46: Many-to-Many Candidate-Set Isolation Rules (v1.0)](../specifications/loft-46-many-to-many-candidate-set-isolation-rules-v1_0.md)
- [x] [Loft Spec 46 Test: Many-to-Many Candidate-Set Isolation Rules](../test-specifications/loft-46-many-to-many-candidate-set-isolation-rules-v1_0.md)
- [x] [Loft Spec 47: Many-to-Many Deterministic Decomposition Order (v1.0)](../specifications/loft-47-many-to-many-deterministic-decomposition-order-v1_0.md)
- [x] [Loft Spec 47 Test: Many-to-Many Deterministic Decomposition Order](../test-specifications/loft-47-many-to-many-deterministic-decomposition-order-v1_0.md)
- [x] [Loft Spec 48: Automatic Decomposability Gate Rules (v1.0)](../specifications/loft-48-automatic-decomposability-gate-rules-v1_0.md)
- [x] [Loft Spec 48 Test: Automatic Decomposability Gate Rules](../test-specifications/loft-48-automatic-decomposability-gate-rules-v1_0.md)
- [x] [Loft Spec 49: Residual Many-to-Many Constraint Escalation (v1.0)](../specifications/loft-49-residual-many-to-many-constraint-escalation-v1_0.md)
- [x] [Loft Spec 49 Test: Residual Many-to-Many Constraint Escalation](../test-specifications/loft-49-residual-many-to-many-constraint-escalation-v1_0.md)
- [x] [Loft Spec 50: Simple Correspondence Regression Fixtures (v1.0)](../specifications/loft-50-simple-correspondence-regression-fixtures-v1_0.md)
- [x] [Loft Spec 50 Test: Simple Correspondence Regression Fixtures](../test-specifications/loft-50-simple-correspondence-regression-fixtures-v1_0.md)
- [x] [Loft Spec 51: Canonical Station-Slice Silhouette Classification (v1.0)](../specifications/loft-51-canonical-station-slice-silhouette-classification-v1_0.md)
- [x] [Loft Spec 51 Test: Canonical Station-Slice Silhouette Classification](../test-specifications/loft-51-canonical-station-slice-silhouette-classification-v1_0.md)

### Topology-Owned Point Correspondence

- [x] [Loft Spec 52: Topology Path Core Records (v1.0)](../specifications/loft-52-topology-path-core-records-v1_0.md)
- [x] [Loft Spec 53: Topology Segment and Landmark Identity Records (v1.0)](../specifications/loft-53-topology-segment-landmark-identity-records-v1_0.md)
- [x] [Loft Spec 54: Path Input Adapters to Topology Paths (v1.0)](../specifications/loft-54-path-input-adapters-to-topology-paths-v1_0.md)
- [x] [Loft Spec 62: Topology Builder Core API (v1.0)](../specifications/loft-62-topology-builder-core-api-v1_0.md)
- [x] [Loft Spec 63: Topology Lifecycle Builder API (v1.0)](../specifications/loft-63-topology-lifecycle-builder-api-v1_0.md)
- [x] [Loft Spec 64: Generated Shape Default Rails (v1.0)](../specifications/loft-64-generated-shape-default-rails-v1_0.md)
- [x] [Loft Spec 55: Authored Rail Priority and Diagnostics (v1.0)](../specifications/loft-55-authored-rail-priority-diagnostics-v1_0.md)
- [x] [Loft Spec 56: High-Confidence Inference and Refusal Policy (v1.0)](../specifications/loft-56-high-confidence-inference-refusal-policy-v1_0.md)
- [x] [Loft Spec 57: Point Lifecycle Event Records (v1.0)](../specifications/loft-57-point-lifecycle-event-records-v1_0.md)
- [x] [Loft Spec 58: Birth/Death Synthetic Support Resolution (v1.0)](../specifications/loft-58-birth-death-synthetic-support-resolution-v1_0.md)
- [x] [Loft Spec 59: Correspondence-Preserving Resampling Contract (v1.0)](../specifications/loft-59-correspondence-preserving-resampling-contract-v1_0.md)
- [x] [Loft Spec 60: Legacy Mesh Debug Correspondence Consumption (retired canonical posture)](../specifications/loft-60-mesh-executor-correspondence-consumption-v1_0.md)
- [x] [Loft Spec 61: Surface Executor Correspondence Consumption (v1.0)](../specifications/loft-61-surface-executor-correspondence-consumption-v1_0.md)

## Polish Specifications

### Advanced Loft Features

- [x] [Loft Spec 18: Probabilistic Ambiguity Disambiguation (v1.0)](../specifications/loft-18-probabilistic-disambiguation-v1_0.md)
- [x] [Loft Spec 18 Test: Probabilistic Ambiguity Disambiguation](../test-specifications/loft-18-probabilistic-disambiguation-v1_0.md)
- [x] [Loft Spec 19: Global Fairness and Skeleton Optimization (v1.0)](../specifications/loft-19-global-fairness-skeleton-optimization-v1_0.md)
- [x] [Loft Spec 19 Test: Global Fairness and Skeleton Optimization](../test-specifications/loft-19-global-fairness-skeleton-optimization-v1_0.md)
- [x] [Loft Spec 20: Interactive Branch Picking API (v1.0)](../specifications/loft-20-interactive-branch-picking-v1_0.md)
- [x] [Loft Spec 20 Test: Interactive Branch Picking API](../test-specifications/loft-20-interactive-branch-picking-v1_0.md)

## Manifest-Derived Specification Work

### Impress Surface Native File Format Architecture
- [x] [Impress Spec 01: .impress Document Root And Schema Version (v1.0)](../specifications/impress-01-impress-document-root-and-schema-version-v1_0.md)
- [x] [Impress Spec 02: .impress Units And Root Validation Policy (v1.0)](../specifications/impress-02-impress-units-and-root-validation-policy-v1_0.md)
- [x] [Impress Spec 03: .impress SurfaceBodyStore And Identity Policy (v1.0)](../specifications/impress-03-impress-surfacebodystore-and-identity-policy-v1_0.md)
- [x] [Impress Spec 04: .impress Body And Shell Payload Codec (v1.0)](../specifications/impress-04-impress-body-and-shell-payload-codec-v1_0.md)
- [x] [Impress Spec 05: .impress Patch Payload Codec (v1.0)](../specifications/impress-05-impress-patch-payload-codec-v1_0.md)
- [x] [Impress Spec 06: .impress Trim Payload Codec (v1.0)](../specifications/impress-06-impress-trim-payload-codec-v1_0.md)
- [x] [Impress Spec 07: .impress Seam Payload Codec (v1.0)](../specifications/impress-07-impress-seam-payload-codec-v1_0.md)
- [x] [Impress Spec 08: .impress Adjacency Payload Codec (v1.0)](../specifications/impress-08-impress-adjacency-payload-codec-v1_0.md)
- [x] [Impress Spec 09: .impress Deterministic JSON Writer (v1.0)](../specifications/impress-09-impress-deterministic-json-writer-v1_0.md)
- [x] [Impress Spec 10: .impress Reader And Load Result Contract (v1.0)](../specifications/impress-10-impress-reader-and-load-result-contract-v1_0.md)
- [x] [Impress Spec 11: .impress Atomic File Write And Error Handling (v1.0)](../specifications/impress-11-impress-atomic-file-write-and-error-handling-v1_0.md)
- [x] [Impress Spec 12: .impress Round Trip And Refusal Tests (v1.0)](../specifications/impress-12-impress-round-trip-and-refusal-tests-v1_0.md)
- [x] [Impress Spec 13: .impress Industry Interchange Boundary (v1.0)](../specifications/impress-13-impress-industry-interchange-boundary-v1_0.md)

### Full Surface Patch Family Architecture
- [x] [Surface Spec 139: Patch Family Capability Matrix And Spec 66 Retirement (v1.0)](../specifications/surface-139-patch-family-capability-matrix-and-spec-66-retirement-v1_0.md)
- [x] [Surface Spec 140: B-Spline Surface Patch Record And Validation (v1.0)](../specifications/surface-140-b-spline-surface-patch-record-and-validation-v1_0.md)
- [x] [Surface Spec 141: B-Spline Surface Evaluation And Derivatives (v1.0)](../specifications/surface-141-b-spline-surface-evaluation-and-derivatives-v1_0.md)
- [x] [Surface Spec 142: NURBS Surface Patch (v1.0)](../specifications/surface-142-nurbs-surface-patch-v1_0.md)
- [x] [Surface Spec 143: Sweep Surface Patch Payload And Frame Policy (v1.0)](../specifications/surface-143-sweep-surface-patch-payload-and-frame-policy-v1_0.md)
- [x] [Surface Spec 144: Sweep Surface Evaluation And Tessellation (v1.0)](../specifications/surface-144-sweep-surface-evaluation-and-tessellation-v1_0.md)
- [x] [Surface Spec 145: Subdivision Surface Control Cage And Crease Payload (v1.0)](../specifications/surface-145-subdivision-surface-control-cage-and-crease-payload-v1_0.md)
- [x] [Surface Spec 146: Subdivision Surface Evaluation And Tessellation (v1.0)](../specifications/surface-146-subdivision-surface-evaluation-and-tessellation-v1_0.md)
- [x] [Surface Spec 147: Implicit Field Node Payload (v1.0)](../specifications/surface-147-implicit-field-node-payload-v1_0.md)
- [x] [Surface Spec 148: Implicit Field Validation Security (v1.0)](../specifications/surface-148-implicit-field-validation-security-v1_0.md)
- [x] [Surface Spec 149: Implicit Field Evaluation (v1.0)](../specifications/surface-149-implicit-field-evaluation-v1_0.md)
- [x] [Surface Spec 150: Implicit Tessellation Bounds And Approximation Metadata (v1.0)](../specifications/surface-150-implicit-tessellation-bounds-and-approximation-metadata-v1_0.md)
- [x] [Surface Spec 151: Cross-Family Tessellation Adapters (v1.0)](../specifications/surface-151-cross-family-tessellation-adapters-v1_0.md)
- [x] [Surface Spec 152: Cross-Family Seam Boundary Participation (v1.0)](../specifications/surface-152-cross-family-seam-boundary-participation-v1_0.md)
- [x] [Surface Spec 153: Family-Aware Boolean Eligibility And Diagnostics (v1.0)](../specifications/surface-153-family-aware-boolean-eligibility-and-diagnostics-v1_0.md)
- [x] [Surface Spec 154: .impress Analytic Patch Payloads (v1.0)](../specifications/surface-154-impress-analytic-patch-payloads-v1_0.md)
- [x] [Surface Spec 155: .impress Spline Patch Payloads (v1.0)](../specifications/surface-155-impress-spline-patch-payloads-v1_0.md)
- [x] [Surface Spec 156: .impress Sweep Patch Payload (v1.0)](../specifications/surface-156-impress-sweep-patch-payload-v1_0.md)
- [x] [Surface Spec 157: .impress Subdivision Patch Payload (v1.0)](../specifications/surface-157-impress-subdivision-patch-payload-v1_0.md)
- [x] [Surface Spec 158: .impress Implicit Patch Payload Security (v1.0)](../specifications/surface-158-impress-implicit-patch-payload-security-v1_0.md)

### Mesh Execution Tessellation Boundary Architecture
- [x] [Surface Spec 159: Mesh Execution Inventory And Classification (v1.0)](../specifications/surface-159-mesh-execution-inventory-and-classification-v1_0.md)
- [x] [Surface Spec 160: Primitive Surface Defaults: Box, Prism, Polyhedron (v1.0)](../specifications/surface-160-primitive-surface-defaults-box-prism-polyhedron-v1_0.md)
- [x] [Surface Spec 161: Primitive Surface Defaults: Cylinder, Cone, Ngon (v1.0)](../specifications/surface-161-primitive-surface-defaults-cylinder-cone-ngon-v1_0.md)
- [x] [Surface Spec 162: Primitive Surface Defaults: Sphere And Torus (v1.0)](../specifications/surface-162-primitive-surface-defaults-sphere-and-torus-v1_0.md)
- [x] [Surface Spec 163: Primitive Mesh Compatibility API Names (v1.0)](../specifications/surface-163-primitive-mesh-compatibility-api-names-v1_0.md)
- [x] [Surface Spec 164: Primitive Planar And Frustum Helper Quarantine (v1.0)](../specifications/surface-164-primitive-planar-and-frustum-helper-quarantine-v1_0.md)
- [x] [Surface Spec 165: Primitive Curved Helper Quarantine (v1.0)](../specifications/surface-165-primitive-curved-helper-quarantine-v1_0.md)
- [x] [Surface Spec 166: Primitive Tessellation Helper Relocation (v1.0)](../specifications/surface-166-primitive-tessellation-helper-relocation-v1_0.md)
- [x] [Surface Spec 167: Loft Spec 60 Revision Or Retirement (v1.0)](../specifications/surface-167-loft-spec-60-revision-or-retirement-v1_0.md)
- [x] [Surface Spec 168: Loft Mesh Emission Relocation (v1.0)](../specifications/surface-168-loft-mesh-emission-relocation-v1_0.md)
- [x] [Surface Spec 169: Text Surface Default Public API (v1.0)](../specifications/surface-169-text-surface-default-public-api-v1_0.md)
- [x] [Surface Spec 170: Text Mesh Compatibility And Empty Text Behavior (v1.0)](../specifications/surface-170-text-mesh-compatibility-and-empty-text-behavior-v1_0.md)
- [x] [Surface Spec 171: Drafting Surface Defaults (v1.0)](../specifications/surface-171-drafting-surface-defaults-v1_0.md)
- [x] [Surface Spec 172: Heightmap Sampled Surface Payload (v1.0)](../specifications/surface-172-heightmap-sampled-surface-payload-v1_0.md)
- [x] [Surface Spec 173: Heightmap Alpha Mask And Cache Policy (v1.0)](../specifications/surface-173-heightmap-alpha-mask-and-cache-policy-v1_0.md)
- [x] [Surface Spec 174: Heightmap Displacement Surface Contract (v1.0)](../specifications/surface-174-heightmap-displacement-surface-contract-v1_0.md)
- [x] [Surface Spec 175: Heightmap Projection Sampling Policy (v1.0)](../specifications/surface-175-heightmap-projection-sampling-policy-v1_0.md)
- [x] [Surface Spec 176: Heightmap Mesh Compatibility And Serialization Guard (v1.0)](../specifications/surface-176-heightmap-mesh-compatibility-and-serialization-guard-v1_0.md)
- [x] [Surface Spec 180: Traditional Hinge Surface Lowering (v1.0)](../specifications/surface-180-traditional-hinge-surface-lowering-v1_0.md)
- [x] [Surface Spec 181: Living And Bistable Hinge Surface Lowering (v1.0)](../specifications/surface-181-living-and-bistable-hinge-surface-lowering-v1_0.md)
- [x] [Surface Spec 182: Surface Transform API Defaults (v1.0)](../specifications/surface-182-surface-transform-api-defaults-v1_0.md)
- [x] [Surface Spec 183: Surface Composition Public Type (v1.0)](../specifications/surface-183-surface-composition-public-type-v1_0.md)
- [x] [Surface Spec 184: Surface Composition Traversal And Tessellation Handoff (v1.0)](../specifications/surface-184-surface-composition-traversal-and-tessellation-handoff-v1_0.md)
- [x] [Surface Spec 185: MeshGroup Explicit Compatibility API (v1.0)](../specifications/surface-185-meshgroup-explicit-compatibility-api-v1_0.md)
- [x] [Surface Spec 186: MeshGroup Import Boundary Enforcement (v1.0)](../specifications/surface-186-meshgroup-import-boundary-enforcement-v1_0.md)
- [x] [Surface Spec 187: Mesh Utility Quarantine (v1.0)](../specifications/surface-187-mesh-utility-quarantine-v1_0.md)
- [x] [Surface Spec 188: No Hidden Mesh Fallback Enforcement (v1.0)](../specifications/surface-188-no-hidden-mesh-fallback-enforcement-v1_0.md)

### Patch Family Integration Architecture
- [x] [Surface Spec 189: Patch Family Availability Gate And Capability Matrix (v1.0)](../specifications/surface-189-patch-family-availability-gate-and-capability-matrix-v1_0.md)
- [x] [Surface Spec 190: Patch Family Availability Promotion Pass (v1.0)](../specifications/surface-190-patch-family-availability-promotion-pass-v1_0.md)
- [x] [Surface Spec 191: .impress Patch Codec Coverage Inventory (v1.0)](../specifications/surface-191-impress-patch-codec-coverage-inventory-v1_0.md)
- [x] [Surface Spec 192: .impress All-Family Round-Trip Identity Coverage (v1.0)](../specifications/surface-192-impress-all-family-round-trip-identity-coverage-v1_0.md)
- [x] [Surface Spec 193: .impress Heightmap Surface Patch Codec (v1.0)](../specifications/surface-193-impress-heightmap-surface-patch-codec-v1_0.md)
- [x] [Surface Spec 194: .impress Displacement Surface Patch Codec (v1.0)](../specifications/surface-194-impress-displacement-surface-patch-codec-v1_0.md)
- [x] [Surface Spec 195: SurfaceBody All-Family Store Fixture Coverage (v1.0)](../specifications/surface-195-surfacebody-all-family-store-fixture-coverage-v1_0.md)
- [x] [Surface Spec 196: SurfaceComposition All-Family Traversal Coverage (v1.0)](../specifications/surface-196-surfacecomposition-all-family-traversal-coverage-v1_0.md)
- [x] [Surface Spec 197: SurfaceComposition Transform And Identity Preservation Coverage (v1.0)](../specifications/surface-197-surfacecomposition-transform-and-identity-preservation-coverage-v1_0.md)
- [x] [Surface Spec 198: Tessellation Adapter Capability Inventory And Metadata Gate (v1.0)](../specifications/surface-198-tessellation-adapter-capability-inventory-and-metadata-gate-v1_0.md)
- [x] [Surface Spec 199: Sampled Surface Tessellation Adapters (v1.0)](../specifications/surface-199-sampled-surface-tessellation-adapters-v1_0.md)
- [x] [Surface Spec 200: Spline Surface Tessellation Adapters (v1.0)](../specifications/surface-200-spline-surface-tessellation-adapters-v1_0.md)
- [x] [Surface Spec 201: Subdivision Surface Tessellation Adapter (v1.0)](../specifications/surface-201-subdivision-surface-tessellation-adapter-v1_0.md)
- [x] [Surface Spec 202: Implicit Surface Tessellation Adapter Safety And Sampling (v1.0)](../specifications/surface-202-implicit-surface-tessellation-adapter-safety-and-sampling-v1_0.md)
- [x] [Surface Spec 203: Loft Patch Family Selection Record (v1.0)](../specifications/surface-203-loft-patch-family-selection-record-v1_0.md)
- [x] [Surface Spec 204: Shared Spline Basis And Loft Control-Net Utilities (v1.0)](../specifications/surface-204-shared-spline-basis-and-loft-control-net-utilities-v1_0.md)
- [x] [Surface Spec 205: Loft B-Spline Output Contract (v1.0)](../specifications/surface-205-loft-b-spline-output-contract-v1_0.md)
- [x] [Surface Spec 206: Loft B-Spline Control-Net Producer (v1.0)](../specifications/surface-206-loft-b-spline-control-net-producer-v1_0.md)
- [x] [Surface Spec 207: Loft NURBS Explicit Rational Intent Contract (v1.0)](../specifications/surface-207-loft-nurbs-explicit-rational-intent-contract-v1_0.md)
- [x] [Surface Spec 208: Loft NURBS Weight-Grid Producer (v1.0)](../specifications/surface-208-loft-nurbs-weight-grid-producer-v1_0.md)
- [x] [Surface Spec 209: Loft Sweep Surface Producer (v1.0)](../specifications/surface-209-loft-sweep-surface-producer-v1_0.md)
- [x] [Surface Spec 210: Primitive Family Appropriateness Audit (v1.0)](../specifications/surface-210-primitive-family-appropriateness-audit-v1_0.md)
- [x] [Surface Spec 211: Subdivision Surface Public Authoring Helper (v1.0)](../specifications/surface-211-subdivision-surface-public-authoring-helper-v1_0.md)
- [x] [Surface Spec 212: Implicit Surface Public Authoring Helper And Safety Gate (v1.0)](../specifications/surface-212-implicit-surface-public-authoring-helper-and-safety-gate-v1_0.md)
- [x] [Surface Spec 213: CSG Unsupported Family Diagnostics (v1.0)](../specifications/surface-213-csg-unsupported-family-diagnostics-v1_0.md)
- [x] [Surface Spec 214: Loft Unsupported Family Diagnostics (v1.0)](../specifications/surface-214-loft-unsupported-family-diagnostics-v1_0.md)
- [x] [Surface Spec 215: .impress Unsupported Family And Malformed Payload Diagnostics (v1.0)](../specifications/surface-215-impress-unsupported-family-and-malformed-payload-diagnostics-v1_0.md)
- [x] [Surface Spec 216: Tessellation Unsupported Family Diagnostics (v1.0)](../specifications/surface-216-tessellation-unsupported-family-diagnostics-v1_0.md)
- [x] [Surface Spec 217: Advanced Patch Family Public API Documentation (v1.0)](../specifications/surface-217-advanced-patch-family-public-api-documentation-v1_0.md)

### Surface CSG And Mesh Boundary Completion
- [x] [Surface Spec 224: Loft Public Surface API Default (v1.0)](../specifications/surface-224-loft-public-surface-api-default-v1_0.md)
- [x] [Surface Spec 225: Loft Reference QA Role Boundary (v1.0)](../specifications/surface-225-loft-reference-qa-role-boundary-v1_0.md)
- [x] [Surface Spec 218: Surface CSG Family Capability Matrix And Refusal Gate (v1.0)](../specifications/surface-218-surface-csg-family-capability-matrix-and-refusal-gate-v1_0.md)
- [x] [Surface Spec 226: CSG Curve Primitive And Tolerance Policy (v1.0)](../specifications/surface-226-csg-curve-primitive-and-tolerance-policy-v1_0.md)
- [x] [Surface Spec 227: CSG Patch-Local Curve Mapping (v1.0)](../specifications/surface-227-csg-patch-local-curve-mapping-v1_0.md)
- [x] [Surface Spec 228: CSG Planar/Linear Analytic Intersections (v1.0)](../specifications/surface-228-csg-planar-linear-analytic-intersections-v1_0.md)
- [x] [Surface Spec 229: CSG Revolution/Conic Analytic Intersections (v1.0)](../specifications/surface-229-csg-revolution-conic-analytic-intersections-v1_0.md)
- [x] [Surface Spec 230: CSG Higher-Order Analytic Intersection Refusal Or Solver Boundary (v1.0)](../specifications/surface-230-csg-higher-order-analytic-intersection-refusal-or-solver-boundary-v1_0.md)
- [x] [Surface Spec 231: CSG Curve Arrangement And Trim Loop Splitting (v1.0)](../specifications/surface-231-csg-curve-arrangement-and-trim-loop-splitting-v1_0.md)
- [x] [Surface Spec 232: CSG Fragment Inside/Outside Classification Predicates (v1.0)](../specifications/surface-232-csg-fragment-inside-outside-classification-predicates-v1_0.md)
- [x] [Surface Spec 233: CSG Operation Selection Rules (v1.0)](../specifications/surface-233-csg-operation-selection-rules-v1_0.md)
- [x] [Surface Spec 234: CSG Shell Assembly From Fragments (v1.0)](../specifications/surface-234-csg-shell-assembly-from-fragments-v1_0.md)
- [x] [Surface Spec 235: CSG Seam And Adjacency Rebuild (v1.0)](../specifications/surface-235-csg-seam-and-adjacency-rebuild-v1_0.md)
- [x] [Surface Spec 236: CSG Validity, Healing, And Provenance Gate (v1.0)](../specifications/surface-236-csg-validity-healing-and-provenance-gate-v1_0.md)
- [x] [Surface Spec 237: CSG Primitive Caller Inventory And Gate Helper (v1.0)](../specifications/surface-237-csg-primitive-caller-inventory-and-gate-helper-v1_0.md)
- [x] [Surface Spec 238: CSG Primitive Migration (v1.0)](../specifications/surface-238-csg-primitive-migration-v1_0.md)
- [x] [Surface Spec 239: CSG Feature Builder Boolean Migration (v1.0)](../specifications/surface-239-csg-feature-builder-boolean-migration-v1_0.md)

### Additional Manifest Splits
- [x] [Impress Spec 14: .impress Base Patch Payload Codec (v1.0)](../specifications/impress-14-impress-base-patch-payload-codec-v1_0.md)
- [x] [Impress Spec 15: .impress Patch Family Dispatch And Validation (v1.0)](../specifications/impress-15-impress-patch-family-dispatch-and-validation-v1_0.md)
- [x] [Loft Spec 65: Topology Point Builder API (v1.0)](../specifications/loft-65-topology-point-builder-api-v1_0.md)
- [x] [Loft Spec 66: Topology Segment Builder API (v1.0)](../specifications/loft-66-topology-segment-builder-api-v1_0.md)
- [x] [Surface Spec 240: .impress Implicit Patch Payload Codec (v1.0)](../specifications/surface-240-impress-implicit-patch-payload-codec-v1_0.md)
- [x] [Surface Spec 241: .impress Implicit Payload Security Refusal Fixtures (v1.0)](../specifications/surface-241-impress-implicit-payload-security-refusal-fixtures-v1_0.md)

### Surface Body Completion Architecture
- [x] [Surface Spec 242: Surface Body Completion Capability And Evidence Gate (v1.0)](../specifications/surface-242-surface-body-completion-capability-and-evidence-gate-v1_0.md)
- [x] [Surface Spec 243: Patch Family Promotion Readiness Audit (v1.0)](../specifications/surface-243-patch-family-promotion-readiness-audit-v1_0.md)
- [x] [Surface Spec 244: CSG Completion Support Matrix And Refusal Records (v1.0)](../specifications/surface-244-csg-completion-support-matrix-and-refusal-records-v1_0.md)
- [x] [Surface Spec 245: Analytic CSG Expansion For Primitive Surface Families (v1.0)](../specifications/surface-245-analytic-csg-expansion-for-primitive-surface-families-v1_0.md)
- [x] [Surface Spec 246: Higher-Order And Sampled CSG Policy Boundary (v1.0)](../specifications/surface-246-higher-order-and-sampled-csg-policy-boundary-v1_0.md)
- [x] [Surface Spec 247: Loft Ambiguity Accumulation And Execution Refusal Gate (v1.0)](../specifications/surface-247-loft-ambiguity-accumulation-and-execution-refusal-gate-v1_0.md)
- [x] [Surface Spec 248: Loft Ambiguity Locator Diagnostics (v1.0)](../specifications/surface-248-loft-ambiguity-locator-diagnostics-v1_0.md)
- [x] [Surface Spec 249: Seam Continuity Promotion And Diagnostics (v1.0)](../specifications/surface-249-seam-continuity-promotion-and-diagnostics-v1_0.md)
- [x] [Surface Spec 250: .impress Whole-Store Fixture Coverage Gate (v1.0)](../specifications/surface-250-impress-whole-store-fixture-coverage-gate-v1_0.md)
- [x] [Surface Spec 251: .impress Unsafe Payload Refusal And Determinism Gate (v1.0)](../specifications/surface-251-impress-unsafe-payload-refusal-and-determinism-gate-v1_0.md)
- [x] [Surface Spec 252: Primitive Patch Producer Selection (v1.0)](../specifications/surface-252-primitive-patch-producer-selection-v1_0.md)
- [x] [Surface Spec 253: Feature Builder Patch Producer Handoff (v1.0)](../specifications/surface-253-feature-builder-patch-producer-handoff-v1_0.md)
- [x] [Surface Spec 254: Surface Body Completion Reference Evidence Matrix (v1.0)](../specifications/surface-254-surface-body-completion-reference-evidence-matrix-v1_0.md)

### Surface Body Depth Completion Architecture
- [x] [Surface Spec 255: Surface CSG Family Pair Solver Registry (v1.0)](../specifications/surface-255-surface-csg-family-pair-solver-registry-v1_0.md)
- [x] [Surface Spec 256: Surface CSG Operation Planning And Executability (v1.0)](../specifications/surface-256-surface-csg-operation-planning-and-executability-v1_0.md)
- [x] [Surface Spec 257: Surface CSG Fragment Graph Builder (v1.0)](../specifications/surface-257-surface-csg-fragment-graph-builder-v1_0.md)
- [x] [Surface Spec 258: Surface CSG Cap Patch Family Selection (v1.0)](../specifications/surface-258-surface-csg-cap-patch-family-selection-v1_0.md)
- [x] [Surface Spec 259: Surface CSG Cut Boundary Trim Construction (v1.0)](../specifications/surface-259-surface-csg-cut-boundary-trim-construction-v1_0.md)
- [x] [Surface Spec 260: Surface CSG Result Shell Assembly (v1.0)](../specifications/surface-260-surface-csg-result-shell-assembly-v1_0.md)
- [x] [Surface Spec 261: Surface CSG Result Validity Handoff (v1.0)](../specifications/surface-261-surface-csg-result-validity-handoff-v1_0.md)
- [x] [Surface Spec 262: Surface CSG Result Provenance Mapping (v1.0)](../specifications/surface-262-surface-csg-result-provenance-mapping-v1_0.md)
- [x] [Surface Spec 263: Seam Continuity Constraint Records (v1.0)](../specifications/surface-263-seam-continuity-constraint-records-v1_0.md)
- [x] [Surface Spec 264: Boundary Derivative Evaluation For Continuity (v1.0)](../specifications/surface-264-boundary-derivative-evaluation-for-continuity-v1_0.md)
- [x] [Surface Spec 265: Higher-Order Continuity Residual Validation (v1.0)](../specifications/surface-265-higher-order-continuity-residual-validation-v1_0.md)
- [ ] [Surface Spec 266: Higher-Order Continuity Violation Locators (v1.0)](../specifications/surface-266-higher-order-continuity-violation-locators-v1_0.md)
- [ ] [Surface Spec 267: Bounded Continuity Enforcement Boundary (v1.0)](../specifications/surface-267-bounded-continuity-enforcement-boundary-v1_0.md)
- [ ] [Surface Spec 268: Intersection Request And Solver Registry (v1.0)](../specifications/surface-268-intersection-request-and-solver-registry-v1_0.md)
- [ ] [Surface Spec 269: Intersection Curve Result And Degeneracy Records (v1.0)](../specifications/surface-269-intersection-curve-result-and-degeneracy-records-v1_0.md)
- [ ] [Surface Spec 270: Analytic To Spline Surface Intersection Solvers (v1.0)](../specifications/surface-270-analytic-to-spline-surface-intersection-solvers-v1_0.md)
- [ ] [Surface Spec 271: Spline To Spline Surface Intersection Solvers (v1.0)](../specifications/surface-271-spline-to-spline-surface-intersection-solvers-v1_0.md)
- [ ] [Surface Spec 272: Subdivision Surface Intersection Adapter (v1.0)](../specifications/surface-272-subdivision-surface-intersection-adapter-v1_0.md)
- [ ] [Surface Spec 273: Implicit Intersection Safety And Budget Policy (v1.0)](../specifications/surface-273-implicit-intersection-safety-and-budget-policy-v1_0.md)
- [ ] [Surface Spec 274: Bounded Implicit Contour Extraction Adapter (v1.0)](../specifications/surface-274-bounded-implicit-contour-extraction-adapter-v1_0.md)
- [ ] [Surface Spec 275: Implicit Intersection Residual And Result Classification (v1.0)](../specifications/surface-275-implicit-intersection-residual-and-result-classification-v1_0.md)
- [ ] [Surface Spec 276: Surface Reference Requirement Matrix (v1.0)](../specifications/surface-276-surface-reference-requirement-matrix-v1_0.md)
- [ ] [Surface Spec 277: Dirty Versus Promoted Reference Artifact Gate (v1.0)](../specifications/surface-277-dirty-versus-promoted-reference-artifact-gate-v1_0.md)
- [ ] [Surface Spec 278: Diagnostic Snapshot Normalization (v1.0)](../specifications/surface-278-diagnostic-snapshot-normalization-v1_0.md)
- [ ] [Surface Spec 279: Negative Diagnostic Fixture Matrix Core (v1.0)](../specifications/surface-279-negative-diagnostic-fixture-matrix-core-v1_0.md)
- [ ] [Surface Spec 280: .impress Unsafe Payload Negative Fixtures (v1.0)](../specifications/surface-280-impress-unsafe-payload-negative-fixtures-v1_0.md)
- [ ] [Surface Spec 281: Mesh Boundary Negative Fixtures (v1.0)](../specifications/surface-281-mesh-boundary-negative-fixtures-v1_0.md)
- [ ] [Surface Spec 282: CSG Negative Diagnostic Fixtures (v1.0)](../specifications/surface-282-csg-negative-diagnostic-fixtures-v1_0.md)
- [ ] [Surface Spec 283: Loft Ambiguity Negative Diagnostic Fixtures (v1.0)](../specifications/surface-283-loft-ambiguity-negative-diagnostic-fixtures-v1_0.md)
- [ ] [Surface Spec 284: Seam Continuity Negative Diagnostic Fixtures (v1.0)](../specifications/surface-284-seam-continuity-negative-diagnostic-fixtures-v1_0.md)
- [ ] [Surface Spec 285: Legacy Primitive Mesh Assumption Inventory (v1.0)](../specifications/surface-285-legacy-primitive-mesh-assumption-inventory-v1_0.md)
- [ ] [Surface Spec 286: Preview And Reference Tests Explicit Tessellation Rewrite (v1.0)](../specifications/surface-286-preview-and-reference-tests-explicit-tessellation-rewrite-v1_0.md)
- [ ] [Surface Spec 287: Mesh Compatibility API Documentation And Example Rewrite (v1.0)](../specifications/surface-287-mesh-compatibility-api-documentation-and-example-rewrite-v1_0.md)
