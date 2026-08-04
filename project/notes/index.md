---
created: 2026-07-23
---

# Impression Critical Review Index

## Focused Capability Inventories

[Loft missing and incomplete capabilities](loft-missing-and-incomplete-capabilities.md)

## Starting Discussion Issues

- [ ] [Dirty STL triage](impression-critical-review-planning.md#dirty-stl-triage)
- [ ] [Correspondence policy](impression-critical-review-planning.md#correspondence-policy)
- [ ] [Text review](impression-critical-review-planning.md#text-review)
- [ ] [Endcap parity](impression-critical-review-planning.md#endcap-parity)

## Current Failing-Test Issues

- [ ] [Documentation policy tests target retired paths](impression-critical-review-planning.md#documentation-policy-tests-target-retired-paths)
- [ ] [Reference expansion CSG test expects a retired refusal](impression-critical-review-planning.md#reference-expansion-csg-test-expects-a-retired-refusal)
- [ ] [Hinge documentation omits the required surface selector](impression-critical-review-planning.md#hinge-documentation-omits-the-required-surface-selector)
- [ ] [Cone connectivity assertion predates the top-cap fix](impression-critical-review-planning.md#cone-connectivity-assertion-predates-the-top-cap-fix)

## Code Quality Issues

[Code quality review](code-quality-principal-engineer-review.md)

- [ ] **P0:** [The public modeling API is an internal inventory, not a curated interface](code-quality-principal-engineer-review.md#p0-the-public-modeling-api-is-an-internal-inventory-not-a-curated-interface)
- [ ] **P0:** [Core geometry modules contain process governance instead of geometry](code-quality-principal-engineer-review.md#p0-core-geometry-modules-contain-process-governance-instead-of-geometry)
- [ ] **P0:** [The loft interface is a parameter train repeated through wrapper layers](code-quality-principal-engineer-review.md#p0-the-loft-interface-is-a-parameter-train-repeated-through-wrapper-layers)
- [ ] **P1:** [Long names are compensating for missing concept boundaries](code-quality-principal-engineer-review.md#p1-long-names-are-compensating-for-missing-concept-boundaries)
- [ ] **P1:** [Simple operations cross too many conceptual layers](code-quality-principal-engineer-review.md#p1-simple-operations-cross-too-many-conceptual-layers)
- [ ] **P1:** [The core is over-modeled with near-duplicate record types](code-quality-principal-engineer-review.md#p1-the-core-is-over-modeled-with-near-duplicate-record-types)
- [ ] **P1:** [Policy is repeated as prose instead of encoded once](code-quality-principal-engineer-review.md#p1-policy-is-repeated-as-prose-instead-of-encoded-once)
- [ ] **P1:** [Module ownership is cyclic and hidden by local imports](code-quality-principal-engineer-review.md#p1-module-ownership-is-cyclic-and-hidden-by-local-imports)
- [ ] **P1:** [Type signatures are broad where the API most needs clarity](code-quality-principal-engineer-review.md#p1-type-signatures-are-broad-where-the-api-most-needs-clarity)
- [ ] **P1:** [Large functions mix orchestration, validation, and domain logic](code-quality-principal-engineer-review.md#p1-large-functions-mix-orchestration-validation-and-domain-logic)
- [ ] **P2:** [Duplicate paths make it hard to know which implementation is real](code-quality-principal-engineer-review.md#p2-duplicate-paths-make-it-hard-to-know-which-implementation-is-real)
- [ ] **P2:** [The reference review UI has a large construction and coordination object](code-quality-principal-engineer-review.md#p2-the-reference-review-ui-has-a-large-construction-and-coordination-object)
- [ ] **P2:** [Documentation density does not match abstraction density](code-quality-principal-engineer-review.md#p2-documentation-density-does-not-match-abstraction-density)

## Efficiency And Reuse Issues

[Efficiency and reuse review](efficiency-and-reuse-principal-engineer-review.md)

- [ ] **P0:** [Scene consumption has multiple incompatible implementations](efficiency-and-reuse-principal-engineer-review.md#p0-scene-consumption-has-multiple-incompatible-implementations)
- [ ] **P1:** [Hot reload invalidates far more than user code](efficiency-and-reuse-principal-engineer-review.md#p1-hot-reload-invalidates-far-more-than-user-code)
- [ ] **P1:** [Cancellation discards results but not expensive work](efficiency-and-reuse-principal-engineer-review.md#p1-cancellation-discards-results-but-not-expensive-work)
- [ ] **P1:** [Text converts curve sampling into topological complexity](efficiency-and-reuse-principal-engineer-review.md#p1-text-converts-curve-sampling-into-topological-complexity)
- [ ] **P1:** [Tessellation repeatedly re-derives shell facts](efficiency-and-reuse-principal-engineer-review.md#p1-tessellation-repeatedly-re-derives-shell-facts)
- [ ] **P2:** [Numerical and transform kernels have multiple owners](efficiency-and-reuse-principal-engineer-review.md#p2-numerical-and-transform-kernels-have-multiple-owners)
- [ ] **P2:** [The modeling core mixes records, policy, evidence, and execution](efficiency-and-reuse-principal-engineer-review.md#p2-the-modeling-core-mixes-records-policy-evidence-and-execution)
- [ ] **P2:** [Shadowed and orphaned modules carry cost without capability](efficiency-and-reuse-principal-engineer-review.md#p2-shadowed-and-orphaned-modules-carry-cost-without-capability)
- [ ] **P2:** [Correspondence assignment has explicit exponential ceilings](efficiency-and-reuse-principal-engineer-review.md#p2-correspondence-assignment-has-explicit-exponential-ceilings)
- [ ] **P2:** [Font discovery repeats host-wide recursive scans](efficiency-and-reuse-principal-engineer-review.md#p2-font-discovery-repeats-host-wide-recursive-scans)
- [ ] **P2:** [Serialization reuse is incomplete](efficiency-and-reuse-principal-engineer-review.md#p2-serialization-reuse-is-incomplete)
- [ ] **P3:** [There is no performance regression gate](efficiency-and-reuse-principal-engineer-review.md#p3-there-is-no-performance-regression-gate)

## Technical And Industry Completeness Issues

[Technical and industry completeness review](technical-and-industry-completeness-principal-engineer-review.md)

- [ ] **P0:** [Installation does not prove a usable product](technical-and-industry-completeness-principal-engineer-review.md#p0-installation-does-not-prove-a-usable-product)
- [ ] **P0:** [The documented first model cannot reach preview or export](technical-and-industry-completeness-principal-engineer-review.md#p0-the-documented-first-model-cannot-reach-preview-or-export)
- [ ] **P0:** [The solid and watertight product claim is not met](technical-and-industry-completeness-principal-engineer-review.md#p0-the-solid-and-watertight-product-claim-is-not-met)
- [ ] **P0:** [Documented boolean workflows are not a coherent feature](technical-and-industry-completeness-principal-engineer-review.md#p0-documented-boolean-workflows-are-not-a-coherent-feature)
- [ ] **P1:** [Units are metadata and labels, not geometric semantics](technical-and-industry-completeness-principal-engineer-review.md#p1-units-are-metadata-and-labels-not-geometric-semantics)
- [ ] **P1:** [Manufacturing validation is too shallow](technical-and-industry-completeness-principal-engineer-review.md#p1-manufacturing-validation-is-too-shallow)
- [ ] **P1:** [Native persistence is rich but isolated](technical-and-industry-completeness-principal-engineer-review.md#p1-native-persistence-is-rich-but-isolated)
- [ ] **P1:** [External interchange is below both additive and CAD baselines](technical-and-industry-completeness-principal-engineer-review.md#p1-external-interchange-is-below-both-additive-and-cad-baselines)
- [ ] **P1:** [Foundational solid modeling operations are missing or retired](technical-and-industry-completeness-principal-engineer-review.md#p1-foundational-solid-modeling-operations-are-missing-or-retired)
- [ ] **P1:** [Surface modeling lacks higher-order continuity completion](technical-and-industry-completeness-principal-engineer-review.md#p1-surface-modeling-lacks-higher-order-continuity-completion)
- [ ] **P1:** [Reference review cannot yet be a release evidence system](technical-and-industry-completeness-principal-engineer-review.md#p1-reference-review-cannot-yet-be-a-release-evidence-system)
- [ ] **P2:** [Text geometry exists, but typography and packaging are incomplete](technical-and-industry-completeness-principal-engineer-review.md#p2-text-geometry-exists-but-typography-and-packaging-are-incomplete)
- [ ] **P2:** [Loft correspondence needs a product-level confidence contract](technical-and-industry-completeness-principal-engineer-review.md#p2-loft-correspondence-needs-a-product-level-confidence-contract)
- [ ] **P2:** [Endcaps exist, but the feature family is split](technical-and-industry-completeness-principal-engineer-review.md#p2-endcaps-exist-but-the-feature-family-is-split)
- [ ] **P2:** [Assemblies, constraints, and product structure are absent](technical-and-industry-completeness-principal-engineer-review.md#p2-assemblies-constraints-and-product-structure-are-absent)
- [ ] **P2:** [Documentation and assurance do not execute product truth](technical-and-industry-completeness-principal-engineer-review.md#p2-documentation-and-assurance-do-not-execute-product-truth)

## Correctness And Release Integrity Issues

[Correctness and release integrity review](correctness-and-release-integrity-principal-engineer-review.md)

- [ ] **P0:** [User-model loading creates a split-brain modeling runtime](correctness-and-release-integrity-principal-engineer-review.md#p0-user-model-loading-creates-a-split-brain-modeling-runtime)
- [ ] **P0:** [The canonical first preview and export contract is broken](correctness-and-release-integrity-principal-engineer-review.md#p0-the-canonical-first-preview-and-export-contract-is-broken)
- [ ] **P0:** [Docs archive extraction permits path traversal](correctness-and-release-integrity-principal-engineer-review.md#p0-docs-archive-extraction-permits-path-traversal)
- [ ] **P0:** [Public CSG signatures and examples do not match runtime behavior](correctness-and-release-integrity-principal-engineer-review.md#p0-public-csg-signatures-and-examples-do-not-match-runtime-behavior)
- [ ] **P0:** [Release gates do not verify the distribution contract](correctness-and-release-integrity-principal-engineer-review.md#p0-release-gates-do-not-verify-the-distribution-contract)
- [ ] **P1:** ["Solid" primitive output frequently is not closed](correctness-and-release-integrity-principal-engineer-review.md#p1-solid-primitive-output-frequently-is-not-closed)
- [ ] **P1:** [Export bypasses watertight and unit contracts](correctness-and-release-integrity-principal-engineer-review.md#p1-export-bypasses-watertight-and-unit-contracts)
- [ ] **P1:** [Reference promotion is not atomic despite its contract](correctness-and-release-integrity-principal-engineer-review.md#p1-reference-promotion-is-not-atomic-despite-its-contract)
- [ ] **P1:** [Mirroring mesh output reverses solid orientation](correctness-and-release-integrity-principal-engineer-review.md#p1-mirroring-mesh-output-reverses-solid-orientation)
- [ ] **P1:** [Preview worker cancellation does not cancel work](correctness-and-release-integrity-principal-engineer-review.md#p1-preview-worker-cancellation-does-not-cancel-work)
- [ ] **P1:** [Importing the package mutates external state](correctness-and-release-integrity-principal-engineer-review.md#p1-importing-the-package-mutates-external-state)
- [ ] **P2:** [CAD packaging contains a shadowed, unreachable adapter](correctness-and-release-integrity-principal-engineer-review.md#p2-cad-packaging-contains-a-shadowed-unreachable-adapter)
- [ ] **P2:** [Dependency and artifact metadata drift](correctness-and-release-integrity-principal-engineer-review.md#p2-dependency-and-artifact-metadata-drift)
- [ ] **P2:** [Frozen geometry still contains mutable identity state](correctness-and-release-integrity-principal-engineer-review.md#p2-frozen-geometry-still-contains-mutable-identity-state)
- [ ] **P2:** [Ownership boundaries are too large to review reliably](correctness-and-release-integrity-principal-engineer-review.md#p2-ownership-boundaries-are-too-large-to-review-reliably)

## Strict TDD Coverage Issues

[Strict TDD coverage review](test-coverage-tdd-review.md)

- [ ] **P0:** [There is no green executable specification](test-coverage-tdd-review.md#p0-there-is-no-green-executable-specification)
- [ ] **P0:** [Test outcomes depend on execution order and shared runtime state](test-coverage-tdd-review.md#p0-test-outcomes-depend-on-execution-order-and-shared-runtime-state)
- [ ] **P1:** [The suite does not provide a usable TDD feedback ladder](test-coverage-tdd-review.md#p1-the-suite-does-not-provide-a-usable-tdd-feedback-ladder)
- [ ] **P1:** [Many tests specify private implementation instead of stable behavior](test-coverage-tdd-review.md#p1-many-tests-specify-private-implementation-instead-of-stable-behavior)
- [ ] **P1:** [Coverage-chasing tests bundle unrelated behaviors](test-coverage-tdd-review.md#p1-coverage-chasing-tests-bundle-unrelated-behaviors)
- [ ] **P1:** [Whole product modules have no test-driven contract](test-coverage-tdd-review.md#p1-whole-product-modules-have-no-test-driven-contract)
- [ ] **P1:** [Public workflows are under-tested relative to internal machinery](test-coverage-tdd-review.md#p1-public-workflows-are-under-tested-relative-to-internal-machinery)
- [ ] **P1:** [Coverage is configured but not a development gate](test-coverage-tdd-review.md#p1-coverage-is-configured-but-not-a-development-gate)
- [ ] **P2:** [Test ownership is concentrated in giant files](test-coverage-tdd-review.md#p2-test-ownership-is-concentrated-in-giant-files)
- [ ] **P2:** [Numerical and topological invariants lack generative verification](test-coverage-tdd-review.md#p2-numerical-and-topological-invariants-lack-generative-verification)
- [ ] **P2:** [Markers are assigned by filename instead of test contract](test-coverage-tdd-review.md#p2-markers-are-assigned-by-filename-instead-of-test-contract)
- [ ] **P2:** [Golden tests protect existing output, including known-bad output](test-coverage-tdd-review.md#p2-golden-tests-protect-existing-output-including-known-bad-output)

## Principal QA Coverage Issues

[Principal QA coverage review](test-coverage-principal-qa-review.md)

- [ ] **P0:** [The actual CI gate is narrow and does not complete reliably](test-coverage-principal-qa-review.md#p0-the-actual-ci-gate-is-narrow-and-does-not-complete-reliably)
- [ ] **P0:** [The release workflow publishes untested artifacts](test-coverage-principal-qa-review.md#p0-the-release-workflow-publishes-untested-artifacts)
- [ ] **P0:** [There is no stable integrated system test result](test-coverage-principal-qa-review.md#p0-there-is-no-stable-integrated-system-test-result)
- [ ] **P1:** [Supported runtime and platform claims are not qualified](test-coverage-principal-qa-review.md#p1-supported-runtime-and-platform-claims-are-not-qualified)
- [ ] **P1:** [Installation, upgrade, and distribution behavior are uncovered](test-coverage-principal-qa-review.md#p1-installation-upgrade-and-distribution-behavior-are-uncovered)
- [ ] **P1:** [Canonical user workflows are not covered end to end](test-coverage-principal-qa-review.md#p1-canonical-user-workflows-are-not-covered-end-to-end)
- [ ] **P1:** [Reference artifacts are not yet release-authoritative evidence](test-coverage-principal-qa-review.md#p1-reference-artifacts-are-not-yet-release-authoritative-evidence)
- [ ] **P1:** [Geometry regression equality is not manufacturing qualification](test-coverage-principal-qa-review.md#p1-geometry-regression-equality-is-not-manufacturing-qualification)
- [ ] **P1:** [Concurrency and cancellation tests do not qualify lifecycle behavior](test-coverage-principal-qa-review.md#p1-concurrency-and-cancellation-tests-do-not-qualify-lifecycle-behavior)
- [ ] **P1:** [Security and abuse-case coverage is incomplete](test-coverage-principal-qa-review.md#p1-security-and-abuse-case-coverage-is-incomplete)
- [ ] **P2:** [Nonfunctional quality has no automated qualification](test-coverage-principal-qa-review.md#p2-nonfunctional-quality-has-no-automated-qualification)
- [ ] **P2:** [Requirements and test traceability are informal](test-coverage-principal-qa-review.md#p2-requirements-and-test-traceability-are-informal)
- [ ] **P2:** [CI does not preserve enough evidence for triage or audit](test-coverage-principal-qa-review.md#p2-ci-does-not-preserve-enough-evidence-for-triage-or-audit)
- [ ] **P2:** [Test data and environment reproducibility are not controlled end to end](test-coverage-principal-qa-review.md#p2-test-data-and-environment-reproducibility-are-not-controlled-end-to-end)
- [ ] **P2:** [Persistence coverage lacks a released-version compatibility corpus](test-coverage-principal-qa-review.md#p2-persistence-coverage-lacks-a-released-version-compatibility-corpus)
