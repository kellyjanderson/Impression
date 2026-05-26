# Loft Spec 56: High-Confidence Inference and Refusal Policy (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `High-Confidence Inference And Refusal Policy`

## Purpose

Define the bounded geometric inference policy used only after authored rails do
not fully resolve point correspondence, including scoring gates, refusal
conditions, and diagnostics.

## Scope

Owns:

- Candidate scoring for cyclic shift and optional reversal.
- Acceptance thresholds for high-confidence inferred correspondence.
- Refusal diagnostics when inference is ambiguous or unsafe.

Does not own:

- Authored rail priority.
- Point birth/death support insertion.
- Executor behavior or mesh/surface fallback.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - add inference scorer, acceptance gate,
    and refusal reporter.
- Supporting modules/files:
  - `src/impression/modeling/topology.py` - source topology identity records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable planner inference policy.
- Tests:
  - `tests/test_loft_correspondence_inference_policy.py` - accepted inference,
    ties, reversal refusal, and ranked diagnostics.

## Chosen Defaults / Parameters

- Inference runs only when authored rails are insufficient and non-conflicting.
- Automatic inference requires no hard refusal condition.
- Automatic inference requires at least two stable protected anchors, or
  compatible authored point counts with preserved authored starts.
- Best normalized cost must be `<= 0.20`.
- Second-best separation must be `>= max(0.10, best_cost * 0.50)`.
- Reversal is allowed only when topology semantics allow it and it does not
  conflict with authored direction.
- Medium- or low-confidence candidates are diagnostic only and cannot become
  executable correspondence.

## Data Ownership

- Source of truth: authored rail map and topology records remain authoritative;
  inference owns only derived candidate rankings and accepted inferred maps.
- Read ownership: loft planner reads inference results and diagnostics.
- Write ownership: inference policy writes accepted inferred maps or refusal
  diagnostics into the planner result.
- Derived/cache data: candidate scores are recomputable from normalized loops,
  protected anchors, and prior interval continuity.
- Privacy/logging constraints: diagnostics may include ids, normalized costs,
  and refusal reasons; no external user data.

## Dependencies And Routes

- Domain/service dependencies:
  - `RailResolutionResult`
  - loop correspondence planner
  - ambiguity diagnostics
  - topology protected landmarks
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; deterministic synchronous planner step.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing loft ambiguity diagnostics.
  - Existing loop cost or cyclic-shift helpers if present.
- Current reuse readiness:
  - add to existing loft planner module.
- Extraction/wrapping needed:
  - wrap any current heuristic into the explicit candidate/result DTOs here.
- Additions to existing library/modules:
  - `score_correspondence_candidates(...)`
  - `accept_or_refuse_inferred_correspondence(...)`
  - `InferenceCandidateScore`
  - `InferenceRefusalDiagnostic`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `InferenceCandidateScore` - fields: `shift`, `reversed`, `cost`,
    `cost_terms`, `protected_anchor_agreement`, `prior_interval_agreement`.
  - `InferenceRefusalDiagnostic` - fields: `reason`, `best_candidate`,
    `second_best_candidate`, `required_rail_hint`.
- Functions/methods:
  - `score_correspondence_candidates(source_loop, target_loop,
    rail_result) -> tuple[InferenceCandidateScore, ...]`
  - `accept_or_refuse_inferred_correspondence(candidates, *,
    topology_semantics) -> InferenceResult`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Candidate enumeration is bounded by loop point count and optional reversal:
  O(n) candidates, each scored O(n), for O(n^2) worst-case per loop pair.
- Specs that invoke inference must preserve existing ambiguity branch limits.
- Diagnostics keep only ranked top candidates needed for refusal explanation.

## Error And State Behavior

- Multiple equally plausible candidates refuse with `ambiguous_phase`.
- Protected-anchor crossing refuses with `crossing_protected_order`.
- Reversal conflicts refuse with `reversal_conflicts_with_authored_direction`.
- No stable anchors refuses with `missing_stable_anchors`.
- Inference failure returns diagnostics, not a mesh fallback.

## Test Strategy

- Unit tests:
  - unique high-confidence cyclic shift accepted.
  - equal-cost symmetric loops refused.
  - second-best separation below threshold refused.
  - reversal accepted only when allowed.
  - protected-anchor crossing refused.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- The planner accepts only high-confidence inferred maps.
- Ambiguous correspondence produces actionable rail-request diagnostics.
- No inferred result overrides explicit authored rails.

## Readiness Checklist

- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
