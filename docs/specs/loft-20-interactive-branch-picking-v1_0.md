# Loft Spec 20: Interactive Branch Picking API (v1.0)

This specification defines user-interactive branch selection for ambiguous loft
intervals in core API workflows.

It introduces a planner-exposed choice surface that allows callers to inspect
candidate decompositions and select one explicitly.

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- planner API that surfaces ambiguous interval candidate sets
- stable candidate identifiers and diagnostics
- explicit caller-selected candidate execution path
- non-interactive fallback for CLI/headless use

Out of scope:

- GUI implementation details for any specific IDE/viewer
- replacing deterministic automatic mode
- ad-hoc mesh editing outside planner/executor contracts

---

## 2. API Additions

Add plan/build controls:

- `ambiguity_mode: "fail" | "auto" | "interactive"`
- `ambiguity_selection: dict[tuple[int, int], str] | None`  
  (interval -> candidate id)
- `ambiguity_selection_policy: "required" | "best_effort"`  
  (default `"required"` in interactive mode)

Add planner query API:

- `loft_plan_ambiguities(...) -> AmbiguityReport`

Where report includes:

- ambiguous interval ids
- candidate ids
- candidate score diagnostics
- predicted action summaries

---

## 3. Candidate Identity Contract

Each candidate must expose a stable id:

- deterministic hash/signature derived from branch graph + closure ownership
- stable across runs for identical input and planner version

Candidate ids must be:

- human-referenceable in logs/CLI
- machine-usable in selection maps

---

## 4. Interactive Flow

Interactive mode flow:

1. Planner detects ambiguity and enumerates candidates.
2. Planner returns ambiguity report without executing.
3. Caller provides candidate selections per interval.
4. Planner validates selections and emits final plan.
5. Executor runs selected plan.

If selections are incomplete:

- `required` => fail with explicit missing-interval error
- `best_effort` => use deterministic fallback for missing entries

---

## 5. Validation

Selection validation must check:

- interval exists and is ambiguous
- candidate id exists for interval
- selected candidate passes topology invariants
- closure ownership remains unique

Invalid selection fails with:

- `invalid_ambiguity_selection`

---

## 6. CLI and Preview Integration Contract

Core API must remain UI-agnostic, but expose enough data for tooling:

- CLI can print interval/candidate tables
- preview/plugin can prompt user and resubmit selection map
- headless automation can store/load selection maps as JSON

No UI-specific logic is allowed inside loft kernel.

---

## 7. Metadata Contract

Plan metadata must include:

- `ambiguity_mode`
- selected interval count
- selected candidate ids per interval
- unresolved ambiguity count
- planner/candidate schema versions

---

## 8. Acceptance Tests

Required:

1. Ambiguous case returns non-empty candidate report.
2. Valid manual selection executes and yields watertight mesh.
3. Invalid interval or candidate id fails with structured error.
4. Replaying same selection map reproduces identical plan/mesh.
5. Interactive mode with `required` rejects incomplete selections.
6. Interactive mode with `best_effort` falls back deterministically.

---

## 9. Definition of Done

Spec 20 is complete when:

1. Ambiguous intervals can be inspected and selected explicitly.
2. Candidate IDs are stable and replayable.
3. Executor honors selected branches exactly.
4. CLI/tooling can integrate through public ambiguity report APIs.
