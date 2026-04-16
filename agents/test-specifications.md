# Test Specification Guidance

Test specifications define how a final feature leaf specification should be verified manually and automatically.

They are downstream of feature specifications.

A test specification exists to make feature verification durable rather than leaving it implied by chat history, ad hoc test files, or one-off manual checks.

---

## Purpose

Test specifications answer:

* how a human should sanity-check the feature in the running app
* what automated smoke coverage should exist
* what automated acceptance coverage should exist

Manual guidance may stay light.

Automated expectations should carry more detail, especially for smoke tests and acceptance tests.

---

## Scope

There should be one test specification for each final feature leaf specification.

For this rule, a "feature leaf specification" means a final leaf whose acceptance describes a product behavior, surfaced workflow, visible UI state, audible playback outcome, operator-facing tool, or other durable feature contract.

Low-level support leaves such as DTO envelopes, internal schemas, or isolated helper algorithms do not automatically need their own standalone test specifications unless they are being treated as first-class feature leaves in the project tree.

---

## Timing

When a feature specification is marked `final`, its test specification should be created in the same refinement pass.

Feature implementation should not begin with the feature leaf fully specified but its verification shape still unwritten.

For existing final feature leaves that predate this rule, test specifications should be backfilled as part of ongoing project maintenance.

---

## Location And Naming

Project test specifications should live in:

```text
project/test-specifications/
```

The preferred filename is the same basename as the feature specification it verifies.

Example:

```text
project/specifications/running-app-multi-voice-playback-switching.md
project/test-specifications/running-app-multi-voice-playback-switching.md
```

---

## Backlink Rule

Each test specification should backlink to exactly one feature specification.

That backlink must point to the feature leaf specification the test document verifies.

---

## Recommended Structure

Test specification documents should generally include:

### Overview

A short description of the feature being tested.

### Backlink

A reference to the feature specification.

### Manual Smoke Check

Short user-facing directions:

* how to reach the interface or workflow
* what to do
* what should be observed

This section may stay intentionally light.

### Automated Smoke Tests

Fast checks that prove the feature can be exercised without obvious failure.

These usually verify:

* construction or rendering
* basic happy-path execution
* absence of crashes, stalls, or empty output
* essential state transitions

### Automated Acceptance Tests

Detailed checks that verify the feature contract described by the parent specification.

These should focus on:

* surfaced outcomes
* important edge cases
* regressions already known to be likely
* durable fixtures or harness inputs that should keep passing over time

### Notes

Optional implementation-facing notes such as:

* recommended fixture documents
* required fake runtimes or test doubles
* observability hooks needed for verification

---

## Manual Guidance Principle

Manual test directions should be just enough to help a human reach the relevant surface and know what "correct" looks like.

Do not let the manual section become the most detailed part of the document unless the feature truly cannot be automated well.

---

## Automated Guidance Principle

Automated guidance should receive more effort than the manual section.

When deciding what to write, prefer:

* smoke tests that fail quickly and clearly
* acceptance tests that directly prove the feature contract
* stable fixtures over vague prose
* explicit regression cases over broad generic statements

---

## Relationship To Code Tests

Test specifications do not replace automated tests in `test/`.

They define what test coverage should exist and what the manual fallback check should be.

Implementation work should normally use the paired test specification to decide:

* what test files to add or update
* what fixture data is required
* what running-app behaviors must be verified before claiming completion

Paired feature test specifications may also appear in `project/planning/progression.md` as explicit verification work items so implementation and testing progress can be tracked separately.

---

## Guiding Principle

A final feature specification should have a durable verification contract, not only an implementation contract.
