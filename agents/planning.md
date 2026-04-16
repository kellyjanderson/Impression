# Planning Guidance

Planning documents define the implementation and verification sequence for specifications.

The primary planning artifact is a dependency-ordered list of specification references with checkboxes for completion in planning/progression.md.

Only final leaf specifications and paired feature test specifications belong in the progression document.

Parent or umbrella specifications must never appear in progression, because they are not executable work units and are never marked complete directly.

Planning answers:

* what must be completed before other work can proceed
* what order specifications should be implemented in
* what has been completed

Planning does not define version intent by itself. That role belongs to release definitions.

---

## Purpose

Planning exists to sequence implementation work.

It provides a simple execution document that defines which specifications must be completed before others and tracks completion as work progresses.

---

## Structure

Planning documents should consist of dependency-ordered specification references grouped into implementation lanes.

Those references must point only to:

- final leaf specifications
- paired feature test specifications

A planning document will usually include:

### Core Functionality

Specifications required for the system to function.

### Obligate Specifications

Specifications that are not strictly required for minimal functionality, but are naturally expected to be present in a complete implementation.

### Polish Specifications

Specifications that improve refinement, usability, clarity, or completeness beyond the obligate level.

Each lane should contain specification references arranged in dependency order, with checkboxes indicating completion state.

Planning documents should not contain:

- parent specifications
- umbrella specifications
- informational tracking sections for non-leaf work

If parent specifications still need attention, that should be handled through further specification refinement, not by placing them in progression.

This means specification work should first build out the tree deeply enough that all executable work has been pushed down to final leaves.

Planning begins only from those final leaves, regardless of how many refinement generations were required to reach them.

Because final leaves are meant to be cohesive implementation-sized units rather than artificially tiny fragments, progression should normally reference leaves that correspond to meaningful chunks of work, not the smallest imaginable subtopics.

For feature leaves, the paired test specification should already exist before the leaf enters active implementation through progression.

---

## Dependencies

Planning should reflect dependency order.

A specification should appear after any specifications that must be completed before it can be implemented correctly or reasonably.

Dependencies may span across lanes.

---

## Secondary Ordering

When multiple specifications are unblocked (no remaining dependencies), a secondary ordering is used.

Secondary ordering is alphabetical.

This provides a simple and deterministic ordering when dependency order does not determine the next step.

Actual implementation priority may still be chosen manually.

---

## Completion Tracking

Planning documents should track completion directly.

Checkboxes should be used to indicate whether a referenced specification has been implemented.

For feature leaves, the implementation checkbox should be marked complete when the implementation work for that leaf is done.

Paired test specification checkboxes exist to track the remaining verification work separately.

That means agents should not leave a feature leaf unchecked merely because manual or broader acceptance verification is still pending.

Instead:

- check the feature leaf when implementation is complete
- leave the paired test specification unchecked until verification is complete

Because only final leaf specifications and paired feature test specifications belong in progression, only those items are ever marked complete there.

---

## Assignment

In small projects, assignment may be trivial and may be omitted.

If needed, planning may also indicate who is responsible for implementing a specification.

---

## Relationship to Other Documents

* architecture defines the system
* release definitions define the holistic version scope
* specifications define the implementation work
* planning defines the order in which specifications are implemented

---

## Guiding Principle

Planning is a sequencing document for specification implementation.
