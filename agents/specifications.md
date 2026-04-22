# Specification Guidance

Specification documents define how architecture is refined into actionable implementation work.

A specification represents a unit of work derived from either:

* an architecture document, or
* a UI definition document, or
* another specification document

Each specification must have exactly one backlink to its parent document.

This parent may be:

* an architecture document, or
* a UI definition document, or
* a specification document

A specification should not have multiple parents.

---

## Purpose

Specifications translate architecture into implementable form.

For UI-specific work, specifications may also translate stabilized UI definitions into implementable form.

They isolate individual implementation concerns and refine them until they can be acted on directly.

Specifications are downstream of architecture.

They should only be written after the relevant architecture branch has been completed breadth first so the system-level picture for that area is already known.

---

## Specification Process

Specifications are not written in a one-shot process.

They are produced through iterative refinement.

By default, specification refinement may be performed locally by one agent
across multiple rounds.

On explicit user request, specification refinement may instead use a fixed
multi-agent review pipeline so the work receives multiple guaranteed passes.

Specification refinement is recursive.

The expected result is a tree structure that may grow both wide and deep:

* architecture produces first-generation specifications
* those specifications may produce child specifications
* those children may produce their own children
* refinement continues downward until the current leaves are sized appropriately for clean implementation

Each architecture document is first broken down into specifications that isolate major implementation concerns described by that architecture.

That step happens only after the relevant architecture documents are complete enough to describe the whole system area being refined.

If important architecture-level relationships, data flow, or cross-domain decisions are still missing, specification work should pause and the process should return to architecture.

After this first round, there should usually be multiple specifications associated with a single architecture document.

Each specification is then evaluated to determine whether it is implementable as written or requires further refinement.

* If a specification can be implemented directly, it should be marked final.
* If a specification cannot yet be implemented directly, it should be refined into one or more child specifications.

If a specification still contains more work than can reasonably be completed in one clean implementation round, it is not final yet, even if the overall idea is coherent.

Refinement should not optimize for the smallest possible leaves.

The goal is to arrive at the largest cohesive implementation unit that can still be completed cleanly in one round without hiding major unresolved decisions.

If several concerns are tightly coupled and would naturally be implemented together in one pass, they should usually stay in one final leaf rather than being split only for the sake of atomization.

Each child specification must backlink to the specification it refines.

This process continues until all leaf specifications are marked final.

For user-facing feature branches, refinement is not complete merely because the internal support leaves are final.

At least one final leaf in a user-facing branch must terminate in behavior that can be observed in the running app.

Examples of observable outcomes include:

* a visible control or UI state
* an audible playback change
* a visible navigation or follow-along behavior
* a surfaced settings behavior such as appearance-mode following

Support or enabling leaves may still be final.

They do not, by themselves, complete the specification of a user-facing feature branch.

Agents should expect to review newly created child specifications in later rounds and either:

* mark them final if they are now implementation-sized, or
* refine them again into smaller child specifications

When the user explicitly requests a multi-agent specification pipeline, the
preferred default is a three-stage handoff:

1. agent 1 drafts or refines the specification tree
2. agent 2 reviews that output for scope bleed, hidden assumptions, and
   non-leaf bundling
3. agent 3 rewrites or refines the result for final clarity, progression
   alignment, and paired test-spec completeness

This pipeline is mediated by the main agent.
Sub-agents do not need to hand work directly to each other.
The main agent may receive the output from one stage and pass the relevant
result into the next stage.

The purpose of this mode is not parallelism.
Its purpose is guaranteed multi-pass refinement when a single agent is likely
to stop too early or refine inconsistently.

This multi-agent path is optional.
The default local refinement path remains valid, and both methods may coexist
so their results can be compared over time.

When deciding whether to refine further, prefer asking:

* would this realistically be implemented as one coherent change set?
* does this leaf still hide another full round of work?
* are there unresolved decisions that would force further design during implementation?

If the answer to those questions is "no," the leaf should usually remain final even if it still contains multiple closely related details.

The process is complete only when every executable leaf in the tree is final.

For user-facing feature work, that means both:

* the supporting implementation leaves are final
* the surfaced running-app outcome leaves are final

At the end of each specification pass, the agent should explicitly report how many specifications still require another refinement round.

That count should reflect the number of branches that still need another refinement step to create or clarify missing executable leaves.

Non-executable parent containers whose implied work is already fully represented by final child leaves do not count as needing another refinement round just because the parent document itself remains broad.

---

## Scope

A specification should cover one implementation concern, such as:

* a feature
* a component
* a behavior
* an interface
* a workflow

A specification should not be so broad that major design decisions still need to be made during implementation.

---

## Final Specifications

A final specification is a specification that can be implemented directly.

A final specification should be small enough and clear enough that implementation can proceed without further specification-level refinement.

`Final` does not mean "complete as a thought" or "the whole concern is now described."

`Final` means the remaining work described by that specification is sized appropriately to be implemented cleanly in one round of work.

If a specification still bundles multiple rounds of implementation work together, it must not be marked final. That work must be pushed down into child specifications until the leaf specifications represent implementation-sized units.

This does not mean every distinguishable subtopic needs its own child specification.

The preferred granularity is the largest cohesive unit that:

* can be implemented in one round
* does not hide major unresolved decisions
* does not imply a second clean implementation pass that should have been specified separately

All implementation work implied by a specification must be represented at the leaf level before execution. Parent specifications may remain broad, but executable work belongs in final leaf specifications only.

For a specification to be considered fully complete in project delivery terms, the resulting implementation must also satisfy:

* paired verification expectations
* durable documentation expectations

A feature is not fully complete without durable documentation.

If a parent specification still implies work that is not represented by child leaves, refinement is not finished yet.

For user-facing branches, a final leaf should make it impossible to claim success based only on internal plumbing.

If the intended feature outcome is something a user should be able to see, hear, or interact with in the running app, at least one final leaf must describe that surfaced outcome explicitly.

Useful questions are:

* can a user observe this outcome in the running app?
* would this leaf still be marked complete if only internal state or infrastructure existed?
* is there another full surfaced behavior pass still hidden after this leaf?

If a user-facing feature could still remain invisible after the current leaves are implemented, refinement is not done yet.

When a feature leaf is marked final, a paired test specification should also be created so the feature has both an implementation contract and a verification contract.

Documentation expectations should also be explicit enough that the feature does not ship with durable docs left implied.

Final features should not ship with documentation left implied.

---

## Specification Size Index

To keep refinement decisions consistent, specifications should be evaluated with
a deterministic size score before being marked final.

This score is not an estimate of implementation hours.

It is a structural sizing tool that answers:

* does this specification still bundle too many independent concerns?
* is this realistically one clean implementation round?
* should this be refined into child specifications before implementation?

This score is called the **Specification Size Index (SSI)**.

### Counting Rules

Count only concerns that are explicitly present in the specification text.

Do not infer hidden future work that is not written.

For sizing, count the following:

### 1. Implementation Concerns

Count the number of distinct implementation concerns explicitly listed in
`Scope`, `Behavior`, `Constraints`, and `Acceptance`.

Examples:

* new data model
* API contract
* migration path
* tessellation policy
* cache behavior
* UI workflow

Rule:

* 1 point per distinct concern

### 2. New or Changed Durable Contracts

Count each explicit durable contract introduced or changed.

Examples:

* a new public API
* a new internal core type
* a new planner/executor contract
* a new persisted document/schema contract

Rule:

* 2 points per durable contract

### 3. System Boundaries Crossed

Count the number of system boundaries the specification must coordinate across.

Examples:

* authored geometry -> topology
* topology -> surface kernel
* surface -> tessellation
* modeling -> preview/export
* planner -> executor

Rule:

* 2 points per boundary crossed

Only count boundaries explicitly named in the specification.

### 4. Distinct Caller or Consumer Classes

Count each distinct class of caller or consumer that must be supported.

Examples:

* modeling APIs
* preview
* export
* analysis
* CLI
* VS Code extension

Rule:

* 1 point per caller/consumer class

### 5. Migration or Compatibility Responsibilities

Count each explicit migration or compatibility responsibility.

Examples:

* adapter layer
* fallback path
* temporary compatibility mode
* deprecation path

Rule:

* 1 point per migration or compatibility responsibility

### 6. Acceptance Burden

Count the number of independent acceptance outcomes.

Examples:

* one new runtime-visible behavior
* one deterministic serialization guarantee
* one watertight tessellation guarantee
* one compatibility guarantee

Rule:

* 1 point per independent acceptance outcome

### Formula

```text
SSI =
  implementation_concerns
  + 2 * durable_contracts
  + 2 * system_boundaries
  + caller_consumer_classes
  + migration_compatibility_responsibilities
  + acceptance_outcomes
```

### Thresholds

Use these thresholds consistently:

* `SSI <= 9`
  Final-candidate size. This specification may be marked final if it contains no
  unresolved architectural decisions.

* `SSI 10-14`
  Borderline size. This specification requires an explicit judgment call and
  should usually be refined unless the concerns are tightly coupled enough to be
  implemented in one coherent change set.

* `SSI >= 15`
  Too large. This specification should be refined into child specifications
  before implementation.

### Hard Stop Conditions

A specification must be refined, even if its SSI is low, when any of the
following are true:

* it contains unresolved architectural decisions
* it implies multiple clean implementation rounds
* it mixes more than one independent migration track
* it requires both kernel-definition work and downstream feature adoption work
  in the same leaf
* it is still acting as an umbrella container rather than an implementation leaf

### Reporting Requirement

During refinement passes, agents should explicitly report:

* the estimated SSI for the specification being evaluated
* whether it is final-candidate, borderline, or too large
* which hard-stop condition applies, if any

### Intended Use

SSI is a consistency tool, not a substitute for judgment.

Its purpose is to prevent repeated ambiguity about whether a specification is
"still too big" by giving the project one shared sizing language.

---

## Recommended Structure

Specification documents should generally include:

### Overview

A short description of the concern being specified.

### Backlink

A reference to the parent architecture or specification document.

### Scope

What this specification covers.

### Behavior

What this part of the system should do.

### Constraints

Requirements that must be satisfied.

### Refinement Status

Whether the specification is final or requires further refinement.

Parent specifications that already have the final child leaves needed for the current planned branch should say so explicitly.

Do not leave `Requires refinement` in place once the branch is fully decomposed into the leaves needed to guide implementation.

### Child Specifications

References to any specification documents derived from this one.

### Acceptance

What must be true for the work to be considered complete.

For user-facing final leaves, acceptance should explicitly state what can be observed in the running app.

If a user-visible surface exists in the running app, that surface should have a named parent UI/specification branch rather than being implied only by scattered leaf behavior.

For feature leaves, acceptance should be mirrored by a paired test specification in `project/test-specifications/` that defines manual smoke checks plus automated smoke and acceptance coverage.

---

## Relationship to Architecture

Architecture defines the system-level solution.

Specifications refine that solution into implementable units.

Specifications should remain aligned to the parent document they refine.

For UI-specific branches, specifications may also refine stabilized UI definitions into implementable UI behavior.

Issue-driven fixes must also remain aligned to this tree.

If a bug fix changes durable system behavior, the affected specification or architecture document should be updated so the fix is represented in project truth rather than only in issue history and code.
