# Delegation Guidance

This document defines how the main agent should delegate work in this repository.

---

## Core Model

The main agent is the orchestrator and the user-facing coordinator.

It owns:

* understanding the request
* choosing the execution plan
* assigning bounded work to sub-agents
* reviewing returned results
* integrating accepted changes
* reporting final outcomes and remaining risk to the user

Sub-agents are execution units, not final decision makers.

The main agent remains accountable for the finished result even when most of the work was delegated.

---

## Default Execution Path

The main agent should work locally by default.

Sub-agents should be used selectively for bounded background work that does not need continuous user interaction.

Use delegation when a task can be stated with:

* a clear objective
* a limited context window
* a defined write scope or read scope
* a recognizable completion condition

Typical delegation targets include:

* focused codebase exploration
* isolated background implementation tasks
* targeted test creation or execution
* narrow debugging or reproduction work
* independent verification of a proposed change

Delegation is most useful when it lets the main agent stay available for interactive work rather than when it merely moves the immediate blocking task elsewhere.

---

## Responsiveness And Capacity

Delegation should stay conservative by default.

Most prompts should use a single delegated sub-agent unless there is a clear benefit to parallel delegation.

The main agent should generally avoid saturating all available sub-agent capacity.
When practical, keep one sub-agent free so the system remains responsive to new incoming work.

If parallel delegation is used, it should be because the work is truly independent, the benefit is clear, and the write scopes remain cleanly separated.

---

## Background Work

Sub-agents are a good fit for long-running planned work when the main agent would otherwise have little to do except wait for implementation to finish.

This is especially useful for:

* progression items in a narrow code slice
* long implementation passes with clear ownership
* bounded test or regeneration runs
* background investigation that can conclude with a concrete answer

In those cases, the main agent may assign one sub-agent as the single owner of that slice and keep the main thread available for:

* new user questions
* related planning or documentation work
* integration review once the background task returns
* other non-conflicting interactive tasks

Background delegation should not turn into hidden multi-owner work.
If the task lives in one narrow code slice, one sub-agent should usually own that slice from start to finish.

---

## Fixed Review Pipelines

Sub-agents may also be used in a fixed sequential review pipeline when the
goal is guaranteed multiple refinement passes rather than background parallel
work.

This is especially appropriate for:

* specification refinement on explicit user request
* adversarial review of a draft before it is marked final
* durable multi-pass critique where one agent tends to stop too early

The preferred pattern is:

1. assign agent 1 to produce the first draft or refinement pass
2. pass that result to agent 2 for critique and narrowing
3. pass the revised result to agent 3 for final refinement or rewrite

In this mode, the main agent remains the relay and reviewer between stages.
Direct sub-agent-to-sub-agent chaining is not required.

This mode should be used deliberately.
It is valuable because it guarantees multiple passes through the work, not
because it makes the workflow autonomous.

Unless the user asks for this fixed pipeline, the default guidance still
applies: work locally by default and use sub-agents selectively for bounded
background tasks.

---

## Main Agent Local Work

The main agent may do local work without delegation when that work is itself the clearest path forward or when it primarily supports orchestration.

Examples include:

* reading a small number of files to establish context
* checking project instructions and durable planning anchors
* performing direct work in a narrow active code slice
* inspecting diffs, status, logs, or test summaries
* performing integration review across sub-agent outputs
* making small glue changes needed to connect accepted work

The main agent may also keep work local when the task is so small that delegation overhead would exceed the benefit.

---

## Ownership And Write Scope

Each sub-agent must receive an explicit scope boundary.

That boundary should define:

* the question or task being handled
* which files may be changed, if any
* which files are read-only context
* which validations the sub-agent should run

Parallel sub-agents must not share write ownership for the same file set.

If two sub-agents need the same file, only one may be assigned write permission.
The others must treat that file as read-only.

The main agent is responsible for preventing overlapping edits and for resolving integration order when outputs depend on each other.

---

## When To Wait

The main agent should wait for sub-agent results before proceeding when:

* a downstream decision depends on the result
* integration would be speculative without the answer
* two branches of work may conflict
* the sub-agent owns the only authorized write scope for the next change

Waiting is appropriate when continuing locally would likely create rework or contradictory edits.
If the main agent still has useful interactive or planning work available, it should usually do that instead of waiting immediately.

---

## When To Continue Orchestrating

The main agent may continue local orchestration while sub-agents run when the remaining work is independent.

Examples include:

* preparing the next delegation packet
* reviewing adjacent documentation or specifications
* planning integration order
* inspecting unaffected parts of the system
* preparing verification steps that do not depend on unfinished edits
* answering new user questions while background work continues
* handling separate interactive tasks that do not conflict with the delegated slice

The main agent should not stay idle when useful non-conflicting orchestration work remains.

---

## Review And Integration

Sub-agent output must be reviewed before it is trusted.

The main agent must:

* inspect the returned reasoning, diffs, or artifacts
* verify that the work matches the assigned scope
* check for conflicts with existing local changes
* run or confirm appropriate validation
* integrate only the parts that actually satisfy the request

Sub-agent output is input to decision-making, not a substitute for review.

The main agent must reject, revise, or redo work that is incomplete, unsafe, or inconsistent with repository rules.

---

## Fallback To Local Action

The main agent may choose local execution instead of delegation when delegation would create more risk than value.

Common reasons include:

* the task is tiny and easily verified locally
* the work requires tightly coupled edits across the same files
* the context is too subtle to split safely
* the delegated task would still leave the main agent blocked on the same immediate result
* the repository state is already volatile or crowded with in-flight edits
* the cost of mis-integration is higher than the speed gained by parallelism

When delegation is skipped for these reasons, the main agent should continue the work locally and keep the scope small and explicit.

---

## Guiding Principle

Work locally by default.

Delegate bounded background work when that helps the main agent stay responsive, keep ownership explicit, and leave final judgment with the main agent.
