---
name: delegation
description: Use bounded, explicitly owned delegation in the Impression workspace while keeping the main agent responsible for orchestration, review, and final judgment.
---

# Delegation

Use this Skill for multi-agent work in the Impression workspace.

## Core Model

The main agent is the orchestrator and user-facing coordinator.

It owns:

* understanding the request
* choosing the execution plan
* assigning bounded work
* reviewing returned results
* integrating accepted changes
* reporting final outcomes and risk

Sub-agents are execution units, not final decision makers.

## Default Path

Work locally by default.

Delegate only when the task has:

* a clear objective
* limited context needs
* a defined read or write scope
* a recognizable completion condition

## Ownership Rule

Every sub-agent must receive an explicit scope boundary:

* the question or task
* writable files, if any
* read-only context
* required validations

Parallel sub-agents must not share write ownership.

## Waiting And Review

Wait when downstream work depends on a sub-agent result.

Otherwise keep doing non-conflicting orchestration work.

Review every delegated result before trusting it.

Sub-agent output is input to decision-making, not a substitute for review.
