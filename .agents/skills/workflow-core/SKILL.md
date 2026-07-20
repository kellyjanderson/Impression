---
name: workflow-core
description: Follow the shared workflow from exploration through stabilization into specification refinement and implementation, with research, release definitions, and test specifications as durable support layers.
---

# Workflow Core

This Skill defines the shared project workflow from product definition to implementation.

## Overview

Work progresses through three phases:

1. exploration
2. stabilization
3. specification refinement and implementation

Research supports all phases.
Release definitions provide version-level cohesion.
Test specifications provide durable verification contracts.

## Exploration Loops

Use explicit feedback loops between:

* product and research
* UI definitions and research
* architecture and research
* release definitions and stabilized project branches

Exploration continues until the relevant branch is clear enough to guide durable downstream work.

## Stabilization

Before implementation-heavy work begins, the relevant product, UI, and architecture branches must be stable enough to guide execution without constant structural churn.

For architecture, stability means the relevant branch has been completed breadth first:

* major parts are identified
* relationships are described
* high-level data flow is described
* cross-domain solutions are resolved

## Specification Refinement Loop

After stabilization:

* break architecture or UI definitions into specifications
* score each specification with Implementation Work Units (IWU)
* annotate each specification near the top with its IWU count and basis
* refine until leaves are implementation-sized
* create paired test specifications for feature leaves as they become final

Implementation should not be used as a substitute for unfinished architecture or unfinished UI definition work.

During refinement, use IWU scoring as the standard signal for whether a branch needs another specification pass:

* intended final leaves should score 1 IWU
* branches may score above 1 IWU as rollups over descendant leaves
* split a specification when the IWU measures are plural, ambiguous, or unnamed
* report branch rollups separately from leaf totals to avoid double counting

## Path Rule

Implementation work must not begin without a durable planning anchor.

Workspace overlays may define the allowed local paths and anchor variants.
