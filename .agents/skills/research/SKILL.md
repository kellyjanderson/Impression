---
name: research
description: Store durable project-local research findings so future architectural, specification, planning, implementation, and related project-process work can reuse them.
---

# Research

Research documents preserve useful information that would otherwise fall out of working context.

## Purpose

Use research documents to keep:

* tool or library findings
* external constraints
* experimental results
* distilled reference knowledge
* implementation-facing notes that are not yet architecture or specification

Research is durable working information for the project.

It exists so that:

* researched information remains available after the current context is gone
* future project work can build on prior research
* agents can use project-local reference material instead of repeatedly rediscovering the same information

## Location

Research should live under:

```text
project/research/
```

Use subfolders when that helps organize topic areas.

## Recommended Structure

Research documents should usually include:

* topic
* findings
* implications
* references

## Relationship To Other Documents

Research supports:

* architectural refinement
* specification writing
* planning
* implementation
* related project-local skill and process work

If research becomes part of the defined system, reference it from the relevant architecture or specification documents.
