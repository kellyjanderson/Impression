---
name: specification-sizing
description: Apply Implementation Work Units to judge whether a specification leaf is still too large or bundled to be marked final.
---

# Specification Sizing

Use this Skill when deciding whether a specification is final or still needs child specifications.

## Purpose

Implementation Work Units (IWU) are a structural sizing tool.

IWU does not estimate hours.
It answers:

* does this specification still bundle too many concerns?
* is this realistically one clean implementation round?
* should this be refined further before implementation?

## Unit Definition

One IWU is one independently deliverable, reviewable change set with its own verification surface.

The unit is intentionally abstract so it can size software, documentation, tooling, service, research, design, and process projects consistently.

## Counting Rule

Count only concerns explicitly present in the specification text.
Do not score imagined future work that is not written.

Count 1 IWU when the work has:

* one primary outcome
* one coherent responsibility boundary
* one reviewable artifact or change set
* one explicit verification method
* declared inputs and outputs
* explicitly named unresolved assumptions or decisions

Increase the IWU count, or refine into child specifications, when any measure becomes plural, ambiguous, or unnamed.

## Evaluate At Least These Areas

When sizing a specification, count and discuss:

* distinct implementation concerns
* new or changed durable contracts
* major workflow or UI surfaces touched
* cross-domain coupling
* migration or compatibility burden
* testing and verification shape

## Annotation Rule

Every specification document should include a `## Work Units` section near the top of the document, immediately after the title/date block when present.

Use this format:

```markdown
## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: N IWU.
Basis: short reason for the count.
```

## Decision Rule

If an intended final leaf scores above 1 IWU, refine it into child specifications before marking it final.

If an intended final leaf scores 1 IWU and the qualitative review still shows one cohesive implementation round, it may remain final.

Branch and parent specifications may score above 1 IWU, but their count is a rollup over descendant implementation leaves and must not be added to descendant counts when reporting totals.

## Reporting Rule

At the end of a refinement pass, report:

* total IWU for implementation leaves
* total IWU for verification or test-specification leaves
* branch rollups separately from leaf totals to avoid double counting
* how many branches still need another refinement round
