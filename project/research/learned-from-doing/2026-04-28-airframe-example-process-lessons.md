# 2026-04-28 Airframe Example Process Lessons

## Purpose

This note records what was learned by actually trying to execute the first
`0.1.0.a` airframe example.

The goal is to preserve observed lessons from the work itself so we can later
compose better modeling directives, model-plan expectations, and modeling-spec
rules.

## Context

The attempted example was meant to show:

- believable aircraft-derived form
- sparse authored loft structure
- dense evidence and fitted interpretation
- reduced progression / retained station logic
- a visible reason the example belonged to `0.1.0.a` rather than `0.0.3a2`

The execution surfaced several failures and several useful process discoveries.

## Learned 01: Orientation Language Must Be Declared Up Front

One of the first real execution failures was orientation drift.

The model spine, canonical camera assumptions, and human review language were
not aligned tightly enough at the start. That made it too easy to misread what
"front", "side", "top", and "length axis" were actually showing.

What this taught:

- orientation language cannot be left implicit
- the modeling process needs a declared orientation contract before serious
  review starts
- preview and review tools may silently inherit orientation assumptions that
  can distort critique if the model is not aligned to them

Practical implication:

- model plans should declare orientation
- model execution should verify that rendered review views really correspond to
  the intended object language

### Question Raised By This Lesson

How do we ensure that every layer is speaking the same orientation language?

More specifically:

- how is handedness communicated?
- how are axis directions communicated?
- how do preview, rendering, sub-agent review, and human language stay aligned?
- how do we avoid ambiguous directional words such as `deep`, `shallow`,
  `forward`, or `up` being interpreted differently at different layers?

### Observed Deeper Problem

This points at a more systemic issue than one example failure.

We likely need a way to communicate handedness and directional syntax so every
layer is talking in the same orientation language.

That shared structure may need to be passed into:

- rendering steps
- preview applications
- visual review agents
- diagnostic generation
- any other layer that can inherit or reinterpret orientation

### Exploratory Structure

One possible structure looks like:

```text
handed: left | right
camera_direction: towards | away
language:
  z: height | tall
  y: depth
  zpositive: up
  znegative: down
  ynegative: back
  ypositive: forward
```

This is not complete yet, but it expresses the right kind of need: a portable
orientation contract rather than an assumed one.

### Ambiguity Discovered While Trying To Fill It Out

Trying to fill out even a rough structure exposed holes in the dimensional
language itself.

For example:

- `shallow` was tempting to use for both `ypositive` and `znegative`
- `deep` was tempting to use for both depth and negative-height style language

That means some real-world descriptive words are contextually useful for
humans, but not safe as canonical axis words.

### Local Versus Global Scope

Another issue surfaced while thinking this through:

- global or top-level orientation language
- local or drilled-in frame-of-reference language

An initial working idea is:

- all coordinates are communicated in global or top-level scope by default
- if the user drills into a scene or part and higher-level context is removed,
  the viewed frame may become the active local language for communication

This is still exploratory, but it suggests the process may need both:

- a global orientation contract
- a local override or local frame contract

### What This Suggests Next

This lesson does not yet resolve the problem, but it does point to a likely
next area of work:

- codify words that map to specific axis references
- separate safe canonical terms from context-rich but ambiguous human terms
- define how handedness, camera direction, and frame scope are passed along the
  chain

In other words: the modeling failure exposed a likely missing orientation
contract system, not just a one-off review mistake.

## Learned 02: Object-Class Read Must Be Tested Before Feature Plumbing Is Trusted

The first airframe result carried a substantial amount of inference scaffolding
and metadata, but the visible object still read as a generic lofted blob.

What this taught:

- inference plumbing is not evidence of a good example
- a model can be technically rich while failing the human-recognition test
- object-class believability needs to be an early gate, not a late polish step

Practical implication:

- the first serious review checkpoint should ask whether the object reads as the
  intended class at all
- if the answer is no, the work should not advance as if the feature
  demonstration is succeeding

## Learned 03: Feature Demonstration Must Be Visible, Not Merely Present

The attempted example computed curve intent, retained stations, reduced
progression, and related inference artifacts. But nothing in the visible output
made those new capabilities legible.

What this taught:

- metadata alone does not make an example a release proof point
- if a viewer cannot point to a visible artifact and say "that required the new
  version," then the example is not doing its job

Practical implication:

- examples for new modeling capability need visible proof lanes
- likely forms of proof include:
  - dense evidence views
  - retained station overlays
  - fitted curve overlays
  - comparison panels
  - explicit diagnostic composition

## Learned 04: A Whole Airframe Is Too Ambitious To Hide Inside One Loft Concept

The attempted example treated "airframe shell" as if it were naturally one loft
task. In practice, believable aircraft structure likely wants decomposition into
multiple parts or multiple operations.

What this taught:

- whole-object ambition hides modeling decomposition work
- "single example" should not be confused with "single operation"
- believable aircraft-like forms may require:
  - multiple lofts
  - root cues
  - secondary fairings
  - CSG combination
  - explicit non-loft support geometry

Practical implication:

- model plans should surface decomposition before execution
- modeling specs should own separate parts or proof lanes instead of letting
  the whole example collapse into one oversized body

## Learned 05: Sub-Agent Visual Critique Was Useful, But Only After The View Contract Was Correct

The visual-review loop with sub-agent critique was helpful, but its usefulness
depended on the views actually meaning what we thought they meant.

What this taught:

- visual sub-agents are valuable for object-class critique
- they are especially good at identifying "still a blob" failures
- but they are only trustworthy if the orientation/view contract is already
  correct

Practical implication:

- render-and-review should remain part of the process
- orientation verification should happen before relying on critique feedback

## Learned 06: Durable Reference Images And Iteration Logs Were Worth Keeping

The modeling work improved because references and iteration renders were kept
durably instead of being treated as ephemeral.

What this taught:

- durable references reduce memory drift
- iteration logs make it easier to understand why a form changed
- research should stay near the model so later planning and critique can reuse
  it

Practical implication:

- example folders should keep:
  - reference images
  - render history
  - research note
  - iteration log

## Learned 07: Preview Quality And Mesh-Quality Validation Are Different Gates

The airframe work also exposed that a model can start to look visually right
while still surfacing downstream tessellation quality issues such as degenerate
faces.

What this taught:

- visual success and geometric consumer health are separate gates
- a preview-looking-good checkpoint is not enough
- examples should also be checked for mesh or consumer-boundary health when
  that matters to the example

Practical implication:

- execution review should include both:
  - object-read review
  - consumer-boundary validation

## Learned 08: Comparison-First Example Design Is Likely Better For `0.1.0.a`

The attempted airframe shell tried to be a single beautiful object first and an
inference demonstration second. That ordering likely worked against the release.

What this taught:

- for inference-oriented release work, comparison-first example composition is
  probably stronger than "hero object only"
- the example may need to be designed around:
  - evidence
  - reduction
  - fitted explanation
  - visible contrast between old and new story

Practical implication:

- future `0.1.0.a` examples should probably be planned as multi-panel or
  multi-lane demonstrations from the start

## Summary

The most important lessons from doing the work were:

1. orientation language must be explicit early
2. object-class read must be validated before trusting feature plumbing
3. feature delta must be visible, not just present in metadata
4. whole-object ambition hides decomposition work
5. visual critique is valuable when the view contract is correct
6. durable references and iteration logs are worth the overhead
7. preview success and mesh-quality success are different gates
8. comparison-first example design is likely better for this release line

## Use

This note should be treated as durable research derived from execution.

If we later turn any of these lessons into directives, that should happen by
composing them intentionally into process documents rather than by assuming the
execution note itself is a directive.
