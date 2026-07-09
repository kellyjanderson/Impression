# 0.1.0.a Example Plans

This folder holds detailed example-planning documents for the `0.1.0.a`
release story.

These are not implementation specs. They are model-design plans that should be
used when we later build the real examples, documentation scenes, and preview
artifacts.

The intent is to avoid generic AI-shaped lofts by deciding ahead of time:

- what the object should visibly be
- what geometric features make it believable
- which stations matter
- which Impression calls we expect to use
- which `0.1.0.a` features the example must exercise

## Execution Posture

When these examples move from planning into implementation:

- model with `impression.modeling`
- prefer surfaced `Loft(...)` unless a convenience lane is the clearest fit
- keep the result app-owned until the preview or export boundary
- render intermediate visual checkpoints
- use a sub-agent visual review loop against those checkpoints before
  finalizing the shape

That visual review loop should check:

- silhouette quality
- whether the object reads as the intended real-world thing
- whether orientation and section rhythm remain believable
- whether the shape is demonstrating the intended release delta rather than
  merely existing

## Planned Examples

- [01 Airframe Shell From Sparse Former Stations](01-airframe-shell-from-sparse-former-stations-plan.md)
- [02 Tight-Shoulder Bottle Without Rail Explosion](02-tight-shoulder-bottle-without-rail-explosion-plan.md)
- [03 Orientation-Safe Motor Mount Or Fairing Spine](03-orientation-safe-motor-mount-or-fairing-spine-plan.md)
- [04 Contour Stack Simplifier For Reverse-Engineered Parts](04-contour-stack-simplifier-for-reverse-engineered-parts-plan.md)
- [05 Loft Triage Dashboard For Uncertain Or Failed Inference](05-loft-triage-dashboard-for-uncertain-or-failed-inference-plan.md)
