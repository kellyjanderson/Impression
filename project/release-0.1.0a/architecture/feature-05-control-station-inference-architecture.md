# Feature 05 — Control-Station Inference Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `5` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [B-Spline Control-Station Inference Architecture](b-spline-control-station-inference-architecture.md)
- [Priority 04 — Control-Station Inference Result Architecture](priority-04-control-station-inference-result-architecture.md)
- [Feature 04 — Non-User-Facing Control Stations Architecture](feature-04-non-user-facing-control-stations-architecture.md)

## Purpose

Define the feature branch that reduces dense linear station sets into a durable,
explainable reduced progression while preserving topology truth.

## Included Scope

- retained topology-station classification
- hidden retained control-station classification
- reduction refusal when structure would be lost
- structural preservation reporting
- replayable reduced progression result

## Core Rule

This feature is not “drop stations until it looks okay.”

It must preserve the distinction between:

- hard structural anchors
- shape-control anchors

and expose the reasons for retention or refusal.

## Output Contract

The expected durable result is:

- reduced progression
- retained topology stations
- retained hidden control stations
- provenance metadata
- structural preservation report
- supporting fit diagnostics

## Architectural Conclusion

Feature `05` is the product-level inference branch that turns the lower-level
curve-fitting work into a compact, trustworthy loft representation.
