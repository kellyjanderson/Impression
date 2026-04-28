# 0.1.0.a Planning

This folder is the isolated planning space for the next version planning pass.

It is intentionally higher-level than active feature implementation work. The
purpose is to create the version-scoped containers where future architecture and
specification branches can be defined without mixing them into unrelated active
planning work.

## Structure

- [architecture/](architecture/README.md)
- [specifications/](specifications/README.md)
- [test-specifications/](test-specifications/README.md)
- [Feature List](feature-list.md)
- [Progression](progression.md)

Initial architecture set:

- [B-Spline Implementation Architecture](architecture/b-spline-implementation-architecture.md)
- [B-Spline Path And Trajectory Architecture](architecture/b-spline-path-and-trajectory-architecture.md)
- [B-Spline Control-Station Inference Architecture](architecture/b-spline-control-station-inference-architecture.md)
- [B-Spline Surface And Reconstruction Architecture](architecture/b-spline-surface-and-reconstruction-architecture.md)

Feature-driven architecture set from the `0.1.0.a` feature list:

- [Feature 01 — B-Spline Curve Support Architecture](architecture/feature-01-b-spline-curve-support-architecture.md)
- [Feature 02 — Explicit Fit Policy And Diagnostics Architecture](architecture/feature-02-explicit-fit-policy-and-diagnostics-architecture.md)
- [Feature 03 — Curve Fitting From Dense Loft Evidence Architecture](architecture/feature-03-curve-fitting-from-dense-loft-evidence-architecture.md)
- [Feature 04 — Non-User-Facing Control Stations Architecture](architecture/feature-04-non-user-facing-control-stations-architecture.md)
- [Feature 05 — Control-Station Inference Architecture](architecture/feature-05-control-station-inference-architecture.md)
- [Feature 06 — Curve-Intent Inference Architecture](architecture/feature-06-curve-intent-inference-architecture.md)
- [Feature 07 — Shared Trajectory Inference And Guidance Architecture](architecture/feature-07-shared-trajectory-inference-and-guidance-architecture.md)
- [Feature 08 — Progression Model Upgrade Architecture](architecture/feature-08-progression-model-upgrade-architecture.md)
- [Feature 09 — Inference Diagnostics And Explainability Architecture](architecture/feature-09-inference-diagnostics-and-explainability-architecture.md)

Priority-driven architecture expansion from the low-level construct gap report:

- [Priority 01 — B-Spline Curve Constructs Architecture](architecture/priority-01-b-spline-curve-constructs-architecture.md)
- [Priority 02 — Parameterization, Knot, And Fit Policy Architecture](architecture/priority-02-parameterization-knot-and-fit-policy-architecture.md)
- [Priority 03 — Path And Trajectory Integration Architecture](architecture/priority-03-path-and-trajectory-integration-architecture.md)
- [Priority 04 — Control-Station Inference Result Architecture](architecture/priority-04-control-station-inference-result-architecture.md)
- [Priority 05 — Spanwise Grouping And Compatibility Architecture](architecture/priority-05-spanwise-grouping-and-compatibility-architecture.md)
- [Priority 06 — Reconstruction And Repair Intermediates Architecture](architecture/priority-06-reconstruction-and-repair-intermediates-architecture.md)
- [Priority 07 — Surfaced B-Spline Patch Family Architecture](architecture/priority-07-surfaced-b-spline-patch-family-architecture.md)

## Purpose

Use this folder to:

- define version-scoped architecture branches
- define version-scoped specification branches
- keep `0.1.0.a` planning isolated from earlier planning work

This folder is itself the durable planning record for the planning-structure
path that created it.

Implementation sequencing for the release is tracked in:

- [progression.md](progression.md)

That progression orders only final leaf specifications and groups them into
five-spec implementation tranches so each delivery prompt can target one bounded
set at a time.
