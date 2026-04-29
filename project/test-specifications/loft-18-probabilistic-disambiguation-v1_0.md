# Loft Spec 18 Test: Probabilistic Ambiguity Disambiguation

## Overview

This test specification defines how the optional probabilistic loft ambiguity
path should be verified.

## Backlink

- [Loft Spec 18: Probabilistic Ambiguity Disambiguation (v1.0)](../specifications/loft-18-probabilistic-disambiguation-v1_0.md)

## Manual Smoke Check

- Run an ambiguous loft fixture with probabilistic mode enabled.
- Repeat once with the same seed and once with a different seed.
- Confirm same-seed replay is identical, different-seed replay may vary, and
  all accepted outputs remain valid.

## Automated Smoke Tests

- planner accepts probabilistic controls and records them in plan metadata
- same input plus same seed produces identical selected plan
- low-confidence path triggers configured fallback or failure behavior

## Automated Acceptance Tests

- reproduce identical plan/mesh for identical seed and controls
- allow differing valid candidate selection for differing seeds
- fail explicitly when no valid candidate survives pre-check filtering
- preserve deterministic default behavior when probabilistic mode is disabled
- assert selected outputs satisfy mesh-quality gates

## Notes

- Use durable ambiguous fixtures with at least one residual many-to-many case.
- Prefer plan-level equality plus mesh-quality analysis for replay checks.
