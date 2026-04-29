# Loft Spec 20 Test: Interactive Branch Picking API

## Overview

This test specification defines how explicit caller-selected ambiguity
resolution should be verified.

## Backlink

- [Loft Spec 20: Interactive Branch Picking API (v1.0)](../specifications/loft-20-interactive-branch-picking-v1_0.md)

## Manual Smoke Check

- Request an ambiguity report for a known ambiguous loft fixture.
- Select a candidate by id and rerun planning/execution.
- Confirm the selected branch is honored exactly and replay is stable.

## Automated Smoke Tests

- ambiguous fixtures return a non-empty ambiguity report
- valid selection maps pass planner validation
- invalid interval ids or candidate ids fail with structured errors

## Automated Acceptance Tests

- required interactive mode rejects incomplete selections
- best-effort mode uses deterministic fallback for missing selections
- same selection map reproduces identical plan and output
- selected candidate ids are stable across repeated runs for identical inputs
- executor output remains valid and watertight for accepted selections

## Notes

- Include one headless JSON-driven selection-map test path.
