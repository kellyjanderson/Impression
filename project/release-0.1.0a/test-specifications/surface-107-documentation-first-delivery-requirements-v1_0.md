# Surface Spec 107 Test: Documentation-First Delivery Requirements

## Overview

This test specification defines verification for the rule that documentation is required for completion.

## Backlink

- [Surface Spec 107: Documentation-First Delivery Requirements (v1.0)](../specifications/surface-107-documentation-first-delivery-requirements-v1_0.md)

## Manual Smoke Check

- Review the shared agent guidance and project-specific agent guidance for documentation rules.
- Confirm that feature delivery branches point to durable docs rather than leaving behavior implied by chat history.

## Automated Smoke Tests

- required documentation guidance files exist in the expected locations
- specification guidance explicitly states that documentation is part of completion

## Automated Acceptance Tests

- project-specific reference-artifact rules exist under `project/agents/`
- shared documentation quality guidance exists under `agents/`
- new final feature leaves in the replacement branch have paired test specifications and explicit documentation expectations
