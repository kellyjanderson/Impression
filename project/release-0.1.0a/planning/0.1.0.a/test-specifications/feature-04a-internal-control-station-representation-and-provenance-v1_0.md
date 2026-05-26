# Feature Spec 04A Test: Internal Control-Station Representation and Provenance

## Overview

This test specification defines verification for hidden control-station records
and provenance.

## Backlink

- [Feature Spec 04A: Internal Control-Station Representation and Provenance (v1.0)](../specifications/feature-04a-internal-control-station-representation-and-provenance-v1_0.md)

## Automated Smoke Tests

- hidden control-station records remain distinct from topology-station records
- provenance metadata is carried with retained hidden control stations

## Automated Acceptance Tests

- internal representation remains planner-owned and non-user-facing
- provenance remains durable enough for later diagnostics
- hidden control stations do not collapse into stealth public authored inputs
