# Loft Spec 33 Test: Branch and Closure Record Contract

## Overview

This test specification defines verification for branch-order and closure
ownership records in the loft plan.

## Backlink

- [Loft Spec 33: Branch and Closure Record Contract (v1.0)](../specifications/loft-33-branch-and-closure-record-contract-v1_0.md)

## Automated Smoke Tests

- branch ordering is explicit per interval
- loop and region closure ownership are explicit

## Automated Acceptance Tests

- duplicate closure ownership is rejected
- invalid branch ordering is rejected
- executor relies on explicit closure records rather than inferred geometry
