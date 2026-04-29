# Testing Spec 23 Test: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules

## Overview

This test specification defines verification for the honesty boundary of
diagnostic panel artifacts and the narrow cases where proof use may be
delegated explicitly.

## Backlink

- [Testing Spec 23: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules (v1.0)](../specifications/testing-23-computer-vision-diagnostic-panel-honesty-boundary-and-proof-delegation-rules-v1_0.md)

## Automated Smoke Tests

- representative diagnostic artifacts declare their default diagnostic-only
  posture
- explicitly delegated proof uses remain distinguishable from default behavior

## Automated Acceptance Tests

- independently rendered panels do not silently imply shared-scale proof
- diagnostic panels support review without replacing authoritative proof lanes
- explicit proof delegation remains narrow and visible in the contract
- grouped panel artifacts remain honest about what they can and cannot prove
