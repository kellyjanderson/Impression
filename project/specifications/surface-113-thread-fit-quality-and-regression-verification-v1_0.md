# Surface Spec 113: Thread Fit, Quality, and Regression Verification (v1.0)

## Overview

This specification defines fit/quality behavior and regression evidence for surfaced threading.

## Backlink

- [Surface Spec 103: Surface-Native Threading Replacement (v1.0)](surface-103-surface-native-threading-replacement-v1_0.md)

## Scope

This specification covers:

- fit presets and compensation behavior
- quality and sampling controls that affect surfaced threads
- regression fixtures and reference artifacts for surfaced threading

## Behavior

This branch must define:

- which fit and quality knobs remain public and deterministic
- how those knobs affect surfaced output without changing authored truth
- what reference fixtures are required before surfaced threading is complete

## Constraints

- fit compensation must remain explicit and testable
- quality controls must not silently change canonical thread meaning
- surfaced threading completion requires durable docs and references

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- fit and quality behavior is explicit for surfaced threading
- required regression evidence is explicit
- verification requirements are defined by its paired test specification

