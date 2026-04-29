# Testing Spec 05 Test: Computer Vision Camera and Framing Contract Compliance

## Overview

This test specification defines verification for the canonical camera and
framing contract used by downstream view-space CV lanes.

## Backlink

- [Testing Spec 05: Computer Vision Camera and Framing Contract Compliance (v1.0)](../specifications/testing-05-computer-vision-camera-and-framing-contract-compliance-v1_0.md)

## Automated Smoke Tests

- representative fixtures can emit deterministic renders under the declared
  camera contract
- camera fields such as pose, target, up vector, and projection mode are
  consumable by the harness

## Automated Acceptance Tests

- pose, up-vector, projection, and visible-extent drift fail as camera-contract
  violations
- framing drift is surfaced before downstream object-view interpretation runs
- camera/framing failures stay distinguishable from semantic object mismatches

