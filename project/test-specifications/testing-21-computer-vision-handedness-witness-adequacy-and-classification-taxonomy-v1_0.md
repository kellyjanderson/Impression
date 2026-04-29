# Testing Spec 21 Test: Computer Vision Handedness Witness Adequacy and Classification Taxonomy

## Overview

This test specification defines verification for witness adequacy and result
classification in the handedness lane.

## Backlink

- [Testing Spec 21: Computer Vision Handedness Witness Adequacy and Classification Taxonomy (v1.0)](../specifications/testing-21-computer-vision-handedness-witness-adequacy-and-classification-taxonomy-v1_0.md)

## Automated Smoke Tests

- representative asymmetric witness fixtures classify into one of the supported
  handedness result classes
- insufficient-witness fixtures surface an explicit unknown result

## Automated Acceptance Tests

- preserved-handedness fixtures classify as same-handedness when the witness is
  adequate
- mirrored fixtures classify as mirrored
- symmetric fixtures cannot claim handedness proof
- witness features must remain visible in the artifact products used by the
  lane
