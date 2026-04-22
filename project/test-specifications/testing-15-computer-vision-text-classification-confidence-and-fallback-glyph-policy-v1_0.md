# Testing Spec 15 Test: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy

## Overview

This test specification defines verification for text semantic classes,
confidence posture, and fallback-glyph failure handling.

## Backlink

- [Testing Spec 15: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy (v1.0)](../specifications/testing-15-computer-vision-text-classification-confidence-and-fallback-glyph-policy-v1_0.md)

## Automated Smoke Tests

- representative text outputs classify into one of the supported text result
  classes
- low-confidence cases surface an explicit unreadable outcome

## Automated Acceptance Tests

- correct visible text classifies as same-text under the declared orientation
  policy
- rotated or mirrored text is distinguishable when the fixture requires it
- fallback glyph or `.notdef` output is treated as a semantic failure
- materially different text classifies as different-text
