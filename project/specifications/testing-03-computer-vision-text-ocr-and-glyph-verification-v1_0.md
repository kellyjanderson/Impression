# Testing Spec 03: Computer Vision Text OCR and Glyph Verification (v1.0)

## Overview

This specification defines the parent branch for architecture-level OCR and
glyph-interpretation tooling used to verify rendered text by visible glyph
meaning rather than by font-specific geometry assumptions.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- canonical text render or crop products for OCR-driven verification
- the OCR or glyph-interpretation tooling contract for those products
- visible text recognition and confidence policy
- orientation and mirror classification for rendered text
- fallback glyph and `.notdef` detection posture

## Behavior

This branch must define:

- the canonical artifact set used to interpret rendered text
- the tooling boundary between text render/crop production and OCR or
  glyph-interpretation consumption
- the durable result classes for text meaning, including same text, rotated
  text, mirrored text, different text, and unreadable text
- fixture policy for intended string, readable orientation expectations, and
  confidence thresholds
- how fallback glyph output is treated as a semantic failure rather than a
  geometry success

## Constraints

- this lane verifies user-visible text meaning, not aesthetic font fidelity
- low-confidence recognition must not silently pass
- fixtures must declare the intended readable orientation when orientation is
  supposed to matter
- unreadable results must remain explicit rather than guessed into a pass class
- this leaf defines verification tooling and facilitation rules rather than
  feature-level text modeling behavior

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 14: Computer Vision Text Canonical Artifact Set and Initial OCR Scope](testing-14-computer-vision-text-canonical-artifact-set-and-initial-ocr-scope-v1_0.md)
- [Testing Spec 15: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy](testing-15-computer-vision-text-classification-confidence-and-fallback-glyph-policy-v1_0.md)

## Acceptance

This specification is complete when:

- initial executable scope is separated from text result-taxonomy and fallback
  policy
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
