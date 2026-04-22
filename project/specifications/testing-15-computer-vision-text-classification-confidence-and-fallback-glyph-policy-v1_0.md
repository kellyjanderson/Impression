# Testing Spec 15: Computer Vision Text Classification, Confidence, and Fallback-Glyph Policy (v1.0)

## Overview

This specification defines the semantic classification, confidence posture, and
fallback-glyph failure policy for the text CV lane.

## Backlink

- [Testing Spec 03: Computer Vision Text OCR and Glyph Verification (v1.0)](testing-03-computer-vision-text-ocr-and-glyph-verification-v1_0.md)

## Scope

This specification covers:

- same text, rotated text, mirrored text, different text, and unreadable
  classes
- confidence thresholds and unreadable posture
- readable-orientation policy
- fallback glyph and `.notdef` failure treatment

## Behavior

This leaf must define:

- the durable text result classes
- when low-confidence output becomes unreadable instead of a pass
- how fixtures declare whether orientation matters for text
- how fallback glyph output is classified as a semantic failure

## Constraints

- unreadable output must remain explicit rather than guessed into a pass
- orientation-sensitive text fixtures must declare readable orientation policy
- fallback glyph output must not pass as valid text meaning

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- text result classes are explicit
- confidence and unreadable posture are explicit
- fallback-glyph failure posture is explicit
- verification requirements are defined by its paired test specification
