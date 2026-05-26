# Testing Spec 22: Computer Vision Diagnostic Panel Layout, Ordering, and Region Extraction Contract (v1.0)

## Overview

This specification defines the deterministic mechanics for diagnostic multi-panel
artifacts such as triptychs and expected/actual/diff groups.

## Backlink

- [Testing Spec 08: Computer Vision Diagnostic Triptych and Panel-Region Presentation (v1.0)](testing-08-computer-vision-diagnostic-triptych-and-panel-region-presentation-v1_0.md)

## Scope

This specification covers:

- deterministic panel ordering and labeling
- panel-region extraction and addressing
- reproducible cropping and layout
- grouped review artifact publication

## Behavior

This leaf must define:

- the panel order and label contract for grouped diagnostic artifacts
- how regions are extracted or addressed reproducibly
- how grouped panels remain interpretable during failure review
- which layout mechanics are shared across triptych-style artifacts

## Constraints

- panel layout and extraction must remain deterministic
- grouped panels must remain human-readable
- layout mechanics must stay separate from proof-lane semantics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- deterministic panel mechanics are explicit
- extraction and addressing rules are explicit
- grouped review artifact expectations are explicit
- verification requirements are defined by its paired test specification
