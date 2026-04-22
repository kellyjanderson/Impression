# Loft Spec 51: Canonical Station-Slice Silhouette Classification (v1.0)

## Overview

This specification defines the station-local silhouette-comparison lane used to
verify loft correspondence output without relying on brittle world-space
heuristics.

## Backlink

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- canonical expected, actual, and diff station-slice artifacts for
  representative loft fixtures
- silhouette relationship classification for station-local comparisons
- fixture policy for whether rotation-only mismatch is a failure
- stable synthetic classifier fixtures that prove the comparison classes remain
  durable

## Behavior

This leaf must define:

- how representative loft fixtures recover expected and actual station slices in
  a shared local frame
- how the comparison contract classifies `same_shape_same_orientation`,
  `same_shape_rotated`, and `different_shape`
- how fixture policy records whether orientation is required or rotation-only
  drift is acceptable
- how expected, actual, and diff section images participate in the dirty/clean
  reference-artifact lifecycle

## Constraints

- station comparison must stay local to the chosen fixture station instead of
  depending on global world-space alignment heuristics
- silhouette comparison must tolerate translation and scale differences that do
  not change the underlying section shape
- orientation mismatch must remain distinguishable from genuine contour drift
- fixtures must not claim orientation-sensitive truth unless their section shape
  is capable of distinguishing rotation from equivalence

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- representative loft fixtures emit expected, actual, and diff station-slice
  artifacts in a canonical comparison frame
- the silhouette classifier reports the three durable relationship classes
- fixture policy can accept or reject rotation-only mismatch explicitly
- synthetic comparison fixtures prove the classifier distinguishes scale/offset
  equivalence, rotated equivalence, and genuinely different silhouettes
