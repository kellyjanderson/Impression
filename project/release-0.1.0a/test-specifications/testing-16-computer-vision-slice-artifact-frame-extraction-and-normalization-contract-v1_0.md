# Testing Spec 16 Test: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract

## Overview

This test specification defines verification for local-frame slice extraction
and normalization rules.

## Backlink

- [Testing Spec 16: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract (v1.0)](../specifications/testing-16-computer-vision-slice-artifact-frame-extraction-and-normalization-contract-v1_0.md)

## Automated Smoke Tests

- representative fixtures emit expected, actual, and diff slice artifacts in
  one declared local frame
- expected silhouette sources are consumable by the harness

## Automated Acceptance Tests

- allowed normalization removes declared framing noise without hiding real
  contour drift
- missing local-frame declarations fail clearly
- grouped slice artifacts remain intact as one completeness contract
- extraction and normalization remain reproducible across repeated runs
