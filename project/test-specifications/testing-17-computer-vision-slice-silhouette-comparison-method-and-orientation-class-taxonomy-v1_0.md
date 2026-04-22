# Testing Spec 17 Test: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy

## Overview

This test specification defines verification for the initial slice comparison
method and its orientation-sensitive classes.

## Backlink

- [Testing Spec 17: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy (v1.0)](../specifications/testing-17-computer-vision-slice-silhouette-comparison-method-and-orientation-class-taxonomy-v1_0.md)

## Automated Smoke Tests

- the classifier returns one of the supported slice relationship classes
- representative witness fixtures demonstrate the orientation-sensitive lane

## Automated Acceptance Tests

- equivalent silhouettes classify as same-shape
- rotated-orientation cases classify distinctly when the fixture has an
  adequate witness and requires orientation
- materially different silhouettes classify as different-shape
- symmetric fixtures cannot claim orientation-sensitive proof
