# Testing Spec 11 Test: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules

## Overview

This test specification defines verification for grouped model-output artifact
completeness and first-run bootstrap behavior.

## Backlink

- [Testing Spec 11: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules (v1.0)](../specifications/testing-11-reference-artifact-grouped-model-output-completeness-and-bootstrap-rules-v1_0.md)

## Automated Smoke Tests

- new named fixtures bootstrap the full declared dirty artifact group
- representative model-output fixtures emit both image and STL by default

## Automated Acceptance Tests

- missing only one artifact from an existing fixture fails clearly rather than
  bootstrapping silently
- documented exceptions can omit an artifact type without failing the contract
- grouped completeness remains explicit when optional section or diagnostic
  artifacts are present
- bootstrap does not silently promote any artifact to clean
