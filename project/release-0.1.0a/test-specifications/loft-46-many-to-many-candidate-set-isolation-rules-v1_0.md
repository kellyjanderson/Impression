# Loft Spec 46 Test: Many-to-Many Candidate-Set Isolation Rules

## Overview

This test specification defines verification for many-to-many candidate-set
isolation in next-generation loft planning.

## Backlink

- [Loft Spec 46: Many-to-Many Candidate-Set Isolation Rules (v1.0)](../specifications/loft-46-many-to-many-candidate-set-isolation-rules-v1_0.md)

## Automated Smoke Tests

- many-to-many transitions expose explicit candidate-set records

## Automated Acceptance Tests

- candidate sets cover all actual source/target regions in the isolated subset
- matched and residual region indices are separated deterministically
- related many-to-many subsets do not use a separate ambiguity regime
