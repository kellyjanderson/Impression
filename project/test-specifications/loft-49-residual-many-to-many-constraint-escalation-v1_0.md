# Loft Spec 49 Test: Residual Many-to-Many Constraint Escalation

## Overview

This test specification defines verification for residual many-to-many
constraint escalation in next-generation loft planning.

## Backlink

- [Loft Spec 49: Residual Many-to-Many Constraint Escalation (v1.0)](../specifications/loft-49-residual-many-to-many-constraint-escalation-v1_0.md)

## Automated Smoke Tests

- blocked many-to-many planning raises structured ambiguity and constraint records

## Automated Acceptance Tests

- residual ambiguity identifies the topology state that holds the ambiguity
- ambiguous region indices in that topology are surfaced
- relationship-group context is included for many-to-many region ambiguity
