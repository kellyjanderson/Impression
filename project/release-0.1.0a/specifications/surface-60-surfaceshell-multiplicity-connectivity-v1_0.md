# Surface Spec 60: SurfaceShell Multiplicity and Connectivity Policy (v1.0)

## Overview

This specification defines how many shells a body may contain and whether a
single shell may represent disconnected components.

## Backlink

Parent specification:

- [Surface Spec 20: Surface Body and Shell Ownership Rules (v1.0)](surface-20-surface-body-shell-ownership-rules-v1_0.md)

## Scope

This specification covers:

- body shell multiplicity
- shell connectivity requirements
- disconnected-component policy

## Behavior

This branch must define:

- whether bodies may contain multiple shells
- whether a shell is required to be connected
- how disconnected geometry is represented when allowed

## Constraints

- multiplicity rules must be explicit
- connectivity rules must be deterministic
- disconnected representation must not be left to caller interpretation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- body shell multiplicity rules are explicit
- shell connectivity rules are explicit
- disconnected-component handling is explicit

