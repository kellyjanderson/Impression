# Surface Spec 87: Geometry Baking Triggers and Required Cases (v1.0)

## Overview

This specification defines when attached transforms must be baked into concrete
surface geometry.

## Backlink

Parent specification:

- [Surface Spec 29: Transform Attachment Versus Baked Geometry Policy (v1.0)](surface-29-transform-attachment-vs-baked-policy-v1_0.md)

## Scope

This specification covers:

- baking triggers
- mandatory baking cases
- prohibited eager-baking cases

## Behavior

This branch must define:

- what events or consumers trigger baking
- which operations require baked geometry
- which situations explicitly must not force baking

## Constraints

- baking triggers must be explicit
- mandatory baking cases must be bounded
- eager baking must not occur implicitly outside documented triggers

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- baking triggers are explicit
- required baking cases are explicit
- prohibited implicit baking cases are explicit

