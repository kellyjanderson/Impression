# Fix 04 Test: Coplanar Loft-Body Union Outcome

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/csg-coincident-contact-architecture.md`

## Overview

Verify that coplanar union returns a complete body or a typed refusal and never loses an operand.

## Application Integration Under Test

- App type: library-only.
- User/caller surface: model code calling `boolean_union(...)`.
- Invocation route: public union -> surface result -> witness/validity gate.
- Wiring owner/module: `src/impression/modeling/csg.py`.
- Observable result: valid complete union or typed refusal.
- Integration validation: minimal and full enclosure fixtures in both operand orders.

## Backlink

[Fix 04 specification](../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)

## Manual Smoke

Replace the grouped-body workaround in the test enclosure with its union and
confirm either one complete body is shown or the declared refusal is displayed.

## Automated Smoke Tests

Union a minimal box and lofted body sharing a coplanar face; assert success has
witness points from both operands, otherwise assert the typed coplanar refusal.

## Automated Acceptance Tests

- Run the exact enclosure composition in both operand orders.
- On success assert bounds, occupied witness regions, watertightness, and validity.
- On refusal assert no partial result is returned and the diagnostic identifies
  coplanar unsupported/invalid-result classification.
- Preserve disjoint, contained, overlapping, and existing supported union tests.

The fixture is committed geometry, not a visual-only reference.

## App-Type Proof

- GUI, console, API/service, and mixed-surface proof: not applicable.
- Library-only proof: public `boolean_union` output is classified and geometrically measured.

## Fixtures And Data

- Minimal coplanar bodies and full test-model enclosure composition.
- Production-data rule: committed deterministic geometry only.

## Acceptance

- [x] Feature spec is canonical and public route behavior is proved.
- [x] Success/refusal outputs and operand-order behavior are covered.
- [x] Helper-only or visual-only evidence cannot satisfy the contract.
