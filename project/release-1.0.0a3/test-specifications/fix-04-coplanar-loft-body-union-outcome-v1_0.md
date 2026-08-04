# Fix 04 Test: Coplanar Loft-Body Union Outcome

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One outcome-classification suite proves the reproduced union cannot silently discard either operand.

## Backlink

[Fix 04 specification](../specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)

## Manual Smoke

Replace the grouped-body workaround in the test enclosure with its union and
confirm either one complete body is shown or the declared refusal is displayed.

## Automated Smoke

Union a minimal box and lofted body sharing a coplanar face; assert success has
witness points from both operands, otherwise assert the typed coplanar refusal.

## Automated Acceptance

- Run the exact enclosure composition in both operand orders.
- On success assert bounds, occupied witness regions, watertightness, and validity.
- On refusal assert no partial result is returned and the diagnostic identifies
  coplanar unsupported/invalid-result classification.
- Preserve disjoint, contained, overlapping, and existing supported union tests.

The fixture is committed geometry, not a visual-only reference.
