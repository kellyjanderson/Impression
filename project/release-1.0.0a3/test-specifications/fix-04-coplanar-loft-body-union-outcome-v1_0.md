# Fix 04 Test: Coplanar Loft-Body Union Outcome

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify the coplanar union preserves or refuses both operands

- Input: a minimal coplanar fixture and the full test-model enclosure composition.
- Work: exercise both operand orders and measure success validity/witnesses or the
  typed refusal classification.
- Output: a committed boolean outcome regression alongside existing union controls.
- Complete when: no fixture can return a partial body and supported unions still pass.

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
