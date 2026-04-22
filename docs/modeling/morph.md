# Modeling — Morph

The legacy profile morph capability has been removed from Impression.

This page remains as a tombstone until shared navigation and public exports are
updated to stop advertising `morph`.

## Status

- `morph(...)` is no longer a supported modeling capability.
- `morph_profiles(...)` is no longer a supported modeling capability.
- Calls now fail immediately with a removal error.

## Migration Direction

Use a supported surface-first workflow instead. In practice that means choosing
an explicit modeling path such as:

- loft planning and execution for shape transitions
- direct section construction followed by lofting
- surface-body operations where the intended result is not a profile blend

## Shared Follow-Up

This tombstone document can be deleted once shared integration files are
updated to remove the remaining public references to `morph`.
