> Status: Deprecated historical spec.
> Active work now lives in the surface-first specification tree and the current
> implementation. Retained for project history only.

# Topology Spec 07: text.py Topology Boundary

## Goal

Keep `text.py` focused on font extraction/layout and remove long-term ownership of generic polygon topology assembly.

## text.py Owns

- font file resolution
- glyph extraction from font outlines
- character and line layout
- conversion from glyph commands to authored paths

## text.py Does Not Own

- path nesting to determine outer/hole structure
- generic containment testing
- loop classification and winding normalization policies

## Required Integration

- text path sets are handed to topology assembly helpers (e.g. `sections_from_paths`).
- topology layer returns region/profile structures for extrusion/loft.

## Deliverables

- explicit API boundary documentation
- topology-backed profile assembly in text pipeline
- no duplicate point-in-polygon or area classifiers in text module

## Completion Criteria

- `text.py` contains no generic topology algorithms.
- all topology assembly behavior is shared and testable in topology surface.
