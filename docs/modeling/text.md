# Modeling - Text

Impression can generate text outlines and turn them into meshes. Under the hood
the text is converted to 2D profiles and then extruded to 3D.

The text API also supports a surface-first path with `backend="surface"`, which
preserves the topology-native outline stage and terminates in surfaced text
before preview/export tessellation.

```python
from impression.modeling import make_text, text, text_profiles, text_sections
```

## Ownership Boundary

`text.py` owns:

- font file resolution
- glyph outline extraction
- character/line layout
- conversion of glyph commands into authored `Path2D` values
- text-local extrusion assembly for mesh output
- dispatch to the private surfaced linear-extrude builder for surfaced output

`text.py` does not own:

- generic path nesting (outer/hole detection)
- loop containment/classification policy
- winding normalization/triangulation behavior
- the public `extrude` module surface area

Those topology concerns are handled by [`topology`](topology.md) via shared
helpers (currently `sections_from_paths` in the text pipeline). See
[Topology Spec 07](../../project/specifications/topology-07-text-boundary.md).

The current implementation intentionally does not route text through the public
`linear_extrude(...)` API. Text keeps its extrusion path local so surface-first
text can evolve without inheriting public extrude-module coupling.

## Options

- `content`: string to render (including multi-line text).
- `depth`: positive value to extrude.
- `font_size`: glyph size.
- `center`: world-space anchor for the text block.
- `direction`: axis the text should face (defaults to +Z).
- `justify`: `"left"`, `"center"`, or `"right"` alignment before positioning.
- `valign`: `"baseline"`, `"top"`, `"middle"`, or `"bottom"` alignment for multi-line blocks.
- `letter_spacing`: extra tracking between glyphs.
- `line_height`: line spacing multiplier (relative to `font_size`).
- `font`: family name (defaults to `"Arial"`).
- `font_path`: explicit font file (recommended for reproducible results).
- `color`: RGB/RGBA tuple or color string (propagates through previews/exports).
- `backend`: `"mesh"` for legacy mesh-primary output, or `"surface"` for surfaced output.

Text uses FontTools to convert glyph outlines into Bezier segments. The helper
`text_profiles(...)`/`text_sections(...)` return a list of topology-native `Section` values you
can reuse for custom extrusions or lofts.

Current regression coverage checks more than non-empty output. The text tests
verify profile alignment/layout, requested extrusion depth, direction-axis
reorientation, and surfaced alias parity.

## Examples

```python
def build():
    brand = make_text("Impression", depth=0.15, font_size=0.4, color="#d07a5c")
    brand.translate((0, 0.1, 0), inplace=True)
    surfaced = make_text(
        "Surface",
        depth=0.12,
        font_size=0.32,
        center=(0, -0.35, 0),
        color="#7b8aa6",
        font_path="assets/fonts/NotoSansSymbols2-Regular.ttf",
        backend="surface",
    )
    tagline = text(
        "Parametric playground",
        depth=0.08,
        font_size=0.22,
        center=(0, -0.25, 0),
        color="#7b8aa6",
    )
    return [brand, tagline, surfaced]


def build_emoji():
    font_path = "assets/fonts/NotoSansSymbols2-Regular.ttf"
    eye = make_text(
        "👁",
        depth=0.12,
        font_size=0.6,
        font_path=font_path,
        color="#6a7fae",
    )
    return eye
```

- Example modules: `docs/examples/text/text_basic.py`, `docs/examples/text/text_emoji.py`
- Preview command: `impression preview docs/examples/text/text_basic.py`
- Bundled emoji-ready font (SIL OFL 1.1): `assets/fonts/NotoSansSymbols2-Regular.ttf`
