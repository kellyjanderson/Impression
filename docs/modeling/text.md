# Modeling - Text

Impression can generate text outlines and turn them into meshes. Under the hood
the text is converted to 2D profiles and then extruded to 3D.

```python
from impression.modeling import make_text, text, text_profiles, text_sections
```

## Ownership Boundary

`text.py` owns:

- font file resolution
- glyph outline extraction
- character/line layout
- conversion of glyph commands into authored `Path2D` values

`text.py` does not own:

- generic path nesting (outer/hole detection)
- loop containment/classification policy
- winding normalization/triangulation behavior

Those topology concerns are handled by [`topology`](topology.md) via shared
helpers (currently `sections_from_paths` in the text pipeline). See
[Topology Spec 07](../specs/topology-07-text-boundary.md).

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

Text uses FontTools to convert glyph outlines into Bezier segments. The helper
`text_profiles(...)`/`text_sections(...)` return a list of topology-native `Section` values you
can reuse for custom extrusions or lofts.

## Examples

```python
def build():
    brand = make_text("Impression", depth=0.15, font_size=0.4, color="#d07a5c")
    brand.translate((0, 0.1, 0), inplace=True)
    tagline = text(
        "Parametric playground",
        depth=0.08,
        font_size=0.22,
        center=(0, -0.25, 0),
        color="#7b8aa6",
    )
    return [brand, tagline]


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
