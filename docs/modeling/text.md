# Modeling — Text

`make_text()` lets you drop 3D (or flat) text directly into scenes and exports. It uses PyVista's Text3D under the hood, so any glyph your system font supports—including emoji—can be rendered.

```python
from impression.modeling import make_text
```

## Options

- `content`: string to render.
- `depth`: set to `0` for flat text or a positive value to extrude.
- `font_size`: uniform scale applied after glyph generation.
- `center`: world-space anchor for the text block.
- `direction`: axis the text should face (defaults to +Z).
- `justify`: `"left"`, `"center"`, or `"right"` alignment before positioning.
- `color`: RGB/RGBA tuple or color string (inherits through previews/exports).

## Example

```python
def build():
    brand = make_text("Impression", depth=0.15, font_size=0.4, color="#ff7a00")
    brand.translate((0, 0.1, 0), inplace=True)
    tagline = make_text(
        "Parametric playground",
        depth=0.0,
        font_size=0.2,
        center=(0, -0.25, 0),
        color=(0.55, 0.75, 1.0),
    )
    return [brand, tagline]
```

- Example module: `docs/examples/text/text_basic.py`
- Preview command: `impression preview docs/examples/text/text_basic.py`
