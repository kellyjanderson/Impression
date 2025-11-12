# Modeling — Text

`make_text()` now tessellates build123d glyphs, so the exact same geometry feeds previews, STL exports, and any CAD downstream step. If a font provides vector outlines, we can extrude it—emoji included—without touching PyVista primitives.

```python
from impression.modeling import make_text
```

## Options

- `content`: string to render.
- `depth`: `0` for a flat face or a positive value to extrude/bas-relief.
- `font_size`: glyph size passed straight to build123d.
- `center`: world-space anchor for the text block.
- `direction`: axis the text should face (defaults to −Y so glyphs look toward the camera); pass any vector to orient glyphs onto other faces.
- `justify`: `"left"`, `"center"`, or `"right"` alignment before positioning.
- `font`: family name (defaults to `"Arial"`).
- `font_path`: explicit font file (handy for bundled assets such as emoji fonts).
- `font_style`: build123d `FontStyle` or a lower/upper-case string (`"bold"`, `"italic"`, ...).
- `tolerance`: tessellation tolerance forwarded to build123d.
- `color`: RGB/RGBA tuple or color string (propagates through previews/exports.).
- Variation selectors (e.g., the trailing U+FE0F in `👁️`) are stripped automatically to avoid stray fallback glyphs in single-character emoji; specify multi-codepoint emoji without forcing text/emoji presentation when possible.

## Examples

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


def build_emoji():
    font_path = "assets/fonts/NotoSansSymbols2-Regular.ttf"
    eye = make_text(
        "👁️",
        depth=0.12,
        font_size=0.6,
        font_path=font_path,
        color="#5A7BFF",
    )
    return eye
```

- Example modules: `docs/examples/text/text_basic.py`, `docs/examples/text/text_emoji.py`, `docs/examples/logo/impression_mark.py`
- Preview command: `impression preview docs/examples/text/text_basic.py`
- Bundled emoji-ready font (SIL OFL 1.1): `assets/fonts/NotoSansSymbols2-Regular.ttf`
