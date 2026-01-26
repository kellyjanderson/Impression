# Impression Design System

## Core Components

- **Primitives:** Base shapes using Impression Blue as default fill; mix Impression Orange for emphasis.
- **Text Blocks:** `make_text(..., color=IMPRESSION_BLUE)` for headers, `...=IMPRESSION_ORANGE` for emphasis (currently disabled).
- **Drafting Overlays:** `make_line`, `make_arrow`, `make_dimension` adopt blue/orange pairings.

## Tokens

```python
IMPRESSION_BLUE = "#5A7BFF"
IMPRESSION_ORANGE = "#FF7A18"
```

## Usage

- Maintain contrast ratio > 4.5:1 when overlaying text on colored backgrounds.
- For booleans, assign base object `IMPRESSION_BLUE` and cutters `IMPRESSION_ORANGE` to produce consistent exports.
- Export guidelines: include brand palette in README/docs for downstream consumers.
