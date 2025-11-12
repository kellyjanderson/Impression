"""Emoji text example leveraging the bundled Noto Sans Symbols font."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_text

BRAND_BLUE = "#5A7BFF"
FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoSansSymbols2-Regular.ttf"


def build():
    if not FONT_PATH.exists():  # pragma: no cover - runtime safety
        raise FileNotFoundError(
            "Emoji font missing. Re-run `scripts/setup_fonts.py` or check assets/fonts/NotoSansSymbols2-Regular.ttf."
        )

    iris = make_text(
        "👁️",
        depth=0.15,
        font_size=0.6,
        font_path=str(FONT_PATH),
        color=BRAND_BLUE,
        center=(0.0, 0.0, 0.0),
    )

    glow = make_text(
        "👁️",
        depth=0.05,
        font_size=0.75,
        font_path=str(FONT_PATH),
        color="#8AA8FF",
        center=(0.0, 0.0, -0.05),
    )

    return [glow, iris]
