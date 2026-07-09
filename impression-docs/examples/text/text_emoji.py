"""Emoji text primitive example."""

from __future__ import annotations

from impression.modeling import make_text

def build():
    return make_text(
        "👁",
        depth=0.12,
        font_size=0.6,
        font_path="assets/fonts/NotoSansSymbols2-Regular.ttf",
        color="#6a7fae",
    )


if __name__ == "__main__":
    build()
