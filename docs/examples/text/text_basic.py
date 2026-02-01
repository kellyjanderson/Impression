"""Basic text primitive example."""

from __future__ import annotations

from impression.modeling import make_text

def build():
    return make_text(
        "Impression",
        depth=0.12,
        font_size=0.4,
        justify="center",
        color="#8b7a6a",
    )


if __name__ == "__main__":
    build()
