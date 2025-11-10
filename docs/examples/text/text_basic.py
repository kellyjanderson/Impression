"""Basic text primitive example."""

from __future__ import annotations

from impression.modeling import make_text


def build():
    hello = make_text(
        "Impression",
        depth=0.15,
        font_size=0.4,
        center=(0, 0, 0),
        color="#ff7a00",
    )
    tagline = make_text(
        "Parametric playground",
        depth=0.0,
        font_size=0.2,
        center=(0, -0.35, 0),
        color=(0.55, 0.75, 1.0),
    )
    return [hello, tagline]


if __name__ == "__main__":
    print("Meshes:", len(build()))
