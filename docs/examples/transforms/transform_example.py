"""Demonstrate translate(), rotate(), scale(), mirror() helpers."""

from __future__ import annotations

from impression.modeling import make_box, translate, rotate, scale, mirror


def build():
    base = make_box(size=(0.8, 0.8, 0.4), center=(0.0, 0.0, 0.2), color="#5A7BFF")
    shifted = translate(base.copy(), (1.2, 0.0, 0.0))
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=45.0)
    scaled = scale(base.copy(), (1.0, 0.6, 1.4))
    flipped = mirror(base.copy(), (1.0, 0.0, 0.0))
    return [base, shifted, turned, scaled, flipped]


if __name__ == "__main__":
    print("Meshes:", len(build()))
