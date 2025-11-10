"""Dimension helper demo."""

from __future__ import annotations

from impression.modeling.drafting import make_dimension


def build():
    dims = make_dimension((0, 0, 0), (1.5, 0, 0), offset=0.2, text="1.50", color="#ff5a36")
    return dims


if __name__ == "__main__":
    print(len(build()))
