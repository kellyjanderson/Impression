from __future__ import annotations

from pathlib import Path

from impression.modeling.drafting import make_dimension

FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "NotoSansSymbols2-Regular.ttf"
)


def test_make_dimension_arrow_only_when_text_missing():
    collection = make_dimension((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), offset=0.2, text=None)
    assert len(collection.items) == 1


def test_make_dimension_with_text_label_surface_collection():
    collection = make_dimension(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        offset=0.2,
        text="2.00",
        font_path=str(FONT_PATH),
    )
    assert len(collection.items) == 2


def test_make_dimension_missing_font_keeps_arrow():
    collection = make_dimension(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        offset=0.2,
        text="2.00",
        font_path="does-not-exist.ttf",
    )
    assert len(collection.items) == 1
