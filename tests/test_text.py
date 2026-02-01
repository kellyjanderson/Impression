from __future__ import annotations

from pathlib import Path

import pytest

from impression.modeling import make_text, text_profiles
from impression.modeling.drawing2d import Profile2D

FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "NotoSansSymbols2-Regular.ttf"
)


def test_text_profiles_returns_profiles():
    profiles = text_profiles("A", font_size=1.0, font_path=str(FONT_PATH))
    assert profiles
    assert all(isinstance(profile, Profile2D) for profile in profiles)
    assert profiles[0].outer.segments


def test_make_text_mesh():
    mesh = make_text("A", depth=0.1, font_size=1.0, font_path=str(FONT_PATH))
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_make_text_invalid_depth():
    with pytest.raises(ValueError):
        make_text("A", depth=0.0, font_path=str(FONT_PATH))


def test_text_missing_font():
    with pytest.raises(FileNotFoundError):
        text_profiles("A", font_path="does-not-exist.ttf")
