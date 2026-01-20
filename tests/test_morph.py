from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import morph
from impression.modeling.drawing2d import make_circle, make_ngon


def test_morph_positive():
    start = make_circle(radius=1.0)
    end = make_ngon(sides=6, radius=1.0)
    result = morph(start, end, t=0.5, samples=60)
    pts = result.outer.sample()
    assert pts.shape[0] > 0


def test_morph_invalid_t():
    start = make_circle(radius=1.0)
    end = make_ngon(sides=6, radius=1.0)
    with pytest.raises(ValueError):
        morph(start, end, t=-0.1)


def test_morph_hole_mismatch():
    start = make_circle(radius=1.0)
    end = make_ngon(sides=6, radius=1.0)
    end.holes.append(start.outer)
    with pytest.raises(ValueError):
        morph(start, end, t=0.5)
