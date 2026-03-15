from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import as_section, morph
from impression.modeling.drawing2d import make_circle, make_ngon


def test_morph_positive():
    start = as_section(make_circle(radius=1.0))
    end = as_section(make_ngon(sides=6, radius=1.0))
    result = morph(start, end, t=0.5, samples=60)
    assert result.regions
    assert result.regions[0].outer.points.shape[0] > 0


def test_morph_invalid_t():
    start = as_section(make_circle(radius=1.0))
    end = as_section(make_ngon(sides=6, radius=1.0))
    with pytest.raises(ValueError):
        morph(start, end, t=-0.1)


def test_morph_hole_mismatch():
    start_profile = make_circle(radius=1.0)
    end_profile = make_ngon(sides=6, radius=1.0)
    end_profile.holes.append(start_profile.outer)
    start = as_section(start_profile)
    end = as_section(end_profile)
    with pytest.raises(ValueError):
        morph(start, end, t=0.5)
