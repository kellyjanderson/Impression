from __future__ import annotations

import numpy as np

from impression.modeling import (
    make_box,
    translate,
    rotate,
    rotate_euler,
    scale,
    resize,
    mirror,
    multmatrix,
)


def _bounds(mesh):
    return np.array(mesh.bounds, dtype=float)


def test_translate_bounds():
    base = make_box(size=(2.0, 4.0, 6.0))
    moved = translate(base.copy(), (1.0, 2.0, 3.0))
    assert np.allclose(_bounds(moved), np.array([0.0, 2.0, 0.0, 4.0, 0.0, 6.0]))


def test_rotate_axis_z_90():
    base = make_box(size=(2.0, 4.0, 1.0))
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=90.0)
    bounds = _bounds(turned)
    assert np.allclose(bounds[[0, 1]], [-2.0, 2.0], atol=1e-6)
    assert np.allclose(bounds[[2, 3]], [-1.0, 1.0], atol=1e-6)


def test_rotate_euler_matches_sequential():
    base = make_box(size=(1.0, 2.0, 3.0))
    sequential = rotate(base.copy(), axis=(1.0, 0.0, 0.0), angle_deg=10.0)
    sequential = rotate(sequential, axis=(0.0, 1.0, 0.0), angle_deg=20.0)
    sequential = rotate(sequential, axis=(0.0, 0.0, 1.0), angle_deg=30.0)

    euler = rotate_euler(base.copy(), angles_deg=(10.0, 20.0, 30.0))
    assert np.allclose(sequential.vertices, euler.vertices, atol=1e-6)


def test_scale_bounds():
    base = make_box(size=(2.0, 4.0, 6.0))
    scaled = scale(base.copy(), (2.0, 0.5, 1.0))
    assert np.allclose(_bounds(scaled), np.array([-2.0, 2.0, -1.0, 1.0, -3.0, 3.0]))


def test_resize_auto_axes():
    base = make_box(size=(2.0, 4.0, 6.0))
    resized = resize(base.copy(), (4.0, 0.0, 0.0), auto=[False, True, True])
    assert np.allclose(_bounds(resized), np.array([-2.0, 2.0, -4.0, 4.0, -6.0, 6.0]))


def test_mirror_flips_axis():
    base = make_box(size=(2.0, 2.0, 2.0), center=(2.0, 0.0, 0.0))
    flipped = mirror(base.copy(), (1.0, 0.0, 0.0))
    assert np.allclose(_bounds(flipped)[[0, 1]], [-3.0, -1.0])


def test_multmatrix_translation():
    base = make_box(size=(2.0, 2.0, 2.0))
    mat = np.eye(4)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    moved = multmatrix(base.copy(), mat)
    assert np.allclose(_bounds(moved), np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0]))
