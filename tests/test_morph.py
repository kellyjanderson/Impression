from __future__ import annotations

import importlib

import pytest

import impression.modeling as modeling


def test_morph_is_no_longer_exported_from_public_modeling_namespace() -> None:
    assert "morph" not in modeling.__all__
    assert "morph_profiles" not in modeling.__all__


def test_morph_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("impression.modeling.morph")
