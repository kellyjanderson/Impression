"""Example Impression model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cube:
    edge: float = 10.0


def build():
    """Return a stub graph describing the model."""
    return Cube()
