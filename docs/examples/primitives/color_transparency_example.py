"""Opaque cube inside a translucent shell."""

from __future__ import annotations

from impression.modeling import make_box


def build():
    shell = make_box(size=(1.4, 1.4, 1.4), color=(0.6, 0.85, 1.0, 0.5))
    shell.translate((0.0, 0.0, 0.4), inplace=True)

    core = make_box(size=(0.8, 0.8, 0.8), color="#ff3b3b")
    core.translate((0.0, 0.0, 0.4), inplace=True)

    return [shell, core]


if __name__ == "__main__":
    meshes = build()
    print("Meshes:", len(meshes))
