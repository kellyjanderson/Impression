"""Two colored cubes example."""

from __future__ import annotations

from impression.modeling import make_box


def build():
    blue = make_box(size=(0.9, 0.9, 0.9), color=(0.55, 0.75, 1.0))
    blue.translate((-0.6, 0.0, 0.0), inplace=True)

    orange = make_box(size=(0.9, 0.9, 0.9), color="#ff7a00")
    orange.translate((0.6, 0.0, 0.0), inplace=True)

    return [blue, orange]


if __name__ == "__main__":
    meshes = build()
    print("Meshes:", len(meshes))
