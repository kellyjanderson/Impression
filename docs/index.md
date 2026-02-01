# Impression Documentation

Impression is a parametric modeling toolkit for building watertight meshes in Python. Models
return internal mesh objects; PyVista is used purely as a viewer. This index is the map to the
library, the tools, and the examples.

## Start Here

- **Quickstart**: [README](../README.md)
- **CLI Reference**: [CLI](cli.md)
- **Examples**: [`docs/examples/`](examples/) (every file has a `build()` function)
- **Agent Bootstrap**: [Agent Guide](agents.md)

## Tutorials

- [Getting Started](tutorials/getting-started.md) - the first full walkthrough.
- [Serious Modeling Workflow](tutorials/serious-modeling.md) - a more complete part built step by step.

## Modeling Guides

- [Primitives](modeling/primitives.md) - boxes, cylinders, spheres, torus utilities, code snippets, and rendered examples.
- [CSG Helpers](modeling/csg.md) - union/difference/intersection helpers with CLI-ready sample modules.
- [Extrusions](modeling/extrusions.md) - linear and rotate extrude of 2D profiles.
- [Paths](modeling/paths.md) - polyline/spline utilities for sweeps and visualization.
- [Path3D](modeling/path3d.md) - true-curve 3D paths (line/arc/bezier).
- [2D Drawing](modeling/drawing2d.md) - profile and path primitives for vector-style modeling.
- [Morph](modeling/morph.md) - profile interpolation between two shapes.
- [Loft](modeling/loft.md) - surface lofting between multiple profiles.
- [Heightmaps](modeling/heightmaps.md) - heightfield generation and image-based displacement.
- [Text](modeling/text.md) - extruded or flat glyphs (including emoji) with full color support.
- [Drafting Helpers](modeling/drafting.md) - lines, planes, arrows, and dimensions for 2.5D annotations.
- [Transforms](modeling/transforms.md) - rotate and translate meshes after creation.
- [Groups](modeling/groups.md) - move collections together without losing the parts.

## Common Commands

```bash
# preview any documented example
impression preview docs/examples/primitives/box_example.py

# export to STL
impression export docs/examples/csg/union_example.py --output dist/union.stl --overwrite

# automated preview test suite + screenshots under dist/preview-tests/
scripts/run_preview_tests.py

# STL export + watertight validation (writes to dist/stl-tests/)
scripts/run_stl_tests.py
```

## Tooling

- [VS Code Integration](vscode.md) - extension commands, packaging, and installation steps.

## Planning & Roadmap

- [Project Plan](project-plan.md) - roadmap, open questions, and future features.
- [Feature Pipeline](feature-pipeline.md) - current implementation order.

## Project Records

- [Brand Guide](brand/brand-guide.md)
- [Design System Notes](brand/design-system.md)
- [Style Guide](brand/style-guide.md)
- [Issue: Preview hot reload fails](issues/impression-preview-hot-reload-fails.md)
- [Issue: Camera resets on hot reload](issues/camera-reset-on-hot-reload.md)
- [PR: Hot reload resilience](prs/bugfix-impression-preview-hot-reload.md)
- [PR: Camera reset on reload](prs/bugfix-camera-reset-on-hot-reload.md)
- [PR: Configurable units](prs/feature-config-unit-defaults.md)
- [Meeting Notes (2025-11-11)](meetings/2025-11-11.md)
- [Findings (2025-11-11)](findings/2025-11-11.md)
