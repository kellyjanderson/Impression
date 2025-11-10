# Impression Documentation Index

Impression is a parametric modeling toolkit that mixes Python, PyVista rendering, and CAD backends (build123d/OpenCascade). Use it to:

- Define parameterized models (`build()` functions) and preview them live with `impression preview`.
- Compose primitives and CSG operations programmatically via `impression.modeling`.
- Export watertight meshes (STL today, with STEP on the roadmap) for fabrication workflows.
- Capture automated preview screenshots using `scripts/run_preview_tests.py` to verify scenes visually.

## Reference Topics

- [Project Plan](project-plan.md) — roadmap, open questions, and future features (CAD fillets, gravitational modeling, etc.).
- Modeling Guides:
  - [Primitives](modeling/primitives.md) — boxes, cylinders, spheres, torus utilities, code snippets, and rendered examples.
  - [CSG Helpers](modeling/csg.md) — union/difference/intersection helpers with CLI-ready sample modules.
  - [Paths](modeling/paths.md) — polyline/spline utilities for sweeps and visualization.
  - [Text](modeling/text.md) — extruded or flat glyphs with full color support.
  - [Drafting Helpers](modeling/drafting.md) — lines, planes, arrows, and dimensions for 2.5D annotations.

Every example module defines `build()` so it works with the CLI out of the box. Common commands:

```bash
# preview any documented example
impression preview docs/examples/primitives/box_example.py

# export to STL
impression export docs/examples/csg/union_example.py --output dist/union.stl --overwrite

# automated preview test suite + screenshots under dist/preview-tests/
scripts/run_preview_tests.py
```

Looking for future work or advanced concepts? Check the project plan and the appendix on gravitational modeling for upcoming experiments.
