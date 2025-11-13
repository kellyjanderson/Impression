# Impression Feature Pipeline

This pipeline reflects the consensus from the cross-functional review on 2025-11-11. It captures the
near-term features in flight, their desired user outcomes, and readiness requirements.

| Priority | Feature | Driver | Status | Notes |
| --- | --- | --- | --- | --- |
| P0 | VS Code integration polish | UX/Tools | In Progress | Embed reliable interpreter detection, one-click auto-install, and prepare for future webview preview. Track in `feature/vscode-interpreter`. |
| P1 | Uniform primitive API | Modeling | Planned | Normalize backend selection, ensure future CAD mesh parity, and document the API in `docs/modeling/primitives.md`. |
| P2 | Custom primitive tessellation | Modeling/QA | Planned | Move box/cylinder/sphere generation to our CAD pipeline so preview/export produce identical meshes. Requires regression tests comparing STL + preview. |
| P3 | Robust path & spline abstraction | Modeling/UX | Planned | Formalize `Path` types (polyline, Bezier, spline) with sweep/extrude helpers and documentation + examples. |
| P4 | Chamfer/fillet tooling | Advanced Modeling | Planned | Provide selection helpers, radii presets, and auto-round utility once primitives/tessellation stabilize. |

### Supporting Workstreams

- **Testing:** adopt pytest for unit-style coverage; continue screenshot/STL suites as acceptance tests.
- **Docs & voice:** keep README/docs aligned with inclusive tone; every feature merges with updated examples and preview images.
- **Design/UX:** capture preview/webview requirements; design minimal controls for future IDE panel.

This document should be updated after each planning sync.
