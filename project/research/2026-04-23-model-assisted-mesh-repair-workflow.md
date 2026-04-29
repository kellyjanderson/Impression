# Model-Assisted Mesh Repair Workflow

## Topic

Research on how Impression's current mesh and surface toolchain could support a
future model-assisted mesh repair workflow.

## Findings

### The current codebase already has the right downstream analysis primitives

The retained mesh toolchain is intentionally downstream and diagnostic:

- `analyze_mesh(...)`
- `section_mesh_with_plane(...)`
- `repair_mesh(...)`

The docs explicitly position them as:

- mesh QA and defect inspection
- plane sectioning for analysis
- foreign-mesh inspection before repair or reconstruction

That is exactly the right posture for a future repair workflow, because it
means the project already distinguishes:

- canonical modeling truth
- downstream mesh salvage and analysis

### The clean repair architecture is: analyze in mesh space, reconstruct in surface space

The future feature should not try to make mesh repair itself canonical.
Instead, the clean workflow is:

1. import foreign mesh
2. analyze and localize the defect in mesh space
3. section the mesh near the damaged span
4. recover contour stations from healthy neighboring slices
5. reconstruct the missing span with loft or another surfaced operation
6. tessellate only when a mesh repair result is needed downstream

This preserves the current project posture that app-owned surfaced geometry is
the canonical internal result.

### The current surfaced kernel is sufficient for bounded band-style reconstruction experiments

Current loft and surface infrastructure already support:

- surfaced `Loft(...)`
- tessellation of `SurfaceBody`
- shell/seam validation
- section-based analysis in tests

That makes bounded band-style repair research immediately plausible without
needing a whole new kernel first.

### Acceptance should be based on both mesh QA and section-space drift

A repaired result should not be accepted just because it is watertight.

The strongest bounded acceptance stack is:

- mesh validity:
  - watertightness
  - manifoldness
  - no degenerate triangles after export
- section agreement:
  - repaired sections agree with neighboring healthy span intent
- bounded geometric drift:
  - bbox / centroid continuity
  - no obvious cross-band jump

This keeps the repair lane aligned with the project's broader preference for
section- and silhouette-based verification.

### Repair should remain explicit about reinterpretation versus exact recovery

Foreign meshes are noisy, incomplete, and often under-defined. So the repair
lane must remain honest about whether it is:

- exactly reconstructing an implied clean band
- or reinterpreting damaged geometry into a plausible surfaced replacement

That distinction matters for both tooling trust and future test contracts.

## Implications

Recommended first future-repair milestone:

- foreign mesh input
- bounded defect localization
- plane-section contour extraction
- section cleanup / canonicalization
- loft reconstruction across a damaged band
- surfaced result as the internal truth
- mesh export only as the downstream repair output

Recommended posture:

- mesh tools remain explicit downstream tools
- surface reconstruction remains the canonical recovered result
- acceptance must report both mesh QA and section-space continuity

## References

- `project/future-features.md`
- `docs/modeling/mesh-tools.md`
- `docs/modeling/loft.md`
- `src/impression/mesh.py`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/surface.py`
