# Future-Feature Research Agenda

## Topic

Research agenda derived from the current future-feature idea documents under
`project/future-features/`.

## Findings

The current future-feature set already contains a useful question backlog, but
the questions are spread across multiple exploratory documents. To make those
ideas easier to refine later, the questions should be grouped into a durable
research agenda and answered in focused notes.

The current idea set reviewed for this agenda:

- `project/future-features/control-station-inference-architecture.md`
- `project/future-features/spanwise-loft-consolidation-architecture.md`
- `project/future-features/spanwise-loft-inline-enhancement-architecture.md`
- `project/future-features/spanwise-loft-postprocessing-optimization-architecture.md`
- `project/future-features/spanwise-loft-repair-tool-architecture.md`
- `project/future-features/trajectory-guided-loft-architecture.md`
- `project/future-features.md`

## Research Topics

### 1. Control Station Inference Semantics

Questions:

- What is the primary inferred truth:
  - point trajectories
  - loop trajectories
  - correspondence-field evolution
  - interval-local span operators
- What makes a station topology-critical versus shape-control-only?
- How should this concept relate to any future control-section authoring model?

Durable note:

- `project/research/2026-04-23-control-station-inference-semantics.md`

### 2. Control Station Inference Workflow And Determinism

Questions:

- How should the tool expose error metrics and acceptance thresholds?
- Should control stations be editable first-class authored objects after
  inference?
- How should user-pinned stations interact with inferred control reduction?
- Does the inferred progression remain fully deterministic under repeated runs?
- Should the first implementation be offline simplification only, or also offer
  interactive refinement?

Durable note:

- `project/research/2026-04-23-control-station-inference-workflow.md`

### 3. Spanwise Consolidation Qualification

Questions:

- What qualifies a run of local intervals as one larger coherent span?
- Should consolidation target fewer patches, different patch families, better
  seam placement, or all three?
- How should consolidation respect topology events that are locally real but
  globally over-segmenting?

Durable note:

- `project/research/2026-04-23-spanwise-loft-consolidation-qualification.md`

### 4. Spanwise Consolidation Branch Strategy

Questions:

- Should the first implementation live inside the planner or outside it?
- How should the tool report approximation or loss when consolidation is not
  exact?
- What larger-span evidence is sufficient for inline consolidation?
- How does the planner represent a larger-span result without breaking the
  current planner/executor boundary?
- How are local topology events preserved inside a longer-span realization?
- What adjacency and compatibility evidence should permit postprocess
  consolidation?
- Should the postprocess tool only merge equivalent local spans, or also refit
  them?
- How are simplification error and seam relocation reported?
- Should repair target only loft-authored geometry, or also foreign mesh-derived
  surface reconstructions?
- How much deviation from the source span is acceptable in a repair result?
- Should the repair branch reuse the same consolidation logic as the other two
  branches, or only share part of it?

Durable note:

- `project/research/2026-04-23-spanwise-loft-branch-strategy.md`

### 5. Trajectory-Guided Loft Representation

Questions:

- What is the first useful attachment level:
  - whole loft
  - region
  - track
- How should trajectory guidance interact with explicit station placement?
- Should explicit trajectory guidance constrain station origins, or only the
  evolution between them?
- How should trajectory inference interact with control-station inference?
- Can trajectory fitting remain deterministic and topology-aware?
- How much of this belongs inside the planner versus a preprocessing tool?

Durable note:

- `project/research/2026-04-23-trajectory-guided-loft-representation.md`

### 6. Curve Intent Inference From Dense Stations

Questions:

- What is the right inferred signal for "this dense progression is really a
  curve"?
- How should station position and station frequency over distance contribute to
  that inference?
- What evidence should distinguish intentional curve faceting from arbitrary
  dense linear sampling?

Durable note:

- `project/research/2026-04-23-curve-intent-inference-from-dense-stations.md`

### 7. Model-Assisted Mesh Repair Workflow

Generated research questions from the current future-feature note:

- What mesh-analysis and repair capabilities already exist in Impression that
  can support a surface-first repair workflow?
- What should the canonical repair pipeline be from foreign mesh to
  reconstructed surfaced result?
- What quality gates should bound repair acceptance?
- Where should repair stay diagnostic and downstream instead of becoming
  canonical modeling truth?

Durable note:

- `project/research/2026-04-23-model-assisted-mesh-repair-workflow.md`

### 8. Patchwise Loft Repair Structure

Questions:

- What minimum local structure is needed to define a patchwise loft repair?
- How should patch adjacency and seam intent be represented for local repair?
- How does patchwise repair relate to the broader `SurfaceBody` seam and
  adjacency model?
- Should patchwise loft be a standalone repair primitive or a mode of a more
  general surface reconstruction system?

Durable note:

- `project/research/2026-04-23-patchwise-loft-repair-structure.md`

## Implications

The most useful next refinement path is:

1. answer the questions at the research level first
2. use those answers to decide which future ideas deserve promotion into active
   architecture
3. only then break the promoted ideas into specification trees

This keeps the future-feature set exploratory without leaving its main unknowns
buried inside the idea docs.

## References

- `project/future-features.md`
- `project/future-features/control-station-inference-architecture.md`
- `project/future-features/spanwise-loft-consolidation-architecture.md`
- `project/future-features/spanwise-loft-inline-enhancement-architecture.md`
- `project/future-features/spanwise-loft-postprocessing-optimization-architecture.md`
- `project/future-features/spanwise-loft-repair-tool-architecture.md`
- `project/future-features/trajectory-guided-loft-architecture.md`
