s # External Research — Control Station Inference

## Topic

External research relevant to control-station inference, especially how dense
section data is reduced, parameterized, and fitted into a more compact
representation.

## Findings

### Loft / skinning literature treats control-point growth as a real problem

Shutao Tang's paper on lofted B-spline interpolation from serial closed contours
is directly relevant because it starts from the same situation as the future
feature:

- serial section data
- varying contour complexity
- pressure to keep the resulting control structure manageable

The paper explicitly calls out that traditional lofted interpolation can cause a
large increase in control points and then studies conditions for robust closed
contour interpolation. That strongly supports the idea that Impression's dense
station stacks should eventually have a reduction / compaction lane instead of
being treated as the final authored form.

### Parameterization and knot placement are first-class subproblems

Two B-spline approximation papers are especially useful here:

- *Deep Learning Parametrization for B-Spline Curve Approximation*
- *A Deep Neural Network for Knot Placement in B-spline Approximation*

Their main value for Impression is not that we should copy the neural approach.
The valuable takeaway is that the literature treats these as explicit subproblems:

- choosing parameter values
- choosing knot number and knot positions

That maps cleanly to Impression's future problem:

- choosing which dense stations become retained control points
- choosing where reduced progression samples should remain

### Iterative fitting literature supports an offline analysis tool first

The RPIA paper on B-spline curve and surface fitting shows a clear pattern that
matches a conservative first implementation for Impression:

- large-scale fitting can be iterative
- fitting can converge toward least-squares results
- you do not need a full interactive authoring UI before the fitting stage is
  useful

That reinforces the earlier repo-grounded conclusion that
`infer_control_stations(...)` should begin as an offline or batch analysis tool
with diagnostics rather than as an immediately interactive modeling mode.

### Industrial CAD loft tools emphasize ordered sections and compatibility, not hidden reduction

Open CASCADE's loft and sweep tooling is instructive because it exposes the
practical industrial posture:

- `BRepOffsetAPI_ThruSections` builds a shell or solid from an ordered sequence
  of wires
- the tutorial explicitly exposes a compatibility check for the section set
- `GeomFill_Pipe` and `BRepOffsetAPI_MakePipeShell` expose path/section
  contracts, continuity options, guide-curve options, and simulation /
  approximation settings

The notable absence is also informative: these tools expect users to manage the
section set intentionally; they do not hide a semantic station-reduction tool
inside the basic loft call.

That suggests Impression should keep control-station inference as an explicit
tool or preprocessor rather than silently folding it into ordinary loft
execution.

## Implications

External research supports these design decisions:

- treat control-station inference as an explicit preprocessing or authoring tool
- treat retained stations as a parameterization / fitting problem, not merely a
  decimation problem
- keep topology-critical stations separate from shape-control stations
- start with deterministic, offline inference plus diagnostics before trying to
  invent a richer live-editing surface

It also suggests a likely first technical focus:

- reduced progression selection
- fit diagnostics
- tolerance / error metrics

rather than immediately attempting a general learned or heuristic curve-intent
engine.

## References

- Shutao Tang, *Optimal lofted B-spline surface interpolation based on serial
  closed contours* (2022)
  - https://arxiv.org/abs/2202.06330
- Pascal Laube, Matthias O. Franz, Georg Umlauf, *Deep Learning
  Parametrization for B-Spline Curve Approximation* (2018)
  - https://arxiv.org/abs/1807.08304
- Jiaqi Luo, Zepeng Wen, Hongmei Kang, Zhouwang Yang, *A Deep Neural Network
  for Knot Placement in B-spline Approximation* (2022 / CAD 2024)
  - https://arxiv.org/abs/2205.02978
- Nian-Ci Wu, Chengzhi Liu, *Randomized progressive iterative approximation for
  B-spline curve and surface fittings* (2022 / AMC 2024)
  - https://arxiv.org/abs/2212.06398
- Open CASCADE `BRepOffsetAPI_ThruSections` / package documentation
  - https://dev.opencascade.org/doc/refman/html/package_brepoffsetapi.html
- Open CASCADE bottle tutorial
  - https://dev.opencascade.org/doc/occt-7.7.0/overview/html/occt__tutorial.html
- Open CASCADE `GeomFill_Pipe`
  - https://dev.opencascade.org/doc/refman/html/class_geom_fill___pipe.html
- Open CASCADE `BRepOffsetAPI_MakePipeShell`
  - https://dev.opencascade.org/doc/refman/html/class_b_rep_offset_a_p_i___make_pipe_shell.html
