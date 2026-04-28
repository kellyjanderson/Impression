# Low-Level Construct Gap Report

## Topic

Low-level modeling and reconstruction constructs that are still missing or not
yet first-class in Impression, based on the current future-feature and research
set.

## Scope

This report does not restate whole future features.

It asks a narrower question:

> What low-level building blocks are implied by the research, but are not yet
> clearly present as first-class constructs in the current repo?

The goal is to identify the foundational objects, records, patch families, and
fit/reconstruction contracts that future work keeps pointing toward.

## Repo Baseline

Current adjacent constructs that already exist:

- `Path3D` with `Line3D`, `Arc3D`, and `Bezier3D`
- topology-aware loft planning and ambiguity inspection
- `SurfaceBody` / `SurfaceShell` / surfaced patch families
- retained mesh analysis and plane sectioning

The strongest current nearby files are:

- `src/impression/modeling/path3d.py`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/surface.py`
- `src/impression/mesh.py`

The strongest currently visible gap is that the research repeatedly points to
constructs beyond those foundations, especially around:

- compact smooth curve representation
- deterministic fitting metadata
- richer patch families
- reconstruction/repair intermediates

## Findings

### 1. First-Class B-Spline Curve Objects Are Missing

The research strongly points to B-spline as a needed low-level representation,
but the current repo surface does not expose a first-class B-spline object
alongside:

- `Line3D`
- `Arc3D`
- `Bezier3D`

Missing constructs:

- `BSpline2D`
- `BSpline3D`
- explicit knot-vector ownership
- explicit degree ownership
- explicit periodic / closed policy
- deterministic evaluation and derivative access

Why this matters:

- control-station inference research repeatedly reduces to parameterization and
  knot-placement questions
- trajectory-guided loft needs a richer smooth guide than piecewise line/arc/
  bezier chains
- reconstruction and refit work needs a compact smooth representation that is
  not just polyline truth

### 2. Parameterization And Knot-Placement Policy Objects Are Missing

The research does not just point to “B-splines” in the abstract. It points to
parameterization and knot placement as explicit subproblems.

Missing constructs:

- parameter-assignment policy object
- knot-count policy object
- knot-placement policy object
- fit configuration record
- fit residual / tolerance record

These should not remain hidden numeric details inside future fitting helpers.

Why this matters:

- control-station inference and curve-intent inference both depend on choosing
  compact explanations for dense data
- if those choices are implicit, the future tools will be hard to debug, hard
  to compare, and hard to trust

### 3. Path3D Does Not Yet Have A High-Control Smooth Segment Type

`Path3D` already exists, but it currently stops at:

- line
- arc
- bezier

That means the path layer is still missing a canonical smooth segment for:

- longer guided trajectories
- imported or fitted smooth paths
- future path-aware loft control

Missing constructs:

- `BSpline3D` path segment integration
- mixed-segment path sampling rules that include B-spline
- path-level normalization and closure rules for B-spline segments

Why this matters:

- trajectory-guided loft research clearly favors whole-loft shared-path
  guidance first
- the path layer is the cleanest first consumer of B-spline support

### 4. Trajectory Attachment Constructs Are Missing

The trajectory-guided research repeatedly distinguishes possible attachment
levels:

- whole loft
- region
- track

Those are conceptual layers today, but not first-class data structures.

Missing constructs:

- shared trajectory input record
- region-trajectory attachment record
- track-trajectory attachment record
- stable region / track trajectory identifiers
- deterministic trajectory sampling + attachment resolution contract

Why this matters:

- without these records, trajectory guidance stays a hand-wavy future idea
- the path itself is not enough; the system also needs a durable answer to
  “what is this curve attached to?”

### 5. Control-Station Inference Result Objects Are Missing

The research has become fairly clear that control-station inference should not
just “return fewer stations.”

It needs a structured result.

Missing constructs:

- retained-station classification record
  - topology station
  - control station
- retained-station provenance metadata
- user-pin / hard-constraint record
- reduced progression result object
- fit diagnostics bundle
- structural preservation report

Why this matters:

- the research now distinguishes topology truth from shape-control retention
- without result objects, future inference risks collapsing into plain decimate-
  and-hope tooling

### 6. Curve-Intent Inference Needs Descriptor-Level Signals That Are Not Yet First-Class

The curve-intent research argues that dense stations communicate more than raw
spacing. The strongest signals are things like:

- loop descriptor continuity
- correspondence stability
- station density over distance
- smooth change in size / centroid / anisotropy

Those are good research conclusions, but they are not yet obvious durable low-
level constructs.

Missing constructs:

- section descriptor record
- loop descriptor record
- correspondence-track descriptor record
- span-local curve-intent evidence record
- curve-intent candidate report

Why this matters:

- future inference will need these as the bridge between “dense stations” and
  “smooth higher-level curve explanation”

### 7. Spanwise Consolidation Needs Explicit Grouping And Compatibility Records

The spanwise consolidation research strongly suggests the first useful path is:

- exact postprocess grouping first
- planner promotion later

But there is no visible first-class grouping object for that yet.

Missing constructs:

- grouped interval run record
- span compatibility report
- exact-vs-approximate consolidation result record
- seam relocation report
- patch-count reduction report
- consolidation stop / refusal diagnostics

Why this matters:

- without these objects, spanwise consolidation risks becoming hidden executor
  magic instead of inspectable planning or postprocess behavior

### 8. Additional Surface Patch Families Are Missing Or Underdefined

The research repeatedly points toward future surfaced reconstruction and
consolidation needing patch families beyond the current practical surfaced set.

The strongest missing patch-family candidates suggested by the research are:

- B-spline surface patch
- curve-network-derived patch families
- reconstruction-oriented local patch families for repair
- possibly developable or guided-surface families in specific reconstruction
  contexts

This is distinct from “we need B-spline curves.”

It is the separate question of whether surfaced reconstruction/refit needs new
native patch species.

Why this matters:

- spanwise refit and patchwise repair both eventually want something richer than
  only ruled / planar / revolution-style surfaced outcomes
- but the research also says this should be later than first-class B-spline
  curve support

### 9. Section Reconstruction Intermediates Are Missing

The repair and reconstruction research points to a future workflow where section
evidence matters a lot.

Current retained mesh sectioning exists, but future surfaced reconstruction will
need better intermediate forms between:

- raw sectioned mesh loops
- final surfaced repair result

Missing constructs:

- canonicalized section contour object
- cleaned section profile record
- section-to-station conversion record
- section confidence / quality report
- sparse cross-section reconstruction input bundle

Why this matters:

- section extraction is currently a useful retained diagnostic tool
- future reconstruction needs a more canonical bridge from extracted sections to
  surfaced modeling inputs

### 10. Local Boundary And Patch-Neighborhood Repair Records Are Missing

The patchwise repair research is especially clear that patchwise repair should
start from explicit local boundary truth rather than vague local geometry.

Missing constructs:

- local boundary-ring record
- patch-neighborhood descriptor
- local seam-intent record
- repair neighborhood input bundle
- repaired patch integration report

Why this matters:

- patchwise repair is not just “run loft on a local region”
- it needs explicit local structural records so the repaired result can rejoin
  the existing `SurfaceBody` model honestly

### 11. Exact Versus Approximate Result Taxonomy Is Missing In Geometry Work

Several research branches point toward a future where the system needs to say
clearly whether a result is:

- exact
- fitted / approximate
- repaired / reinterpreted

That posture already exists strongly in testing work, but not yet as an obvious
low-level geometry contract for future fitting/reconstruction features.

Missing constructs:

- geometric result taxonomy record
- approximation drift metrics bundle
- repair deviation report
- consumer-facing “exact vs approximate” metadata contract

Why this matters:

- spanwise consolidation
- B-spline fitting
- surfaced reconstruction
- repair tooling

all become much easier to trust if this boundary is first-class.

## Consolidated Missing Construct List

Grouped into the smallest meaningful low-level families, the research currently
points toward these missing constructs:

### Curve / Fit Core

- `BSpline2D`
- `BSpline3D`
- knot vector ownership
- parameterization policy
- knot-placement policy
- fit configuration + residual record

### Path / Trajectory

- B-spline-aware `Path3D` segment support
- shared trajectory record
- region-trajectory attachment record
- track-trajectory attachment record

### Inference / Reduction

- retained-station classification record
- retained-station provenance record
- user-pin / hard-constraint record
- reduced progression result object
- section / loop / track descriptor records
- curve-intent candidate report

### Span Consolidation

- grouped interval run record
- compatibility analysis report
- exact/approximate consolidation result record
- seam relocation report

### Surface / Reconstruction

- B-spline surface patch family
- reconstruction-oriented patch families
- canonicalized section contour object
- section-to-station conversion record
- local boundary-ring record
- patch-neighborhood repair record

### Cross-Cutting Truth / Diagnostics

- exact-vs-approximate taxonomy
- drift / deviation metadata
- structural preservation report

## Priority Read

The research suggests this priority order:

1. B-spline curve constructs
2. parameterization / knot / fit policy constructs
3. path / trajectory integration constructs
4. control-station inference result objects
5. spanwise grouping and compatibility records
6. reconstruction / repair intermediates
7. later surfaced B-spline patch families

That matches the broader architectural direction:

- curves first
- fitting and trajectory usage next
- surfaced patch expansion later

## Conclusion

The most important low-level gaps are not just “missing patch species.”

Across the research spectrum, the bigger missing foundation is:

- compact smooth curve primitives
- fit / parameterization ownership
- structured inference results
- reconstruction intermediates
- exact-vs-approximate reporting

New patch families are part of the gap, but they are not the whole gap and they
are probably not the first gap to close.

## References

- `project/future-features.md`
- `project/future-features/control-station-inference-architecture.md`
- `project/future-features/spanwise-loft-consolidation-architecture.md`
- `project/future-features/trajectory-guided-loft-architecture.md`
- `project/research/2026-04-23-control-station-inference-semantics.md`
- `project/research/2026-04-23-control-station-inference-workflow.md`
- `project/research/2026-04-23-curve-intent-inference-from-dense-stations.md`
- `project/research/2026-04-23-trajectory-guided-loft-representation.md`
- `project/research/2026-04-23-spanwise-loft-consolidation-qualification.md`
- `project/research/2026-04-23-spanwise-loft-branch-strategy.md`
- `project/research/2026-04-23-model-assisted-mesh-repair-workflow.md`
- `project/research/2026-04-23-patchwise-loft-repair-structure.md`
- `project/research/2026-04-23-external-control-station-inference.md`
- `project/research/2026-04-23-external-curve-intent-from-dense-stations.md`
- `project/research/2026-04-23-external-spanwise-loft-consolidation.md`
- `project/research/2026-04-23-external-trajectory-guided-loft.md`
- `project/research/2026-04-23-external-model-assisted-mesh-repair.md`
- `project/research/2026-04-23-external-patchwise-loft-repair.md`
- `src/impression/modeling/path3d.py`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/surface.py`
- `src/impression/mesh.py`
