# Surface Body Patch Families Research

## Topic

Research surface-body patch families for Impression's surface-first kernel.

This note answers:

- which patch families matter for a complete `SurfaceBody` implementation
- what each family must implement to be complete enough for modeling,
  tessellation, trimming, adjacency, booleans, and export
- which algorithms and data structures are shared across patch families
- which primitive modeling constructors are required by surface-body modeling

This is research, not a final architecture or specification.

## Project Baseline

Current project code defines the required v1 patch families as:

- `planar`
- `ruled`
- `revolution`

Current project code defers these families:

- `nurbs`
- `bspline`
- `subdivision`
- `implicit`
- `sweep`

Current code already has these surface-core records:

- `ParameterDomain`
- `TrimLoop`
- `SurfacePatch`
- `SurfaceShell`
- `SurfaceBody`
- `SurfaceBoundaryRef`
- `SurfaceSeam`
- `SurfaceAdjacencyRecord`

The current `SurfacePatch` contract requires at least:

- `point_at(u, v)`
- `derivatives_at(u, v)`
- `normal_at(u, v)` derived from first derivatives
- `frame_at(u, v)` derived from first derivatives
- `sample_grid(...)`
- `bounds_estimate(...)`
- stable canonical payload / cache identity

This baseline is consistent with the external B-rep pattern where faces carry
surface geometry, edges or seams carry shared boundary truth, and face-local
trim curves constrain the usable part of the surface.

## Findings

### Family Taxonomy

Surface-body modeling needs three groups of patch families.

The minimal constructive group is:

- planar patches
- ruled or linear-extrusion patches
- revolution patches

These cover boxes, polygonal solids, caps, prisms, cylinders, cones, spheres,
tori, basic text extrusion, simple hinges, and many initial loft surfaces.

The freeform analytic group is:

- Bezier patches
- B-spline patches
- NURBS patches

These cover high-quality lofts, smooth profile curves, fillets/blends,
import/export compatibility, and exact conic representation when using rational
weights.

The advanced procedural group is:

- sweep patches
- offset patches
- heightfield or displacement patches
- subdivision patches
- implicit patches

These cover path-driven modeling, thickening, terrain/textures, organic
surfaces, and procedural volumes. They are useful, but most should remain
deferred until the shared kernel algorithms are stable.

### Required Shared Patch Contract

Every patch family should expose the same kernel-facing contract:

- stable family name
- rectangular native parameter domain, even when trims make the visible area
  non-rectangular
- patch-local trim loops in parameter space
- optional periodicity flags for `u` and `v`
- optional singular boundary declarations, such as sphere poles or cone apexes
- `point_at(u, v)` evaluation
- first derivatives for normals and frames
- optional higher derivatives for curvature, continuity checks, fairing, and
  adaptive tessellation
- local parameter projection or closest-point query
- inverse mapping support for seams, picks, and boolean intersection curves
- bounding box estimation with family-specific exact or conservative paths
- boundary evaluator for named patch sides and trim-loop segments
- split/subdomain extraction for booleans and trimming
- transform handling that preserves exact parameters when possible
- canonical payload for hashing and cache reuse
- metadata split between kernel truth and consumer data

The v1 interface can keep higher derivatives and inverse mapping optional, but
complete surface-body modeling eventually needs them.

### Shared Data Structures

These data structures are common across patch types:

- `SurfaceBody`: body-level transform, metadata, and one or more shells.
- `SurfaceShell`: patch membership, seam membership, adjacency, closed/open
  classification inputs, and shell-level transform.
- `SurfacePatch`: family-specific geometry plus domain, trims, transform, and
  metadata.
- `ParameterDomain`: native `u`/`v` bounds and normalization policy.
- `TrimLoop`: ordered 2D loop in patch parameter space with `outer` or `inner`
  role.
- `SurfaceBoundaryLoop`: explicit loop-level structure for ordered boundary
  uses. The architecture recommends this even though the current code does not
  yet have a dedicated record.
- `SurfaceSeam`: shared shell-level boundary truth.
- `PatchBoundaryUse`: oriented patch-local use of one shared seam. The
  architecture recommends this as the richer replacement for bare boundary refs.
- `SurfaceAdjacencyRecord`: derived or compatibility view of patch adjacency.
- `Curve2D` / p-curve: patch-local trim curve in parameter space.
- `Curve3D`: canonical shared boundary geometry owned by the seam.
- `IntersectionCurve`: temporary or durable curve produced by patch/patch
  intersection.
- `FragmentGraph`: temporary boolean/decomposition graph of split patch
  fragments.
- `TessellationPlan`: deterministic samples, seam samples, grid policy, and
  tolerance policy.
- `TolerancePolicy`: global and local tolerances for coincidence,
  self-intersection, parameter equality, collapse, and curvature refinement.

### Shared Algorithms

These algorithms recur across patch families:

- parameter validation and clamping
- point and derivative evaluation
- normal/frame computation
- bounding box estimation
- curve evaluation in 2D and 3D
- trim-loop winding, containment, and orientation normalization
- polygon and curve clipping in parameter space
- closest-point projection from 3D to a patch
- seam compatibility checks between shared 3D curves and patch-local p-curves
- patch boundary evaluation for rectangular sides and trim-loop boundaries
- continuity classification at seams: positional, tangent, curvature, and
  intentional sharpness
- adaptive tessellation by tolerance, curvature, and trim boundaries
- seam-first tessellation: sample shared seams once and reuse samples on all
  participating patches
- degenerate-boundary handling for collapsed edges, poles, zero-area trims, and
  short sliver fragments
- patch/patch intersection discovery
- intersection mapping into both patches' parameter spaces
- patch splitting by trim or intersection curves
- fragment classification against another shell
- shell reconstruction from surviving fragments
- deterministic hashing and canonical ordering
- transform composition and transform baking for comparison
- metadata/provenance propagation

These are more important than individual patch classes because booleans,
watertight tessellation, and import/export all depend on them.

### Planar Patch Family

Representation:

- origin point
- two independent basis vectors
- rectangular parameter domain
- optional trims for polygons and holes

Evaluation:

```text
S(u, v) = origin + u * u_axis + v * v_axis
```

Complete implementation requires:

- exact point and derivative evaluation
- robust plane normal and orientation
- parameter projection from 3D to plane coordinates
- trim-loop support for arbitrary planar polygons and holes
- polygon winding normalization
- planar point-in-loop and loop containment
- exact or conservative bounds from domain and trims
- split by 2D lines, trim curves, and boolean intersection curves
- coplanar overlap classification
- face merging or canonicalization where adjacent planar patches are
  intentionally co-planar
- deterministic tessellation for rectangular and trimmed planar faces
- seam p-curves for each boundary use

Primary uses:

- caps
- boxes
- polyhedra
- planar profiles
- annotations and drafting planes
- boolean cut faces

Key risks:

- treating trimmed planar faces as simple rectangles
- losing hole orientation
- inconsistent coplanar boolean classification
- using approximate triangulation as the only source of planar truth

### Ruled Patch Family

Representation:

- two compatible guide curves
- parameter domain
- interpolation direction between curves

Evaluation:

```text
S(u, v) = (1 - u) * C0(v) + u * C1(v)
```

Complete implementation requires:

- guide-curve abstraction beyond polylines
- compatible guide parameterization and sampling
- exact or sampled derivatives from both guide curves
- detection of developable, cylindrical, conical, and general ruled cases
- singularity checks when guide curves collapse or cross
- seam compatibility along all four sides
- support for loft sidewalls and extrusion sidewalls
- split support in both `u` and `v`
- adaptive tessellation that respects guide curvature and rulings
- degenerate handling for apex-like ruled patches
- continuity classification between adjacent ruled spans

Primary uses:

- linear extrusion sidewalls
- simple loft bridges
- prism/frustum sidewalls
- developable bridges
- some sweep approximations

Key risks:

- assuming same point count means same parameterization
- normal flips when guide curves reverse
- bad tessellation near collapsed or nearly collapsed rulings

### Revolution Patch Family

Representation:

- profile or meridian curve
- axis origin
- axis direction
- start angle
- sweep angle
- parameter domain

Evaluation:

```text
S(u, v) = rotate(profile(v), axis, angle(u))
```

Complete implementation requires:

- profile curve abstraction beyond polylines
- axis/profile validation
- partial and full revolution support
- periodic `u` support for full sweeps
- collapsed-boundary support for poles and apexes
- exact handling of cylinder, cone, sphere, torus, and general revolved surface
  subtypes
- seam records for wrap boundaries
- planar cap generation for partial sweeps or capped solids
- robust derivatives near the axis
- adaptive tessellation near poles and high curvature
- intersection support with planes and other revolution patches
- profile projection and curve extraction for slicing

Primary uses:

- cylinders
- cones/frustums
- spheres
- tori
- rotate-extrude operations

Key risks:

- singular normals at poles or axis-touching profiles
- duplicate seam samples at full 360-degree wraps
- ambiguous inside/outside classification when a profile self-intersects

### Bezier Patch Family

Representation:

- rectangular control net
- degree in `u` and `v`
- optional rational weights

Complete implementation requires:

- Bernstein basis evaluation
- de Casteljau subdivision in both parameter directions
- derivative control net computation
- rational evaluation when weights exist
- conversion to/from B-spline spans where possible
- adaptive tessellation by flatness and curvature
- exact subpatch extraction
- boundary curve extraction
- continuity classification across matching boundaries

Primary uses:

- internal representation for B-spline/NURBS spans
- localized smooth patches
- subdivision limit approximations
- import/export bridge

Key risks:

- numerical instability at high degree
- uncontrolled degree growth after operations
- weight handling errors in rational patches

### B-Spline Patch Family

Representation:

- control-point grid
- degree in `u` and `v`
- knot vectors or knot values plus multiplicities
- optional periodicity
- optional weights if rationalized

Complete implementation requires:

- basis-function evaluation, preferably de Boor
- knot insertion
- degree elevation
- span extraction into Bezier patches
- derivative evaluation
- knot multiplicity and continuity rules
- periodic surface support
- boundary curve extraction
- subdomain extraction
- trimming and p-curve compatibility
- adaptive tessellation through span decomposition
- canonicalization of knots, multiplicities, poles, and periodic closures

Primary uses:

- smooth lofts
- fair surfaces
- imported CAD-like surfaces
- smooth path and profile infrastructure

Key risks:

- conflating B-spline and NURBS
- ignoring knot multiplicity effects on continuity
- unstable inverse mapping without robust projection

### NURBS Patch Family

Representation:

- B-spline control-point grid
- knot vectors and degrees
- positive weights
- optional periodicity

NURBS are rational B-splines. They should be implemented as a weighted
extension of the B-spline patch family, not as an unrelated evaluator.

Complete implementation requires everything required by B-spline patches plus:

- rational homogeneous evaluation
- rational derivatives
- positive weight validation
- exact conic support through weights
- weight-preserving transform and export behavior
- robust handling of near-zero denominator cases

Primary uses:

- exact circles, cylinders, cones, spheres, and tori when analytic primitives
  are represented as rational splines
- CAD import/export
- advanced loft and blend surfaces

Key risks:

- invalid or near-zero weights
- losing exactness by tessellating too early
- treating rational and non-rational continuity identically without checking
  weighted form

### Sweep Patch Family

Representation:

- profile curve or profile section
- rail/path curve
- frame or transport law
- scale/twist law
- optional guide curves

Complete implementation requires:

- path parameterization by arclength or stable normalized parameter
- moving-frame computation: Frenet where safe, parallel transport where
  curvature vanishes
- profile placement along the rail
- twist control
- scale law and orientation law
- self-intersection checks
- section correspondence for closed/open profiles
- seam construction along profile closure and rail boundaries
- conversion to ruled/B-spline/NURBS patches when possible
- fallback tessellation only at consumer boundary

Primary uses:

- pipes
- rails
- path-driven solids
- advanced loft replacement for sweep-like workflows

Key risks:

- frame flips at inflection points
- profile self-intersections on tight curvature
- accidental mesh-primary implementation

### Offset Patch Family

Representation:

- basis surface reference
- offset distance
- inherited or adjusted domain and trims

Evaluation:

```text
O(u, v) = S(u, v) + d * normal(S, u, v)
```

Complete implementation requires:

- basis surface with at least C1 continuity where normals must be unique
- normal evaluation
- singularity and self-intersection detection
- trim propagation or recomputation
- boundary offset curves
- shell thickening support
- fallback conversion for analytic cases where offset has a simpler exact form

Primary uses:

- shell/thicken operations
- clearance offsets
- surface displacement foundation

Key risks:

- offsets can self-intersect
- C0 basis surfaces do not have unique normals at seams
- offset trims often need reconstruction rather than blind copying

### Heightfield / Displacement Patch Family

Representation:

- base parameter domain
- height function or sampled grid
- base plane or base surface
- displacement direction or normal rule

Complete implementation requires:

- bilinear/bicubic or basis-function interpolation
- sampled-data storage with resolution metadata
- derivative estimation
- bounds including min/max displacement
- adaptive tessellation driven by height variation
- edge compatibility with neighboring patches
- optional displacement over arbitrary basis surfaces
- clear distinction between analytic heightfields and mesh-only relief

Primary uses:

- embossing
- terrain-like surfaces
- texture-derived geometry
- local relief and annotation effects

Key risks:

- treating the sample grid as a mesh kernel
- cracks at tile boundaries
- undersampling high-frequency displacement

### Subdivision Patch Family

Representation:

- control cage mesh
- subdivision scheme such as Catmull-Clark or Loop
- creases, corners, boundary rules, and optional hierarchical edits

Complete implementation requires:

- half-edge or equivalent manifold control-cage topology
- support for Catmull-Clark and/or Loop subdivision
- crease and boundary interpolation rules
- limit-surface evaluation
- patch extraction for regular regions
- extraordinary-vertex handling
- adaptive tessellation
- picking/projection onto the limit surface
- interoperability boundary with B-spline/Bezier patch extraction where possible

Primary uses:

- organic shapes
- sculptural surfaces
- imported subdivision assets

Key risks:

- non-manifold cages
- undefined behavior at extraordinary vertices
- hard integration with exact B-rep booleans

### Implicit Patch Family

Representation:

- scalar field `F(x, y, z) = 0`
- bounding domain
- optional material/field composition tree

Complete implementation requires:

- field evaluation
- gradient evaluation for normals
- bounding volume and interval estimates
- contouring/tessellation such as marching cubes, dual contouring, or adaptive
  continuation
- trimming or clipping support
- intersection classification against explicit patches
- conversion to explicit patches where possible
- clear boolean/composition semantics

Primary uses:

- metaballs and blends
- procedural volumes
- signed-distance modeling
- advanced soft booleans

Key risks:

- difficult exact seams with explicit B-rep patches
- topology changes during contouring
- expensive robust classification

## Primitive Constructors Required By Surface-Body Modeling

### Core 2D Inputs

Surface-body modeling needs these planar/profile primitives:

- point
- vector
- line segment
- polyline
- circular arc
- ellipse and elliptic arc
- Bezier curve
- B-spline curve
- NURBS curve
- rectangle
- circle
- ellipse
- regular polygon
- arbitrary polygon
- planar region with holes
- profile section with multiple loops

### Core 3D Curves

Surface-body modeling needs these spatial curves:

- line
- polyline
- circular arc
- helix
- Bezier curve
- B-spline curve
- NURBS curve
- path with arclength sampling
- curve frame or transport samples

### Surface Primitive Bodies

Surface-body modeling needs these body constructors:

- box: six planar patches with explicit seams
- plane/sheet: one planar patch, usually open
- polygonal prism: planar caps plus ruled sidewalls
- general linear extrude: planar caps plus ruled sidewalls
- rotate extrude: revolution sidewalls plus optional planar caps
- cylinder: revolution sidewalls plus planar caps
- cone/frustum: revolution sidewalls plus optional planar caps
- sphere: revolution patch or NURBS/analytic sphere with pole handling
- torus: revolution patch or NURBS/analytic torus with two periodic directions
- wedge: planar and ruled patches
- polyhedron: trimmed planar patches plus explicit seams
- text extrusion: planar glyph regions plus ruled sidewalls and planar caps
- hinge components: compositions of boxes, cylinders, boolean cuts, and
  structured thin surfaces
- heightfield body: displaced surface plus sidewalls/caps when solidified

### Modeling Operations That Generate Patches

Surface-body modeling needs these patch-generating operations:

- planar face construction from region
- linear extrusion
- rotation extrusion
- loft between profiles
- ruled bridge between curves
- sweep along path
- cap construction
- trim application
- split by curve
- boolean union/difference/intersection
- shell/thicken/offset
- fillet/blend, eventually
- chamfer, eventually
- transform and instance
- merge/coalesce compatible patches

## Implementation Completeness Checklist

For each patch family, complete implementation means the family can participate
in the same kernel flows as other families:

- construction validates all parameters and rejects degenerate state early
- evaluation returns deterministic points for all valid parameters
- first derivatives are available or explicitly unsupported for limited
  preview-only families
- normals and frames are correct except at declared singularities
- boundaries can be evaluated consistently
- trims can constrain the native domain
- seams can reference every boundary segment through p-curves or equivalent
- patch identity is stable under canonical payload generation
- transforms preserve or deliberately bake geometry
- tessellation is deterministic and seam-first
- bounds are conservative
- projection/inverse mapping exists for booleans and picks
- patch splitting can produce valid subpatches or valid trimmed fragments
- continuity can be classified at shared seams
- metadata/provenance survives construction, splitting, booleans, and
  reconstruction
- failure modes distinguish invalid input, unsupported family interaction, and
  numerical degeneracy

## Suggested Implementation Order

The lowest-risk order is:

1. finish the shared seam/use/trim data model
2. finish planar patch completeness
3. finish ruled patch completeness
4. finish revolution patch completeness
5. add a general curve abstraction used by profiles, trims, and seams
6. add Bezier patch support as the simplest freeform evaluator
7. add B-spline patch support
8. add NURBS as rational B-spline support
9. add sweep as an operation that emits ruled/B-spline/NURBS patches where
   possible
10. add offset, heightfield, subdivision, and implicit families only when their
   interactions with trims, seams, and booleans are explicitly bounded

## Implications

### For Impression v1 Surface Work

The current required families are the right minimal foundation:

- planar covers caps, boxes, text faces, drafting planes, and boolean cut faces
- ruled covers extrude sidewalls, simple loft spans, and prism/frustum sides
- revolution covers round primitives and rotate-extrude workflows

However, these are not complete until they share stronger boundary and trim
machinery. The biggest missing implementation surface is not another patch
class; it is the common infrastructure for p-curves, patch-boundary uses,
projection, splitting, and seam-first tessellation.

### For Booleans

Surface-body booleans require:

- patch/patch intersection
- intersection curves mapped to both parameter spaces
- patch-local trim fragment reconstruction
- fragment classification
- shell reconstruction
- explicit failure classification for unsupported family pairs

The first boolean implementation can be narrow if it is honest about supported
family pairs. Planar/planar, planar/ruled, planar/revolution, and simple
revolution/revolution cases provide the natural initial matrix.

### For Loft

Ruled patches are enough for simple loft sidewalls, but high-quality lofting
requires B-spline or NURBS support plus a better curve abstraction. Loft should
not solve all surface-kernel problems by itself; it should consume shared curve,
trim, seam, and tessellation infrastructure.

### For Primitives

Primitive coverage should be measured by whether a primitive emits valid
surface-body topology, not by whether it visually tessellates. Several current
internal primitive builders already emit useful patches but still have bounded
closed-shell or cap-seam limitations called out in comments.

### For Testing

Each patch family needs:

- constructor validation tests
- evaluation and derivative tests
- transform tests
- trim validation tests
- seam boundary tests
- tessellation determinism tests
- watertightness tests where closed shells are claimed
- boolean participation tests for supported family-pair interactions
- reference artifact tests for visually meaningful primitives

## References

Project-local:

- [Surface-first internal model architecture](../architecture/surface-first-internal-model.md)
- [SurfaceBody seam and adjacency architecture](../architecture/surfacebody-seam-adjacency-architecture.md)
- [SurfaceBody CSG architecture](../architecture/surfacebody-csg-architecture.md)
- [Surface-native capability replacement architecture](../architecture/surface-native-capability-replacement-architecture.md)
- [`src/impression/modeling/surface.py`](../../src/impression/modeling/surface.py)
- [`src/impression/modeling/_surface_primitives.py`](../../src/impression/modeling/_surface_primitives.py)
- [`src/impression/modeling/_surface_ops.py`](../../src/impression/modeling/_surface_ops.py)

External:

- Open CASCADE `Geom_Surface` reference. Documents common surface operations
  and lists concrete surface implementations including planes, B-splines,
  trimmed surfaces, linear extrusion, revolution, and analytic quadrics:
  <https://dev.opencascade.org/doc/refman/html/class_geom___surface.html>
- Open CASCADE BREP format documentation. Documents storage of vertices, edges,
  wires, faces, shells, solids, curves, surfaces, and triangulations:
  <https://dev.opencascade.org/doc/occt-7.9.0/overview/html/specification__brep_format.html>
- Open CASCADE `Geom_BSplineSurface` reference. Documents poles, weights, knots,
  multiplicities, degree, periodicity, knot insertion, and degree elevation:
  <https://dev.opencascade.org/doc/refman/html/class_geom___b_spline_surface.html>
- Open CASCADE `Geom_SurfaceOfLinearExtrusion` reference. Documents extruded
  surfaces from a basis curve and direction:
  <https://dev.opencascade.org/doc/refman/html/class_geom___surface_of_linear_extrusion.html>
- Open CASCADE `Geom_SurfaceOfRevolution` reference. Documents revolved
  surfaces, meridians, axis, angle parameterization, periodicity, and analytic
  subforms:
  <https://dev.opencascade.org/doc/refman/html/class_geom___surface_of_revolution.html>
- Open CASCADE `Geom_OffsetSurface` reference. Documents offset surfaces,
  normal-based evaluation, and C1 basis-surface requirements:
  <https://dev.opencascade.org/doc/occt-6.9.0/refman/html/class_geom___offset_surface.html>
- CGAL Polygon Mesh Processing manual. Useful as contrast for mesh boolean
  corefinement and exact-predicate requirements at mesh boundaries:
  <https://doc.cgal.org/latest/Polygon_mesh_processing/>
- Pixar OpenSubdiv subdivision surface documentation. Documents Catmull-Clark,
  Loop, feature-adaptive subdivision, hierarchical edits, and stencil-based
  evaluation:
  <https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html>
