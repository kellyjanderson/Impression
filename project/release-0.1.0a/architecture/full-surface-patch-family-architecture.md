# Full Surface Patch Family Architecture

## Overview

This architecture replaces the earlier assumption that some surface patch
families can remain deferred.

The surface-first kernel should support every patch family needed for authored
surface body modeling as first-class `SurfacePatch` implementations. Patch
families may still land in staged implementation order, but they are no longer
architectural exclusions.

The complete surface patch family set is:

- planar
- ruled
- revolution
- B-spline
- NURBS
- sweep
- subdivision
- implicit

The goal is not to bolt these on as special cases. The goal is to make them
all obey the same surface-kernel laws:

- evaluable parameter domain
- stable identity
- trim compatibility
- transform policy
- metadata policy
- tessellation boundary
- `.impress` persistence
- seam and adjacency participation
- explicit unsupported-operation diagnostics where an operation cannot yet
  consume a given family

## Relationship To Existing Architecture

This document extends:

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)

It also supersedes the deferral posture described by:

- [Surface Spec 66: Deferred Patch Families and Explicit Exclusions](../specifications/surface-66-deferred-patch-families-exclusions-v1_0.md)

Spec 66 may remain historically useful as a statement of the original bounded
V1 scope, but it is no longer acceptable as the target architecture.

Surface Spec 139 replaces Spec 66 with `PATCH_FAMILY_CAPABILITY_MATRIX`. No
family is architecturally deferred; a family may be `available` or `planned` by
operation, and unsupported operations must report capability-aware diagnostics.

## Architectural Principle

Patch-family support is a kernel capability, not a feature-specific escape
hatch.

Every family must answer the same core questions:

- What is the family parameter domain?
- How does `point_at(u, v)` work?
- How do derivatives and normals work?
- How are trims interpreted?
- How does tessellation sample it?
- How does it participate in seams and adjacency?
- How is it persisted in `.impress`?
- Which modeling operations can produce it?
- Which modeling operations can consume it exactly?

If a family cannot yet participate in a downstream operation, the operation
must refuse with an explicit family-aware diagnostic. It must not silently mesh
fallback or degrade surface truth.

## Shared Patch Contract

All patch families implement the existing `SurfacePatch` contract:

- `family`
- `domain`
- `capability_flags`
- `trim_loops`
- `transform_matrix`
- `metadata`
- `point_at(u, v)`
- `derivatives_at(u, v)`
- `geometry_payload()`
- `canonical_payload()`
- `stable_identity`
- `cache_key`

The shared implementation should remain small. Family-specific geometry belongs
in family-specific patch classes or helpers.

## Common Data Structures

### ParameterDomain

Every patch has a parameter domain.

For analytic patches, the domain may be normalized to `[0, 1] x [0, 1]`.
For spline and subdivision patches, the domain may reflect knot or chart
structure internally, but public evaluation should still normalize through the
shared `ParameterDomain` contract unless a future spec explicitly adds
multi-chart domains.

### TrimLoop

Trims are always expressed in patch-local parameter space.

Every patch family must define what UV space means and how trim boundaries map
to surface-space curves.

### SurfaceBoundaryRef

Boundary references must remain family-neutral. A seam can connect a planar
edge to a NURBS edge, or a sweep boundary to a subdivision boundary, if the
geometric and topological validity rules pass.

### SurfaceSeam

Seams remain shell-owned shared-boundary truth.

Patch families do not own shared topology independently; they expose boundary
uses that seams can reference.

### Tessellation Request

Tessellation should be a family-neutral request plus a family-specific sampling
strategy. The strategy may differ by family, but the output contract remains a
mesh consumer result with traceable surface-patch metadata.

### .impress Payload

`.impress` must serialize every supported patch family with enough typed
geometry payload to reconstruct the runtime patch exactly.

The `.impress` architecture should be updated so B-spline, NURBS, sweep,
subdivision, and implicit are no longer listed as deferred load failures once
their specs exist.

## Family Responsibilities

### PlanarSurfacePatch

Planar patches represent flat parametric surfaces.

Responsibilities:

- evaluate points from origin/u-axis/v-axis
- provide constant derivatives
- support outer and inner trims
- support caps, planar faces, box faces, flat boolean fragments
- provide exact planar boundary curves for seams

Primary producers:

- planar primitives
- caps
- planar boolean fragments
- drafting geometry

### RuledSurfacePatch

Ruled patches interpolate between two compatible guide curves.

Responsibilities:

- evaluate points by interpolating between start/end guide curves
- support loft side walls and linear bridge surfaces
- preserve protected boundary sample order where supplied by loft
- expose both guide curves as seam-capable boundaries

Primary producers:

- loft
- extrude-like operations
- bridge operations
- linear sweeps where no richer sweep patch is required

### RevolutionSurfacePatch

Revolution patches sweep a profile around an axis by angle.

Responsibilities:

- evaluate analytic rotational surfaces
- support partial and full revolutions
- define seam behavior at periodic closure
- support lathe-style primitives and rotate-extrude operations

Primary producers:

- cylinders/cones/spheres/torus-like primitives where represented analytically
- rotate-extrude
- round threading and hinge elements where rotational truth is useful

### BSplineSurfacePatch

B-spline patches represent non-rational tensor-product spline surfaces.

Responsibilities:

- store degree in `u` and `v`
- store knot vectors in `u` and `v`
- store control net
- evaluate basis functions and derivatives
- support open, clamped, periodic, and non-uniform knot policies where planned
- expose isoparametric and trim-compatible boundary curves

Primary producers:

- fitted loft surfaces
- fairing/reconstruction
- smooth blends
- user-authored spline surface tools

Required common algorithms:

- de Boor evaluation
- basis derivative evaluation
- knot validation
- control-net validation
- adaptive tessellation by curvature/error
- optional knot insertion/refinement

### NURBSSurfacePatch

NURBS patches extend B-spline patches with rational weights.

Responsibilities:

- store degree, knot vectors, weighted control net, and weights
- evaluate homogeneous rational surface points
- evaluate derivatives with rational correction
- represent conics and many CAD-style exact analytic surfaces
- preserve weight identity and validation in `.impress`

Primary producers:

- imported CAD-like surface data
- exact conic surfaces when represented as rational splines
- advanced fillets/blends
- future STEP/IGES adapters

Required common algorithms:

- B-spline basis evaluation
- homogeneous coordinate evaluation
- rational derivative evaluation
- weight validation
- knot insertion/refinement preserving rational shape

Relationship to B-spline:

NURBS should reuse the B-spline basis and knot infrastructure. The rational
layer should be a thin extension over the non-rational implementation.

### SweepSurfacePatch

Sweep patches represent a profile transported along a path.

Responsibilities:

- store profile curve or profile section reference
- store trajectory/path curve
- store frame/transport policy
- evaluate profile points along path parameters
- support twist, scale, and orientation semantics
- expose start/end/profile/path boundaries for seams

Primary producers:

- path extrude
- pipe/tube
- threading
- cables/rails
- swept text/strokes

Required common algorithms:

- path evaluation
- moving-frame transport
- orientation/twist interpolation
- profile evaluation
- self-intersection and degeneracy diagnostics

Important distinction:

A sweep patch is not just a ruled patch with more samples. It preserves the
authoring truth that a profile moved along a trajectory, which matters for
editing, persistence, and exact downstream behavior.

### SubdivisionSurfacePatch

Subdivision patches represent control-cage driven surfaces.

Responsibilities:

- store control cage vertices/faces
- store subdivision scheme
- evaluate limit surface or approved approximation
- provide boundary and crease metadata
- participate in tessellation with deterministic refinement

Primary producers:

- organic modeling tools
- imported subdivision assets
- future sculpting workflows

Required common algorithms:

- subdivision scheme evaluation
- limit-position and limit-normal evaluation where available
- crease/sharpness handling
- deterministic refinement
- boundary extraction for seams

V1 likely scheme:

Catmull-Clark should be the first subdivision scheme unless a later spec
chooses otherwise.

### ImplicitSurfacePatch

Implicit patches represent surfaces defined by scalar fields or signed
distance-style functions.

Responsibilities:

- store a bounded domain or extraction volume
- store field definition in a safe, serializable representation
- evaluate scalar field and optionally gradient
- define trim or boundary interaction with shells
- tessellate through deterministic extraction

Primary producers:

- metaballs
- procedural fields
- offset/thickening fields
- imported implicit assets
- certain blend/fillet approximations

Required common algorithms:

- scalar field evaluation
- gradient evaluation where available
- bounding domain validation
- deterministic isosurface extraction
- seam/trim approximation policy

Security requirement:

`.impress` must not persist executable arbitrary code for implicit fields.
Implicit definitions must be declarative, typed, and allow-listed.

## Cross-Family Shared Algorithms

The following should be shared rather than reimplemented per family:

- parameter-domain validation
- trim-loop validation
- finite numeric validation
- transform composition
- stable identity canonicalization
- adaptive tessellation request normalization
- seam sampling agreement
- boundary reference normalization
- `.impress` payload encoding/decoding
- metadata splitting and inheritance
- diagnostic shape for unsupported operations

Spline-family common algorithms:

- knot vector validation
- basis function evaluation
- basis derivative evaluation
- adaptive curve/surface sampling
- control-net canonicalization

Path-family common algorithms:

- path evaluation
- frame transport
- profile placement
- twist/scale interpolation

Field-family common algorithms:

- bounded domain validation
- field sampling
- extraction tolerance policy
- gradient/normal estimation

## Tessellation Strategy

Each family needs a tessellation adapter under the shared tessellation boundary.

Analytic patches:

- planar: grid/trim boundary triangulation
- ruled: sample compatible guide curves and connect spans
- revolution: sample angle/profile with periodic seam handling

Spline patches:

- curvature/error adaptive sampling
- knot-aware sampling
- trim-aware triangulation

Sweep patches:

- sample path by curvature and frame change
- sample profile by curvature
- connect transported profile rings

Subdivision patches:

- deterministic subdivision level or error target
- limit surface sampling where implemented

Implicit patches:

- deterministic isosurface extraction inside declared bounds
- explicit quality/tolerance controls

## Seam And Adjacency Strategy

All families participate in seams through boundary uses.

Boundary comparison should never rely only on mesh vertices. It should compare
the surface-native boundary intent where possible:

- analytic boundary curve
- spline boundary curve
- sweep start/end/profile boundary
- subdivision boundary curve/edge chain
- implicit extracted boundary with declared approximation status

When exact comparison is impossible, the seam record must carry approximation
metadata and tolerances explicitly.

## Boolean And CSG Strategy

Booleans must become family-aware.

Initial exact support can land by family pair:

- planar/planar
- planar/ruled
- planar/revolution
- spline/spline
- analytic/spline
- sweep/spline
- implicit/analytic

Unsupported family pairs should refuse with a diagnostic naming:

- left family
- right family
- operation
- unsupported phase
- required future capability

They must not fallback to mesh as hidden execution.

## .impress Persistence Strategy

`.impress` schema must carry a typed patch payload per family.

Recommended family payload roots:

```text
PlanarSurfacePatch.geometry
RuledSurfacePatch.geometry
RevolutionSurfacePatch.geometry
BSplineSurfacePatch.geometry
NURBSSurfacePatch.geometry
SweepSurfacePatch.geometry
SubdivisionSurfacePatch.geometry
ImplicitSurfacePatch.geometry
```

Payloads must be declarative and deterministic. Runtime-only caches, tessellated
meshes, and executable callback functions do not belong in kernel geometry
payloads.

## Implementation Sequencing

The architecture target is no deferred families.

The maintained implementation truth is the patch-family capability matrix:

| Family | Phase | Capability Obligations |
| --- | --- | --- |
| planar | available | caps, planar primitives, trimmed faces, tessellation, `.impress` |
| ruled | available | extrude, loft, linear bridge surfaces, tessellation, `.impress` |
| revolution | available | rotate-extrude, revolved primitives, tessellation, `.impress` |
| B-spline | planned | surface record, evaluation, tessellation, `.impress` |
| NURBS | planned | rational surface record, evaluation, tessellation, `.impress` |
| sweep | planned | sweep record, frame policy, evaluation, tessellation, `.impress` |
| subdivision | planned | control cage, crease payload, evaluation, tessellation, `.impress` |
| implicit | planned | field-node payload, validation security, evaluation, tessellation, `.impress` |

The phase column is not an exclusion list. It is a staged implementation status
that downstream operations use to produce explicit diagnostics when a family is
not yet supported by that operation.

Implementation can still be staged:

1. Update architecture/spec manifest to remove deferral posture.
2. Add base records for every family with strict validation.
3. Add `.impress` payload contracts for every family.
4. Add evaluation and derivative support by family.
5. Add tessellation adapters by family.
6. Add seam/boundary support by family.
7. Add operation-specific consumption, including loft/booleans/sweeps.
8. Keep Spec 66 retired and keep the capability matrix current as families and
   operations land.

## Specification Manifest for Discovery

The following manifest uses the shared `specification-manifest-entry` template
from `/Users/k/Documents/Projects/.agents/process/templates/manifest-entry-template.md`.

Scores follow the shared policy:

- `25+`: split required before implementation
- `16-24`: explicit split review required
- `0-15`: small/cohesive if readiness fields are complete

Spec promotion status: final specification documents have been created for every candidate in this manifest.

### Candidate Spec: Patch Family Capability Matrix And Spec 66 Retirement

Discovery purpose:
- Replace deferred-family/exclusion posture with a first-class capability matrix
  that tracks staged support without declaring families out of scope.

Responsibilities:
- Functions/methods:
  - capability matrix generator or maintained table
  - Spec 66 retirement note
- Data structures/models:
  - patch family capability record
  - operation support phase record
- Dependencies/services:
  - existing Surface Specs 65-67
  - full patch family architecture
- Returns/outputs/signals:
  - capability matrix
  - retired/replaced deferred-family spec status
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current patch family docs/specs
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - revises/retire existing spec posture
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `project/release-0.1.0a/specifications/`
- Chosen defaults / parameters:
  - no patch family is architecturally deferred; unsupported operations report
    capability-aware diagnostics
- Test strategy:
  - documentation/spec review plus tests in family leaf specs
- Data ownership:
  - capability matrix owns support status truth
- Routes:
  - architecture to replacement specs
- Reuse/extraction decision:
  - revise existing specs; no code module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Matrix must distinguish "family exists" from "operation supports this family."

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.
- This can remain one replacement spec because it is a policy/matrix artifact.

### Candidate Spec: B-Spline Surface Patch Record And Validation

Discovery purpose:
- Define B-spline surface payload shape and validity rules before evaluation.

Responsibilities:
- Functions/methods:
  - B-spline patch constructor
  - knot/control-net validator
- Data structures/models:
  - `BSplineSurfacePatch`
  - knot vectors
  - control net
- Dependencies/services:
  - surface patch base contract
  - numerical basis utilities
- Returns/outputs/signals:
  - valid patch object
  - validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch base classes
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: B-spline basis utility if absent
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family module selected by implementation spec
- Chosen defaults / parameters:
  - B-spline and NURBS use separate patch classes over shared basis utilities
- Test strategy:
  - validation tests for degree, knot, domain, and control-net shape
- Data ownership:
  - patch payload owns knots/control net
- Routes:
  - constructor to patch record validation
- Reuse/extraction decision:
  - add reusable basis utility if no existing equivalent exists
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Evaluation is split because basis math and derivative behavior are separate implementation risks.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: record shape and validation are one constructor boundary.

### Candidate Spec: B-Spline Surface Evaluation And Derivatives

Discovery purpose:
- Define B-spline point evaluation, derivative evaluation, and bounded basis reuse.

Responsibilities:
- Functions/methods:
  - basis evaluation
  - point evaluation
  - derivative evaluation
- Data structures/models:
  - basis cache
  - evaluation result
- Dependencies/services:
  - B-spline patch record
  - numerical basis utilities
  - tessellation
- Returns/outputs/signals:
  - evaluated point/derivative
  - evaluation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: validated B-spline patch record
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: none beyond basis utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family evaluator implementation
- Chosen defaults / parameters:
  - basis evaluation reuses validated knot/control data
- Test strategy:
  - point, derivative, boundary, and tessellation sampling tests
- Data ownership:
  - patch evaluator owns runtime evaluation; patch payload owns data
- Routes:
  - patch API to evaluator to tessellation
- Reuse/extraction decision:
  - add to patch family module and shared basis utility
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Derivative behavior should not be invented independently by tessellation.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: evaluation and derivatives are one numeric evaluator boundary.

### Candidate Spec: NURBS Surface Patch

Discovery purpose:
- Define rational NURBS surface payload and evaluation as an extension over the
  shared B-spline basis infrastructure.

Responsibilities:
- Functions/methods:
  - NURBS patch constructor
  - rational evaluation
  - weighted derivative evaluation
- Data structures/models:
  - `NURBSSurfacePatch`
  - weight grid
  - rational control net view
- Dependencies/services:
  - B-spline basis utilities
  - surface patch base contract
  - tessellation
- Returns/outputs/signals:
  - evaluated point/derivative
  - weight validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: B-spline basis utilities
  - Additions to existing reusable library/module: surface patch family module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds new patch family
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - rational evaluation should share cached basis values
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family module selected by implementation spec
- Chosen defaults / parameters:
  - NURBS is a separate patch class over shared B-spline basis utilities
- Test strategy:
  - unit tests for rational validation, evaluation, derivative behavior,
    tessellation, and `.impress` payload
- Data ownership:
  - patch payload owns knots/control net/weights
- Routes:
  - `SurfacePatch` API to rational evaluator to tessellation
- Reuse/extraction decision:
  - reuse B-spline basis; add family class
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Weight zero/negative policy must be explicit; default is reject non-positive
  weights unless a later spec allows signed rational forms.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: rational payload and rational evaluation are inseparable for
  a usable NURBS patch.

### Candidate Spec: Sweep Surface Patch Payload And Frame Policy

Discovery purpose:
- Define sweep payload shape, profile/path references, and frame transport policy.

Responsibilities:
- Functions/methods:
  - sweep patch constructor
  - frame policy validator
- Data structures/models:
  - `SweepSurfacePatch`
  - profile topology reference
  - path curve reference
  - frame policy
- Dependencies/services:
  - topology paths
  - curve evaluators
- Returns/outputs/signals:
  - valid sweep payload
  - frame policy diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: topology path records where available
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: frame transport utility if absent
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family module selected by implementation spec
- Chosen defaults / parameters:
  - V1 stores both profile topology and path curve reference when both are available
- Test strategy:
  - validation tests for profile/path references and frame policy
- Data ownership:
  - sweep patch owns profile/path references and frame policy
- Routes:
  - constructor to frame policy validation
- Reuse/extraction decision:
  - reuse topology path records; create frame utility only if needed
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Evaluation is split because frame transport and sampling carry hidden numeric behavior.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 4 x 1 = 4
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: payload and frame policy are one authoring contract.

### Candidate Spec: Sweep Surface Evaluation And Tessellation

Discovery purpose:
- Define profile/path evaluation and tessellation behavior for sweep patches.

Responsibilities:
- Functions/methods:
  - frame transport evaluator
  - profile/path evaluator
  - sweep tessellation adapter
- Data structures/models:
  - swept surface evaluation
  - frame discontinuity diagnostic
- Dependencies/services:
  - sweep patch payload
  - curve evaluators
  - tessellation
- Returns/outputs/signals:
  - evaluated swept surface
  - tessellated mesh
  - discontinuity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: sweep payload and path records
  - Additions to existing reusable library/module: patch family module, tessellation
  - New reusable library/module to create: none beyond frame utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - evaluator/tessellation implementation
- Chosen defaults / parameters:
  - tessellation samples are request-driven, not modeled truth
- Test strategy:
  - tests for straight, curved, and discontinuous-frame sweeps
- Data ownership:
  - sweep evaluator owns runtime samples; tessellation owns mesh output
- Routes:
  - sweep evaluator to tessellation adapter
- Reuse/extraction decision:
  - add to patch family and tessellation modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Sweep modeling helpers should remain separate from this patch evaluator spec.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: evaluation and tessellation share one request-driven sampling boundary.

### Candidate Spec: Subdivision Surface Control Cage And Crease Payload

Discovery purpose:
- Define subdivision patch control cage, crease/sharpness data, and payload validation.

Responsibilities:
- Functions/methods:
  - subdivision patch constructor
  - control cage validator
  - crease validator
- Data structures/models:
  - `SubdivisionSurfacePatch`
  - control cage
  - crease/sharpness data
- Dependencies/services:
  - surface patch base
  - subdivision evaluator utility
- Returns/outputs/signals:
  - valid subdivision payload
  - validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch base
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: subdivision evaluator utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family module selected by implementation spec
- Chosen defaults / parameters:
  - V1 payload records finite-level subdivision intent and sharpness metadata
- Test strategy:
  - validation tests for cage topology and crease data
- Data ownership:
  - patch payload owns control cage and subdivision parameters
- Routes:
  - constructor to cage/crease validation
- Reuse/extraction decision:
  - create reusable subdivision evaluator utility
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Evaluation level and tessellation are split because they are approximation policy.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: cage and crease payload are one subdivision input contract.

### Candidate Spec: Subdivision Surface Evaluation And Tessellation

Discovery purpose:
- Define Catmull-Clark refinement, deterministic finite-level evaluation, and tessellation approximation metadata.

Responsibilities:
- Functions/methods:
  - finite-level evaluator
  - Catmull-Clark refinement
  - subdivision tessellation adapter
- Data structures/models:
  - refinement result
  - approximation metadata
- Dependencies/services:
  - subdivision patch payload
  - tessellation
  - evaluator utility
- Returns/outputs/signals:
  - evaluated subdivision surface
  - tessellated approximation
  - refinement diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: subdivision payload
  - Additions to existing reusable library/module: patch family module, tessellation
  - New reusable library/module to create: none beyond evaluator utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - evaluator/tessellation implementation
- Chosen defaults / parameters:
  - finite-level Catmull-Clark evaluation carries approximation metadata
- Test strategy:
  - refinement level, crease behavior, tessellation, and metadata tests
- Data ownership:
  - evaluator owns refinement; tessellation owns mesh output
- Routes:
  - subdivision evaluator to tessellation adapter
- Reuse/extraction decision:
  - add to evaluator utility and tessellation module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Limit exactness is not required for V1 but must not be hidden.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: refinement and tessellation are one approximation execution boundary.

### Candidate Spec: Implicit Field Node Payload

Discovery purpose:
- Define the allow-listed declarative field node payload shape for implicit surfaces.

Responsibilities:
- Functions/methods:
  - field node schema builder
  - implicit patch payload constructor
- Data structures/models:
  - allow-listed field node
  - `ImplicitSurfacePatch` payload
- Dependencies/services:
  - patch family module
  - field validation
- Returns/outputs/signals:
  - typed field payload
  - unsupported node diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch base
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: declarative field validator
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - patch family payload implementation
- Chosen defaults / parameters:
  - V1 allows only typed allow-listed declarative field nodes
- Test strategy:
  - tests for allowed nodes and unsupported node refusal
- Data ownership:
  - patch payload owns declarative field tree
- Routes:
  - payload constructor to field validator
- Reuse/extraction decision:
  - create reusable field validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Validation security is split to keep payload shape separate from safety rules.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: field node payload is one schema boundary.

### Candidate Spec: Implicit Field Validation Security

Discovery purpose:
- Define safe validation rules for implicit fields, including no executable code and bounded tree size.

Responsibilities:
- Functions/methods:
  - field security validator
  - bounded field tree validator
- Data structures/models:
  - validation diagnostic
  - field safety policy
- Dependencies/services:
  - field validator
  - `.impress` payload security rules
- Returns/outputs/signals:
  - safe field acceptance
  - unsafe field refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: field node payload once defined
  - Additions to existing reusable library/module: field validator
  - New reusable library/module to create: none beyond validator
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - validation/security behavior
- Chosen defaults / parameters:
  - rejects executable code, dynamic imports, and unbounded field trees
- Test strategy:
  - tests for unsafe payloads and bounded validation
- Data ownership:
  - validator owns safety before evaluation or object construction
- Routes:
  - payload load/constructor to validator
- Reuse/extraction decision:
  - add to reusable field validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This is security-sensitive enough to remain separate from normal payload schema.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: field security validation is one safety boundary.

### Candidate Spec: Implicit Field Evaluation

Discovery purpose:
- Define bounded runtime evaluation of declarative implicit fields.

Responsibilities:
- Functions/methods:
  - declarative field evaluator
  - domain evaluator
- Data structures/models:
  - field evaluation result
  - evaluation domain
- Dependencies/services:
  - field validator
  - patch family module
- Returns/outputs/signals:
  - field value
  - evaluation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: validated field payload
  - Additions to existing reusable library/module: patch family module
  - New reusable library/module to create: field evaluator
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - evaluator implementation
- Chosen defaults / parameters:
  - evaluator consumes declarative data only
- Test strategy:
  - tests for field values, domains, and invalid evaluation cases
- Data ownership:
  - patch owns field data; evaluator owns runtime interpretation
- Routes:
  - field payload to evaluator
- Reuse/extraction decision:
  - create reusable field evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Tessellation bounds are split because they involve approximation and performance policy.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: field evaluation is one runtime interpreter boundary.

### Candidate Spec: Implicit Tessellation Bounds And Approximation Metadata

Discovery purpose:
- Define bounded sampling/tessellation and approximation metadata for implicit surfaces.

Responsibilities:
- Functions/methods:
  - bounded sampling adapter
  - implicit tessellation adapter
- Data structures/models:
  - tessellated approximation
  - approximation metadata
- Dependencies/services:
  - field evaluator
  - tessellation
- Returns/outputs/signals:
  - tessellated mesh
  - bounds diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation request contract
  - Additions to existing reusable library/module: tessellation and patch family module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tessellation adapter implementation
- Chosen defaults / parameters:
  - no code execution; bounded declarative evaluator only
- Test strategy:
  - tests for bounded sampling, refusal of unbounded domains, and metadata
- Data ownership:
  - tessellation owns mesh approximation
- Routes:
  - field evaluator to tessellation adapter
- Reuse/extraction decision:
  - add to tessellation module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Approximation metadata must be explicit so implicit support is not mistaken for exact B-rep evaluation.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: bounded implicit tessellation is one approximation boundary.

### Candidate Spec: Cross-Family Tessellation Adapters

Discovery purpose:
- Ensure every surface patch family tessellates through one request contract
  without family-specific mesh execution leaking into modeling APIs.

Responsibilities:
- Functions/methods:
  - family tessellation adapter registry
  - per-family sampling hooks
  - seam-aware adapter entrypoint
- Data structures/models:
  - tessellation request
  - family adapter record
- Dependencies/services:
  - `tessellation.py`
  - all patch family modules
- Returns/outputs/signals:
  - mesh consumer payload
  - tessellation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `TessellationRequest`
  - Additions to existing reusable library/module: `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes tessellation dispatch
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - adapter dispatch and sampling must honor quality bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - tessellation adapter dispatch is family-neutral and request-driven
- Test strategy:
  - tests for every family adapter, seam interaction, and unsupported family
    diagnostics
- Data ownership:
  - tessellation owns mesh output; patch family owns evaluation
- Routes:
  - `SurfaceBody` to patch adapter to mesh result
- Reuse/extraction decision:
  - add to existing tessellation module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec should come after first family records or be implemented in staged
  family increments.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: adapter registry and per-family hooks are one tessellation
  boundary, but implementation may land family by family.

### Candidate Spec: Cross-Family Seam Boundary Participation

Discovery purpose:
- Define how non-planar and advanced patch families participate in seams,
  boundary references, adjacency, and continuity metadata.

Responsibilities:
- Functions/methods:
  - boundary extraction
  - seam compatibility check
  - continuity classification
- Data structures/models:
  - family boundary descriptor
  - seam participation record
  - continuity metadata
- Dependencies/services:
  - seam/adjacency architecture
  - patch family evaluators
- Returns/outputs/signals:
  - seam validation result
  - adjacency update
  - unsupported continuity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and boundary records
  - Additions to existing reusable library/module: surface seam/adjacency module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes seam validation for new families
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - boundary comparison must avoid relying only on dense mesh sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - surface seam/adjacency module selected by implementation spec
- Chosen defaults / parameters:
  - compare analytic/parametric boundaries first; approximation metadata required
    when exact comparison is unavailable
- Test strategy:
  - tests for compatible/incompatible family seams, continuity metadata, and
    tessellation watertightness
- Data ownership:
  - seams own shared boundary truth; patches own boundary evaluation
- Routes:
  - patch boundary descriptor to seam validation to tessellation
- Reuse/extraction decision:
  - add to existing seam/adjacency logic
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Exactness policy may need per-family approximation metadata.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: all family seam behavior uses the same seam contract; per-
  family fixtures can be test cases.

### Candidate Spec: Family-Aware Boolean Eligibility And Diagnostics

Discovery purpose:
- Define how surface booleans classify family support and refuse unsupported
  family pairs without falling back to mesh.

Responsibilities:
- Functions/methods:
  - boolean family eligibility check
  - unsupported family diagnostic builder
- Data structures/models:
  - family pair support matrix
  - unsupported phase diagnostic
- Dependencies/services:
  - surface boolean module
  - patch family capability matrix
- Returns/outputs/signals:
  - eligibility result
  - unsupported result
  - required future capability
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SurfaceBooleanResult`
  - Additions to existing reusable library/module: `csg.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes unsupported boolean result behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - eligibility should be structural and avoid expensive sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported family pairs return explicit diagnostics; no mesh fallback
- Test strategy:
  - tests for supported, unsupported, and mixed-family boolean eligibility
- Data ownership:
  - boolean module owns operation eligibility; capability matrix owns family
    status
- Routes:
  - boolean API to eligibility check to result/diagnostic
- Reuse/extraction decision:
  - add to existing CSG module and capability matrix
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The matrix should name the unsupported phase, not just the family pair.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: eligibility and diagnostics are one refusal boundary and
  should be specified together.

### Candidate Spec: `.impress` Analytic Patch Payloads

Discovery purpose:
- Define `.impress` payloads for planar, ruled, and revolution patch families.

Responsibilities:
- Functions/methods:
  - analytic patch payload encoder
  - analytic patch payload decoder
- Data structures/models:
  - planar payload
  - ruled/revolution payload
- Dependencies/services:
  - `.impress` codec
  - analytic patch modules
- Returns/outputs/signals:
  - encoded analytic payload
  - decoded analytic patch
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch family records once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown payload data
- Performance-sensitive behavior:
  - validation avoids dense geometry evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - family payloads are typed and versioned inside the `.impress` schema
- Test strategy:
  - round-trip and invalid payload tests for the named family group
- Data ownership:
  - `.impress` payload owns serialized surface truth
- Routes:
  - file codec to patch family codec
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Analytic payloads should stay simple and avoid inheriting spline payload complexity.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one family payload group.

### Candidate Spec: `.impress` Spline Patch Payloads

Discovery purpose:
- Define `.impress` payloads for B-spline and NURBS patch families.

Responsibilities:
- Functions/methods:
  - spline patch payload encoder
  - spline patch payload decoder
- Data structures/models:
  - B-spline payload
  - NURBS payload
- Dependencies/services:
  - `.impress` codec
  - spline patch modules
- Returns/outputs/signals:
  - encoded spline payload
  - decoded spline patch
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch family records once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown payload data
- Performance-sensitive behavior:
  - validation avoids dense geometry evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - family payloads are typed and versioned inside the `.impress` schema
- Test strategy:
  - round-trip and invalid payload tests for the named family group
- Data ownership:
  - `.impress` payload owns serialized surface truth
- Routes:
  - file codec to patch family codec
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Spline payloads carry knot/control-net/weight validation separate from analytic payloads.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one family payload group.

### Candidate Spec: `.impress` Sweep Patch Payload

Discovery purpose:
- Define `.impress` payloads for sweep patch profile/path/frame records.

Responsibilities:
- Functions/methods:
  - sweep payload encoder
  - sweep payload decoder
- Data structures/models:
  - sweep profile/path payload
  - frame policy payload
- Dependencies/services:
  - `.impress` codec
  - sweep patch module
- Returns/outputs/signals:
  - encoded sweep payload
  - decoded sweep patch
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch family records once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown payload data
- Performance-sensitive behavior:
  - validation avoids dense geometry evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - family payloads are typed and versioned inside the `.impress` schema
- Test strategy:
  - round-trip and invalid payload tests for the named family group
- Data ownership:
  - `.impress` payload owns serialized surface truth
- Routes:
  - file codec to patch family codec
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Sweep payloads depend on profile/path reference policy, not subdivision cage policy.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one family payload group.

### Candidate Spec: `.impress` Subdivision Patch Payload

Discovery purpose:
- Define `.impress` payloads for subdivision control cage, crease, and approximation metadata.

Responsibilities:
- Functions/methods:
  - subdivision payload encoder
  - subdivision payload decoder
- Data structures/models:
  - control cage payload
  - crease/approximation payload
- Dependencies/services:
  - `.impress` codec
  - subdivision patch module
- Returns/outputs/signals:
  - encoded subdivision payload
  - decoded subdivision patch
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch family records once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown payload data
- Performance-sensitive behavior:
  - validation avoids dense geometry evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - family payloads are typed and versioned inside the `.impress` schema
- Test strategy:
  - round-trip and invalid payload tests for the named family group
- Data ownership:
  - `.impress` payload owns serialized surface truth
- Routes:
  - file codec to patch family codec
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Subdivision payloads need bounded cage validation and approximation metadata distinct from sweep payloads.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one family payload group.

### Candidate Spec: `.impress` Implicit Patch Payload Security

Discovery purpose:
- Define `.impress` serialization and refusal rules for implicit patch payloads.

Responsibilities:
- Functions/methods:
  - implicit payload encoder
  - implicit payload decoder
  - executable-payload refusal
- Data structures/models:
  - implicit payload
  - allow-listed field node
  - security diagnostic
- Dependencies/services:
  - `.impress` codec
  - implicit field validator
- Returns/outputs/signals:
  - encoded implicit payload
  - decoded implicit patch
  - unsafe payload refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit field validator once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - rejects executable code, dynamic imports, and unknown field nodes
- Performance-sensitive behavior:
  - validation bounds field tree size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - implicit payloads are declarative allow-listed data only
- Test strategy:
  - tests for safe round-trip and unsafe payload refusal
- Data ownership:
  - `.impress` payload owns serialized field data; evaluator owns runtime interpretation
- Routes:
  - file codec to implicit field validator
- Reuse/extraction decision:
  - add to `.impress` codec module and reuse field validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This must not wait until after general implicit evaluation, because serialization security is part of the payload contract.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: implicit serialization security is one payload boundary.

## Open Decisions

- Whether B-spline and NURBS share one class with rational optionality or use
  separate patch classes over shared basis infrastructure.
- Whether sweep patches store profiles as 2D topology, 3D curves, or both.
- Whether subdivision V1 requires limit evaluation or allows deterministic
  finite-level evaluation with explicit approximation metadata.
- Which declarative implicit field nodes are allowed in V1.
- How much exact boolean support must exist for each family pair before the
  family is called complete.
- Whether `.impress` V1 schema expands immediately for all families or versions
  family payloads independently.

## Change History

- 2026-05-26: Further split high-scoring manifest entries where review exposed hidden payload-family boundaries.
- 2026-05-26: Split all manifest candidates that scored 25+ into smaller assessed candidates for spec promotion.
- 2026-05-26: Replaced the lightweight specification list with a
  template-assessed Specification Manifest for Discovery for full patch-family
  implementation.
- 2026-05-26: Initial full patch-family architecture. Created after deciding
  that deferred patch families are not acceptable and must become first-class
  surface-native targets.
