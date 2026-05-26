# .impress Surface-Native File Format Architecture

## Overview

`.impress` is the native durable file format for Impression surface-first
modeling data.

The purpose of `.impress` is to persist the same surface-native truth that the
runtime uses for modeling:

```text
SurfaceBody
-> SurfaceShell
-> SurfacePatch
-> TrimLoop
-> SurfaceSeam
-> SurfaceAdjacencyRecord
-> transforms, metadata, identity, and authoring provenance
```

The file format is not a tessellated export format. It is the surface-native
model document format that can later be tessellated for preview, STL export,
analysis, or other mesh-only consumers.

The first implementation should be a versioned JSON document with extension
`.impress`. A later binary or package form may reuse the same logical schema,
but the first target is inspectable, deterministic, source-control friendly,
and easy to test.

## Relationship To Existing Architecture

This architecture extends:

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Surface Mesh Decommission Architecture](surface-mesh-decommission-architecture.md)

Those documents define the in-memory surface kernel and the expectation that
`SurfaceBody` is canonical modeling truth.

This document defines how that truth is written to disk and loaded back without
turning into mesh data.

## Design Goals

`.impress` must:

- round-trip V1 `SurfaceBody` objects without geometry loss
- preserve shell, patch, trim, seam, adjacency, transform, metadata, and stable
  identity inputs
- preserve authoring and diagnostic metadata when it is part of the surface
  model
- remain deterministic enough for stable hashes and meaningful diffs
- fail explicitly on unsupported patch families or schema versions
- be simple enough to implement before industry interchange support

`.impress` must not:

- store only tessellated mesh output
- silently drop trims, seams, adjacency, metadata, or transforms
- use mesh fallback as a load strategy
- claim STEP/IGES/BREP compatibility before an adapter exists

## Format Choice

### V1 Physical Encoding

V1 `.impress` files should be UTF-8 JSON.

The document root is a typed object:

```json
{
  "format": "impress",
  "schema_version": "1.0",
  "producer": {
    "name": "impression",
    "version": "0.1.0a"
  },
  "units": "model",
  "document": {
    "bodies": []
  }
}
```

The schema should use plain JSON-compatible values:

- objects
- arrays
- strings
- booleans
- null
- finite numbers

NumPy arrays are serialized as nested arrays. On load, constructors normalize
those arrays back into the runtime data types.

### Why JSON First

JSON is the right first format because the surface kernel already has
`canonical_payload()` and stable identity support.

JSON is also:

- human inspectable
- easy to regression test
- easy to diff in review
- easy to generate from examples
- independent of heavyweight CAD exchange dependencies

### Future Encodings

The logical `.impress` schema should not depend on JSON forever.

If large models require it, later versions may define:

- compressed `.impress` packages
- binary array sidecars
- content-addressed geometry chunks
- external asset references

Those should be additive evolutions of the same logical document model, not a
replacement for the V1 schema.

## Core Components

### SurfaceBodyStore

`SurfaceBodyStore` is the document-level container for persisted surface
bodies.

It owns:

- schema version
- producer metadata
- unit policy
- one or more `SurfaceBody` payloads
- optional document metadata
- optional named object table
- optional diagnostic bundle

It does not own:

- tessellated mesh caches as source truth
- preview camera state unless explicitly placed in consumer metadata
- external CAD import/export adapter state unless explicitly namespaced

Recommended record shape:

```text
SurfaceBodyStore
  format: "impress"
  schema_version: "1.0"
  producer: ProducerInfo
  units: str
  bodies: tuple[SurfaceBody]
  metadata: dict
```

### SurfaceBody Payload

Each body payload corresponds to one runtime `SurfaceBody`.

It includes:

- body id or stable identity
- shell payloads
- body transform matrix
- body metadata

The body payload should be based on `SurfaceBody.canonical_payload()` but must
also include enough type information to reconstruct runtime objects.

### SurfaceShell Payload

Each shell payload includes:

- shell id or stable identity
- ordered patch payloads
- shell connectivity flag
- seam payloads
- adjacency payloads
- shell transform matrix
- shell metadata

Patch order is durable. Shell traversal after load must match shell traversal
before save.

### SurfacePatch Payload

Each patch payload includes:

- patch kind
- patch family
- parameter domain
- capability flags
- trim loops
- transform matrix
- metadata
- geometry payload

The complete target patch family set is defined by the
[Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md).

The initial runtime already has these families:

- `planar`
- `ruled`
- `revolution`

The following families are no longer acceptable as architectural deferrals.
They require `.impress` payload schemas and loaders before the native format can
claim full surface-body coverage:

- `nurbs`
- `bspline`
- `subdivision`
- `implicit`
- `sweep`

Until those family payload schemas are implemented, loading a file that
contains one of those families must refuse explicitly with a family-specific
diagnostic. That refusal is a temporary implementation gap, not the target
architecture.

### Trim Payload

Each trim loop payload includes:

- category: `outer` or `inner`
- ordered UV points
- normalized orientation

The loader must validate trim loops against the patch domain. Invalid trim
payloads are invalid files, not recoverable mesh fallback cases.

### Seam Payload

Each seam payload includes:

- seam id
- participating boundary references
- continuity classification
- metadata

Seam identity must be stable enough that adjacency can refer to seams after a
round trip.

### Adjacency Payload

Each adjacency payload includes:

- source boundary reference
- optional target boundary reference
- seam id where applicable
- relationship classification
- continuity metadata

Adjacency must not be inferred silently during load. The loader may validate
adjacency, but it should not invent missing adjacency unless a future repair
mode explicitly asks for that behavior.

## Data Flow

### Save Flow

```text
SurfaceBody or iterable[SurfaceBody]
-> SurfaceBodyStore
-> canonical, schema-versioned payload
-> deterministic JSON encoding
-> .impress file
```

The save path should:

1. Validate body, shell, patch, trim, seam, and adjacency invariants.
2. Convert runtime objects to schema payloads.
3. Canonicalize finite numeric arrays into JSON arrays.
4. Write deterministic JSON with stable key ordering.
5. Optionally verify that loading the written payload would succeed.

### Load Flow

```text
.impress file
-> JSON payload
-> schema/version validation
-> SurfaceBodyStore
-> SurfaceBody runtime objects
```

The load path should:

1. Parse JSON.
2. Validate `format == "impress"`.
3. Validate supported `schema_version`.
4. Validate all required fields.
5. Reconstruct typed runtime objects.
6. Recompute identities rather than trusting stored hashes blindly.
7. Report mismatch diagnostics if stored identities disagree with computed
   identities.

## Identity And Determinism

Runtime stable identity is computed from canonical payloads.

`.impress` may store identity fields for diagnostics and fast comparison, but
stored identities are not the source of truth. The loader should recompute
identity after object reconstruction.

Identity behavior:

- identical body payloads should produce identical stable identities
- irrelevant JSON formatting must not affect identity
- load/save/load round trips must preserve stable identities
- metadata included in canonical payload affects identity unless explicitly
  placed outside kernel metadata

## Metadata Policy

Metadata follows the existing kernel/consumer split:

- kernel metadata is part of modeling truth
- consumer metadata is for preview/export/user-interface consumers

`.impress` stores both, but loaders and validators must preserve the boundary.

Unknown metadata keys are allowed when they are valid JSON-compatible values.
Unknown kernel object types are not allowed.

## Error Policy

The file format should fail early and clearly.

Required refusal cases:

- unsupported schema version
- unsupported patch family
- unknown patch kind
- invalid transform matrix
- invalid parameter domain
- invalid trim loop
- missing shell or body content
- broken seam or adjacency reference
- non-finite numeric value
- identity mismatch when strict validation is enabled

The loader should not:

- tessellate as a recovery strategy
- drop unsupported patches
- ignore broken adjacency
- silently convert unknown geometry into mesh payloads

## Industry Format Relationship

`.impress` is Impression's native format.

STEP, IGES, BREP, JT, and other CAD formats are interchange formats. They are
valuable, but they should be implemented as adapters around the surface-native
store rather than as the internal persistence model.

### V1 Boundary Policy

`.impress` V1 does not promise STEP, IGES, BREP, JT, STL, or imported-mesh
compatibility. The only V1 persistence promise is native `SurfaceBody` truth as
defined by this architecture and the active `.impress` schema version.

The V1 reader must refuse industry-format payloads, mesh payloads, executable
implicit/code payloads, and unknown adapter records when they appear in the
native document root. Refusal is the correct behavior because accepting those
records would imply an adapter contract that has not been specified.

The V1 writer must not emit industry-interchange adapter state. It may emit only
surface-native payloads, metadata that is valid JSON data, and schema fields
defined by this architecture.

Recommended adapter posture:

```text
.impress <-> SurfaceBodyStore <-> SurfaceBody
                                 |
                                 +-> STEP export/import adapter
                                 +-> IGES export/import adapter
                                 +-> mesh export adapter
```

The native format should preserve Impression-specific information that
industry formats may not represent cleanly:

- authored topology rails
- diagnostic metadata
- generator provenance
- patch-family capability flags
- exact release-schema assumptions

### Imported Mesh Classification

Imported meshes are tool data, not `.impress` surface truth.

An imported mesh may be represented in future work as an explicit imported mesh
record owned by the mesh toolchain. That record must be classified separately
from `SurfaceBody`, must not masquerade as a surface patch, and must not become
a hidden fallback for failed surface-body loading.

V1 `.impress` has no imported mesh record. If a file contains a root-level mesh
payload, an embedded STL-like payload, or a patch whose geometry is merely a
mesh, the loader must reject the file with an unsupported payload diagnostic.

### External Adapter Placeholder

Future external adapters should attach outside the native persistence path:

```text
External CAD file
-> adapter parser / validator
-> SurfaceBodyStore conversion report
-> SurfaceBody runtime objects
-> optional .impress save
```

Adapter placeholders are documentation-only in V1. No adapter placeholder in a
`.impress` file is executable, dynamically imported, or interpreted as code.
When adapter support is eventually specified, it must define:

- supported source/target format and version
- data-only adapter payload schema
- validation and refusal diagnostics
- conversion report shape
- lossiness and unsupported-entity policy
- security limits for external payloads

Until then, STEP/IGES/BREP/JT import/export and STL mesh import remain outside
the `.impress` native file contract.

## Security And Robustness

The loader should treat `.impress` files as untrusted input.

Required safeguards:

- no code execution during load
- no dynamic imports based on file payload
- finite numeric validation
- bounded array sizes where practical
- schema validation before object construction
- clear failure diagnostics

## Public API Boundary

Recommended first public API:

```text
save_impress(path, body_or_store) -> None
load_impress(path) -> SurfaceBodyStore
surface_body_to_impress_payload(body) -> dict
surface_body_from_impress_payload(payload) -> SurfaceBody
```

Convenience API:

```text
save_surface_body(path, body) -> None
load_surface_body(path) -> SurfaceBody
```

The convenience API is appropriate when the file contains exactly one body.
The store API is authoritative when a document can contain multiple bodies.

## Specification Manifest for Discovery

The following manifest uses the shared `specification-manifest-entry` template
from `/Users/k/Documents/Projects/.agents/process/templates/manifest-entry-template.md`.

Scores follow the shared policy:

- `25+`: split required before implementation
- `16-24`: explicit split review required
- `0-15`: small/cohesive if readiness fields are complete

Spec promotion status: final specification documents have been created for every candidate in this manifest.

### Candidate Spec: `.impress` Document Root And Schema Version

Discovery purpose:
- Define the minimal root envelope and schema-version contract for `.impress` files.

Responsibilities:
- Functions/methods:
  - document root constructor
  - schema version validator
- Data structures/models:
  - document root
  - schema version
- Dependencies/services:
  - `.impress` persistence module
  - JSON codec
- Returns/outputs/signals:
  - valid root envelope
  - unsupported schema diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: standard JSON/path utilities
  - Additions to existing reusable library/module: future `.impress` module
  - New reusable library/module to create: `.impress` persistence module
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
  - candidate `.impress` persistence module
- Chosen defaults / parameters:
  - schema version is required and explicit
- Test strategy:
  - root/schema validation tests
- Data ownership:
  - document root owns file-level metadata
- Routes:
  - public save/load to root validator
- Reuse/extraction decision:
  - create `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Units are split out because they are a policy decision, not just envelope shape.

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
- Review for split. Cohesion reason: root and schema version are one file-envelope contract.

### Candidate Spec: `.impress` Units And Root Validation Policy

Discovery purpose:
- Define unit defaults, root-level refusal behavior, and bounded root validation for `.impress`.

Responsibilities:
- Functions/methods:
  - units validator
  - root security validator
- Data structures/models:
  - units record
  - root validation diagnostic
- Dependencies/services:
  - `.impress` persistence module
  - schema validator
- Returns/outputs/signals:
  - normalized units metadata
  - refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: root validation from document envelope
  - Additions to existing reusable library/module: future `.impress` module
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
  - future `.impress` persistence module
- Chosen defaults / parameters:
  - units default to symbolic `unitless` unless provided
- Test strategy:
  - unit default, invalid unit, unsafe root payload tests
- Data ownership:
  - root validator owns file-level refusal before object construction
- Routes:
  - load path to root validator before payload decode
- Reuse/extraction decision:
  - add to `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Unit conversion is not included; this only preserves declared units.

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
- Review for split. Cohesion reason: units and root validation are one load/save policy boundary.

### Candidate Spec: `.impress` SurfaceBodyStore And Identity Policy

Discovery purpose:
- Define the persisted body store shape and stable identity requirements.

Responsibilities:
- Functions/methods:
  - store validation
  - body reference validation
- Data structures/models:
  - `SurfaceBodyStore` payload
  - stored stable identity record
  - body entry record
- Dependencies/services:
  - `SurfaceBody`
  - `SurfaceShell`
  - future `.impress` persistence module
- Returns/outputs/signals:
  - validated body store
  - identity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface body records
  - Additions to existing reusable library/module: future `.impress` module
  - New reusable library/module to create: none beyond `.impress` module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes `.impress` files
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown store payloads
- Performance-sensitive behavior:
  - store validation linear in body count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` persistence module
- Chosen defaults / parameters:
  - multi-body store is allowed by default; stable identities are required
- Test strategy:
  - tests for single/multi-body stores, missing identities, duplicate identities, and ordering
- Data ownership:
  - `.impress` document owns persisted body store
- Routes:
  - save/load API to body store validator
- Reuse/extraction decision:
  - add to `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Authoring topology rails are not part of this V1 store unless a later modeling document layer adds them.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
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
- Review for split. Cohesion reason: store shape and identity policy are one persisted-truth boundary.

### Candidate Spec: `.impress` Body And Shell Payload Codec

Discovery purpose:
- Define encoding and decoding for body and shell payloads through public constructors.

Responsibilities:
- Functions/methods:
  - encode body/shell
  - decode body/shell
  - constructor validation bridge
- Data structures/models:
  - body payload
  - shell payload
- Dependencies/services:
  - surface model
  - `.impress` root
- Returns/outputs/signals:
  - encoded body/shell payload
  - decoded body/shell objects
  - validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface constructors
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none beyond codec module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` payloads
- Security/privacy-sensitive behavior:
  - refuses unsafe/unknown payload data
- Performance-sensitive behavior:
  - codec linear in body/shell count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` codec module
- Chosen defaults / parameters:
  - decode validates through public constructors
- Test strategy:
  - round-trip and invalid payload tests for body and shell fields
- Data ownership:
  - codec owns serialized form; surface constructors own runtime invariants
- Routes:
  - save/load API to codec to surface constructors
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Body/shell ordering must preserve deterministic traversal.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Review for split. Cohesion reason: body and shell codec are one container-level codec boundary.

### Candidate Spec: `.impress` Patch Payload Codec

Discovery purpose:
- Define patch payload encoding/decoding for base patch fields and family dispatch.

Responsibilities:
- Functions/methods:
  - encode patch
  - decode patch
  - patch constructor validation bridge
- Data structures/models:
  - patch payload
  - family dispatch record
- Dependencies/services:
  - surface model
  - patch family modules
  - `.impress` root
- Returns/outputs/signals:
  - encoded patch payload
  - decoded patch object
  - patch diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch constructors
  - Additions to existing reusable library/module: `.impress` codec
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
  - future `.impress` codec module
- Chosen defaults / parameters:
  - decode validates through public patch constructors
- Test strategy:
  - patch round-trip and invalid family tests
- Data ownership:
  - codec owns serialized form; patch constructors own invariants
- Routes:
  - save/load to patch codec
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Advanced family payload details remain in family-specific payload specs.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Review for split. Cohesion reason: patch payload dispatch and validation are one codec boundary.

### Candidate Spec: `.impress` Trim Payload Codec

Discovery purpose:
- Define trim-loop payload encoding/decoding independent of patch-family geometry.

Responsibilities:
- Functions/methods:
  - encode trim
  - decode trim
  - trim constructor validation bridge
- Data structures/models:
  - trim payload
  - trim orientation/category record
- Dependencies/services:
  - surface model
  - `.impress` root
- Returns/outputs/signals:
  - encoded trim payload
  - decoded trim object
  - trim diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: trim loop constructors
  - Additions to existing reusable library/module: `.impress` codec
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
  - future `.impress` codec module
- Chosen defaults / parameters:
  - trims validate orientation/category without evaluating dense geometry
- Test strategy:
  - trim round-trip, invalid orientation, and missing-loop tests
- Data ownership:
  - codec owns serialized form; trim constructors own invariants
- Routes:
  - save/load to trim codec after patch decode
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Trim codec is separated because trim validity can fail even when patch payloads are valid.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Review for split. Cohesion reason: trim payload is one non-family-specific boundary contract.

### Candidate Spec: `.impress` Seam Payload Codec

Discovery purpose:
- Define seam payload encoding/decoding and boundary-reference validation.

Responsibilities:
- Functions/methods:
  - encode seam
  - decode seam
  - boundary reference validation
- Data structures/models:
  - seam payload
  - boundary reference payload
- Dependencies/services:
  - seam model
  - `.impress` root
- Returns/outputs/signals:
  - encoded seam payload
  - decoded seam object
  - missing boundary diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and boundary records
  - Additions to existing reusable library/module: `.impress` codec
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
  - future `.impress` codec module
- Chosen defaults / parameters:
  - seam references validate after patches load
- Test strategy:
  - seam round-trip and missing boundary tests
- Data ownership:
  - codec owns serialized form; seam model owns runtime invariants
- Routes:
  - save/load to seam codec after patch decode
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Seams are split from adjacency because they are persisted kernel truth.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Review for split. Cohesion reason: seam payload and boundary references are one persisted topology boundary.

### Candidate Spec: `.impress` Adjacency Payload Codec

Discovery purpose:
- Define adjacency payload encoding/decoding as a validated view over persisted surface topology.

Responsibilities:
- Functions/methods:
  - encode adjacency
  - decode adjacency
  - adjacency reference validation
- Data structures/models:
  - adjacency payload
  - adjacency diagnostic
- Dependencies/services:
  - adjacency model
  - `.impress` root
- Returns/outputs/signals:
  - encoded adjacency payload
  - decoded adjacency view
  - missing reference diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: adjacency records/views
  - Additions to existing reusable library/module: `.impress` codec
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
  - future `.impress` codec module
- Chosen defaults / parameters:
  - adjacency references validate after body/shell/patch/seam load
- Test strategy:
  - adjacency round-trip and invalid reference tests
- Data ownership:
  - codec owns serialized form; adjacency view owns navigation assumptions
- Routes:
  - save/load to adjacency validation after seam decode
- Reuse/extraction decision:
  - add to `.impress` codec module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Adjacency may be rebuildable later, but V1 needs an explicit persisted/refusal policy.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Review for split. Cohesion reason: adjacency payload is one cross-reference validation boundary.

### Candidate Spec: `.impress` Deterministic JSON Writer

Discovery purpose:
- Define byte-stable deterministic JSON writing for `.impress` payloads.

Responsibilities:
- Functions/methods:
  - `save_impress` writer path
  - deterministic JSON writer
- Data structures/models:
  - save options
  - serialized document payload
- Dependencies/services:
  - `.impress` codec
  - JSON encoder
  - filesystem paths
- Returns/outputs/signals:
  - deterministic persisted file
  - write error
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: standard JSON and pathlib
  - Additions to existing reusable library/module: `.impress` persistence module
  - New reusable library/module to create: none beyond persistence module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes user-selected files
- Security/privacy-sensitive behavior:
  - does not execute payloads; writes validated data only
- Performance-sensitive behavior:
  - V1 may serialize in memory; output order must be deterministic
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` persistence module
- Chosen defaults / parameters:
  - V1 writes sorted-key deterministic JSON with explicit schema version
- Test strategy:
  - tests for byte-stable output and write errors
- Data ownership:
  - writer owns file boundary; codec owns payload validation
- Routes:
  - public save API to codec to deterministic writer
- Reuse/extraction decision:
  - add to `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Atomic replacement is handled by the separate atomic write/error spec.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Review for split. Cohesion reason: deterministic writing is one file-output behavior.

### Candidate Spec: `.impress` Reader And Load Result Contract

Discovery purpose:
- Define public load behavior, load result shape, and schema/payload error reporting.

Responsibilities:
- Functions/methods:
  - `load_impress`
  - deterministic JSON reader
- Data structures/models:
  - load result
  - schema error
  - payload error
- Dependencies/services:
  - `.impress` codec
  - JSON decoder
  - filesystem paths
- Returns/outputs/signals:
  - loaded body store
  - IO/schema errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: standard JSON and pathlib
  - Additions to existing reusable library/module: `.impress` persistence module
  - New reusable library/module to create: none beyond persistence module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads user-selected files
- Security/privacy-sensitive behavior:
  - refuses unsafe schema and does not execute payloads
- Performance-sensitive behavior:
  - load validation linear in entity count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` persistence module
- Chosen defaults / parameters:
  - load returns `SurfaceBodyStore` or raises structured diagnostics according to final error policy
- Test strategy:
  - tests for load round trip, malformed JSON, unsupported schema, and invalid payload
- Data ownership:
  - reader owns file boundary; codec owns payload validation
- Routes:
  - public load API to reader to codec
- Reuse/extraction decision:
  - add to `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need final error style before implementation spec promotion: exceptions versus result object.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
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
- Review for split. Cohesion reason: reading, load result, and load diagnostics are one input boundary.

### Candidate Spec: `.impress` Atomic File Write And Error Handling

Discovery purpose:
- Define atomic write behavior and filesystem error semantics for `.impress` save operations.

Responsibilities:
- Functions/methods:
  - atomic write helper
  - write error mapper
- Data structures/models:
  - temporary path policy
  - write error diagnostic
- Dependencies/services:
  - filesystem paths
  - `.impress` writer
- Returns/outputs/signals:
  - atomically replaced file
  - filesystem write error
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: pathlib/os replace behavior
  - Additions to existing reusable library/module: `.impress` persistence module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes and replaces user-selected files
- Security/privacy-sensitive behavior:
  - no extra privacy-sensitive behavior beyond local file paths
- Performance-sensitive behavior:
  - write path should avoid duplicate large serialization beyond V1 in-memory buffer
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - future `.impress` persistence module
- Chosen defaults / parameters:
  - write to temporary sibling path, then replace destination atomically where supported
- Test strategy:
  - tests for successful replace, failed write cleanup, and error diagnostics
- Data ownership:
  - writer owns file boundary and temporary file lifecycle
- Routes:
  - public save API to deterministic writer to atomic replace
- Reuse/extraction decision:
  - add to `.impress` persistence module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Filesystem semantics may vary by platform; tests should avoid depending on platform-specific error text.

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
- Review for split. Cohesion reason: atomic write and write error cleanup are one filesystem boundary.

### Candidate Spec: `.impress` Round Trip And Refusal Tests

Discovery purpose:
- Define acceptance tests proving `.impress` preserves identity/metadata and
  refuses invalid files.

Responsibilities:
- Functions/methods:
  - round-trip fixture builder
  - invalid-file fixture builder
  - assertion helpers
- Data structures/models:
  - round-trip fixture
  - invalid-file case
  - metadata preservation assertion
- Dependencies/services:
  - `.impress` IO API
  - surface model fixtures
- Returns/outputs/signals:
  - passing round-trip test
  - explicit refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing surface fixtures
  - Additions to existing reusable library/module: test fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary test files
- Security/privacy-sensitive behavior:
  - invalid-file tests include unsafe implicit/code payload refusal
- Performance-sensitive behavior:
  - fixture sizes are bounded
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/`
- Chosen defaults / parameters:
  - tests compare semantic payload and deterministic serialized output where
    appropriate
- Test strategy:
  - this is the test specification branch for `.impress`
- Data ownership:
  - tests own fixtures; IO API owns behavior under test
- Routes:
  - tests to public `.impress` save/load API
- Reuse/extraction decision:
  - add reusable test helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Should be paired with implementation specs rather than waiting until the end.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
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
- Review for split.
- Cohesion reason: round-trip and refusal tests are the paired acceptance gate
  for the same file format boundary.

### Candidate Spec: `.impress` Industry Interchange Boundary

Discovery purpose:
- Define what `.impress` does and does not promise relative to STEP, IGES, STL,
  and imported mesh workflows.

Responsibilities:
- Functions/methods:
  - interchange boundary policy
  - imported mesh object classification
- Data structures/models:
  - imported mesh record
  - external format adapter placeholder
- Dependencies/services:
  - `.impress` document root
  - mesh toolchain
- Returns/outputs/signals:
  - explicit unsupported adapter diagnostic
  - imported mesh classification
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: mesh toolchain architecture
  - Additions to existing reusable library/module: `.impress` docs/specs
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - documentation/spec only for V1
- Security/privacy-sensitive behavior:
  - imported external payloads are data only, never executable
- Performance-sensitive behavior:
  - not applicable for V1 policy spec
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `project/release-0.1.0a/specifications/`
- Chosen defaults / parameters:
  - STEP/IGES adapters are explicitly excluded from `.impress` V1 unless later
    planned; STL/mesh remains export/import tool data, not surface truth
- Test strategy:
  - documentation review plus refusal tests if placeholder APIs exist
- Data ownership:
  - `.impress` owns native surface format; external adapters own interchange
    only when separately specified
- Routes:
  - architecture/spec policy to future adapter specs
- Reuse/extraction decision:
  - reuse existing mesh/import/export boundaries
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec prevents scope creep and should be finalized early.

Score:
- Functions/methods: 2 x 2 = 4
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
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 0 x 2 = 0
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
- Cohesion reason: this is one scope-boundary policy spec.

## Open Decisions

- Whether `.impress` V1 allows multiple bodies per file by default or treats
  multi-body files as a store-only mode.
- Whether stored stable identities are required fields or optional diagnostic
  fields.
- Whether document units are initially symbolic only or normalized into a
  unit-conversion policy.
- Whether authoring topology rails belong directly in V1 `.impress` payloads
  or in a later modeling-document layer above `SurfaceBody`.

## Change History

- 2026-05-26: Further split high-scoring manifest entries where review exposed hidden schema, codec, trim, seam, and adjacency boundaries.
- 2026-05-26: Split all manifest candidates that scored 25+ into smaller assessed candidates for spec promotion.
- 2026-05-26: Replaced the lightweight specification list with a
  template-assessed Specification Manifest for Discovery for `.impress` file
  format work.
- 2026-05-26: Initial `.impress` surface-native file format architecture.
  Added to define durable persistence for `SurfaceBody`/`SurfaceBodyStore`
  before specification work begins.
