---
created: 2026-07-23
---

# Principal Engineer Review: Technical And Industry Completeness

## Review Frame

This third pass resets perspective again. It evaluates the whole product as a
sequence of user workflows, then compares the implemented capability set with
current additive-manufacturing and Python CAD expectations.

It asks:

1. Can a user complete each workflow Impression already claims?
2. Are failures explicit, dimensionally safe, and recoverable?
3. What is missing for a loft-centric alpha, a production additive tool, and a
   general parametric CAD system?

The industry comparison uses primary sources. It is a positioning tool, not a
requirement that every competitor feature enter the next release.

## Verdict

Impression has a deep and distinctive surface/loft research kernel, but the
current product is incomplete before the advanced-feature question even begins.
Install, first preview, first export, common solids, mesh CSG examples, unit
handling, release verification, and reference promotion all have end-to-end
gaps.

The codebase is therefore best described as an alpha geometry kernel plus
developer review tooling, not yet a dependable modeling distribution.

There are three different completion targets:

- **Loft-centric alpha toolkit:** repair the claimed workflows and publish hard
  capability boundaries.
- **Production additive tool:** add manufacturing-safe validation, 3MF, units,
  provenance, and durable artifact workflow.
- **General parametric CAD:** additionally add broad solid operations,
  selection/history semantics, STEP/DXF, sketches, patterns, and assemblies.

Keeping those targets separate is essential to honest release planning.

## End-To-End Workflow Matrix

| Workflow | Current state | Completion gate |
| --- | --- | --- |
| Install released package | Installer builds a wheel | Install declared deps, `pip check`, execute CLI and model smoke |
| Import library | Works on this checkout | No package/user-state mutation; clean wheel import |
| Build first primitive | Public `SurfaceBody` API exists | All claimed solids closed-valid |
| Preview Python model | Mesh models work | Canonical `SurfaceBody` and compositions work |
| Preview `.impress` | Explicit preview loader exists | Same consumer and unit behavior as Python models |
| Compose/boolean | Large surface matrix; mesh helper exists | Public dispatch and docs agree; supported cases execute |
| Export STL | Writer exists | Export intent, watertight/orientation checks, unit conversion |
| Save native model | Rich `.impress` V1 format | Migration policy, bounded files, unit semantics, CLI parity |
| Review references | Discovery, UI, dirty/gold concepts exist | Transactional promotion and stable worker lifecycle |
| Release | Wheel/docs workflows exist | Full tests, wheel smoke, Python support matrix, exact-tag gate |

## Findings

### P0: Installation Does Not Prove A Usable Product

The canonical installer manually installs a subset of dependencies and installs
the wheel with `--no-deps`. It omits declared packages used by CAD and SDF
features. Its module smoke command exits zero without invoking the CLI, and the
executable is checked for existence rather than executed.

The release workflow does not install or test the artifact. CI does not test the
supported `>=3.10` Python range and currently cannot collect the documented
hinge surface without an undeclared sibling package.

Completion gate:

- build once, install the exact wheel into a clean environment;
- install through normal dependency metadata;
- run `pip check`;
- execute `impression --version`, `--help`, preview payload generation, and
  canonical STL export;
- test the minimum Python and release Python;
- publish only that verified artifact.

### P0: The Documented First Model Cannot Reach Preview Or Export

The README's first model returns `make_box()`, a `SurfaceBody`. Primary preview
scene collection does not support that type. Reference preview separately does.
The canonical box export fails before producing an artifact.

This must be the first feature-completeness gate. A sophisticated loft or CSG
matrix cannot compensate for a broken first ten minutes.

Completion gate:

- one scene protocol for body, collection, composition, mesh compatibility, and
  path/drawing values;
- exact README preview and export tests;
- tests from an installed wheel, not an editable checkout;
- deterministic error messages for genuinely unsupported outputs.

### P0: The Solid And Watertight Product Claim Is Not Met

The README positions Impression around watertight STL generation. Fresh default
probes found:

- closed: box, sphere, torus;
- open: cylinder, cone, prism, nhedron;
- additional public polygonal solid constructors follow the same disconnected
  cap pattern.

The current tests intentionally accept open shells for several "solid"
primitives. Text surface extrusion uses the same disconnected cap/sidewall
pattern. The CLI then bypasses `export_tessellation_request()` and writes
preview meshes without requiring watertightness.

Completion gate:

- define which constructors promise a solid versus an open surface;
- seam-connect caps and sidewalls with correct orientation;
- require closed-valid surface truth and watertight tessellation for solid
  output;
- reject nonmanifold, degenerate, self-intersecting, disconnected, and inverted
  meshes at manufacturing export;
- read exported artifacts back and verify dimensions and topology.

### P0: Documented Boolean Workflows Are Not A Coherent Feature

The public boolean names claim mesh and surface inputs, while mesh examples fail
and surface execution supports a bounded matrix. The current matrix has 300
operation/family rows:

| Support state | Rows |
| --- | ---: |
| exact | 27 |
| declared tolerance | 120 |
| unsupported | 153 |

An explicit support matrix and refusal policy are good foundations. They do not
make the generic documented examples complete, and "declared tolerance" needs
physical residual and topology evidence per operation.

Completion gate:

- one typed public dispatch contract;
- executable canonical union, difference, and intersection examples;
- an externally readable support matrix with tolerances and limitations;
- topology, residual, watertightness, and no-hidden-fallback tests;
- honest separation of exact, approximate, promoted, compatibility, and
  unsupported routes.

### P1: Units Are Metadata And Labels, Not Geometric Semantics

Configuration defines millimeter, meter, and inch scales, but no export path
applies them. STL stores raw coordinates and carries no unit declaration.
`.impress` V1 stores symbolic units while explicitly placing conversion outside
its validation contract.

This is a high-risk manufacturing gap: a dimension of `1` can silently be
interpreted as 1 mm regardless of the configured inch or meter label.

Completion gate:

- choose one canonical internal length unit or make units part of every value;
- convert at all import/export boundaries;
- preserve units in formats that support them;
- require explicit STL unit policy;
- test dimensions by round-trip and external-reader convention.

3MF is especially relevant because its core format defines units and a
manufacturing-oriented container. The current 3MF specification also includes
standard extensions for materials, production, displacement, booleans, beam
lattices, slices, and volumetric data:
[3MF specification suite](https://3mf.io/spec/).

### P1: Manufacturing Validation Is Too Shallow

`MeshAnalysis` checks vertex/face counts, degeneracy, invalid vertices, boundary
edges, and nonmanifold edge incidence. It does not establish:

- consistently oriented edges and outward normals;
- connected components and unintended shells;
- self-intersection;
- duplicate/coincident faces;
- minimum feature or wall thickness;
- clearances and fit allowances;
- overhangs, trapped volumes, support access, or build orientation;
- dimensional tolerance against authored surface truth.

The unused `printability.py` helper is not an integrated design-for-additive
workflow.

Completion gate for production positioning:

- a layered validation report separating mesh validity, solid validity,
  dimensional fidelity, and process advisories;
- configurable manufacturing tolerances and severity;
- export refusal on geometry-invalid conditions;
- reference artifacts with independent slicer or standard-library validation.

### P1: Native Persistence Is Rich But Isolated

`.impress` has deterministic JSON, patch-family payload versions, unit metadata,
and safety budgets for implicit fields. It accepts only the exact current root
schema version (`1.0`) and has no migration path.

Product gaps:

- CLI export does not share the `.impress`-aware preview loader;
- unit metadata is not converted;
- file reads are not globally size-bounded;
- there is no compatibility/migration policy;
- the format is not an industry interchange contract.

Completion gate:

- document backward/forward compatibility and migration ownership;
- add bounded loading for large arrays and files;
- support preview/export through the same scene consumer;
- add golden files from released schemas and migration tests;
- keep `.impress` as native parametric truth while using standards for external
  exchange.

### P1: External Interchange Is Below Both Additive And CAD Baselines

The public product currently exports STL and persists `.impress`. It does not
provide public 3MF, STEP, DXF, SVG, or glTF interchange.

Industry implications:

- 3MF is now an ISO/IEC standard and is designed for full-fidelity additive
  transfer with defined units and extensible manufacturing metadata:
  [3MF Consortium](https://3mf.io/).
- STEP AP242 covers managed model-based 3D engineering, product structure,
  configuration, long-term archiving, and manufacturing information:
  [ISO 10303-242:2025](https://www.iso.org/standard/84300.html).
- The already-declared build123d dependency exposes STEP, STL, 3MF, SVG, DXF,
  BREP, and glTF paths:
  [build123d import/export](https://build123d.readthedocs.io/en/stable/import_export.html).

Recommended priority:

1. 3MF export/import for additive workflows, units, labels, color/material
   extension planning, and multi-part builds.
2. STEP import/export for CAD collaboration, with explicit conversion fidelity
   between Impression surfaces and B-rep.
3. DXF/SVG for 2D profiles and drawings.
4. glTF only if visualization interchange becomes a product requirement.

Adapters need fidelity, tolerance, unit, provenance, and refusal contracts.
Dependency association alone is not capability.

### P1: Foundational Solid Modeling Operations Are Missing Or Retired

The public facade is loft-centric. Legacy mesh `linear_extrude()` and
`rotate_extrude()` exist in `modeling/extrude.py`, but documentation calls the
public extrusion path retired. There is no coherent public solid workflow for:

- linear extrude with taper and termination conditions;
- revolve;
- sweep/pipe along a path;
- fillet and chamfer;
- shell/thicken;
- draft;
- split/section;
- hole/countersink/counterbore features;
- linear, circular, and path patterns;
- robust face/edge selection across feature changes.

For comparison, the current build123d operation set includes extrude, revolve,
sweep, loft, fillet, chamfer, draft, offset, split, section, thicken, projection,
and selection APIs:
[build123d operations](https://build123d.readthedocs.io/en/latest/operations.html).
Open CASCADE likewise treats fillets, chamfers, offsets, sewing, and form
features as core modeling algorithms:
[OCCT modeling algorithms](https://dev.opencascade.org/doc/occt-7.8.0/overview/html/occt_user_guides__modeling_algos.html).

Not all belong in the next loft-centric alpha. Extrude/revolve, sweep,
fillet/chamfer, shell/thicken, selection stability, and patterns are baseline
requirements before "comprehensive parametric modeling" is a credible claim.

### P1: Surface Modeling Lacks Higher-Order Continuity Completion

The surface architecture has many patch families, seams, transformations,
intersection planning, and explicit diagnostics. Continuity enforcement beyond
positional `C0` is still marked unsupported or not implemented for `C1`, `G1`,
`C2`, and `G2` (`modeling/surface.py:7413-7422`).

For organic and industrial-design positioning, users need:

- tangent and curvature continuity constraints;
- continuity diagnostics with measurable residuals;
- blend/transition surfaces;
- trim, extend, join/sew, offset, and repair workflows;
- local control-point editing and fairing;
- stable face/edge identity after modification.

This is a deeper industry differentiator than adding more isolated patch record
types.

### P1: Reference Review Cannot Yet Be A Release Evidence System

The review application has fixture discovery, payload generation, dirty/gold
artifacts, status, notes, async execution, and promotion concepts. However:

- multi-artifact promotion is not transactional;
- one approval route can delete existing gold before complete validation;
- status/provenance writes are not consistently atomic;
- stale worker processes can survive cancellation and shutdown;
- the full suite can fatally abort inside Qt;
- CI does not run the repository-wide model and artifact contract.

Completion gate:

- transactional artifact plus provenance promotion;
- immutable source identity and checksums;
- rollback and crash-recovery tests;
- deterministic headless capture where possible;
- isolated UI tests;
- release policy requiring clean approved artifacts for claimed workflows.

### P2: Text Geometry Exists, But Typography And Packaging Are Incomplete

Current text strengths:

- surface-first `make_text()`;
- topology-native sections/profiles;
- explicit mesh compatibility output;
- multiline, alignment, letter spacing, font path, and glyph outlines.

Current gaps:

- open cap/sidewall topology prevents manufacturing-safe solid output;
- empty text returns hidden micro-geometry;
- the documented font is absent from the wheel;
- default Arial resolution is host-dependent;
- no guaranteed boolean route;
- one code point is mapped directly through `cmap` and advanced with `hmtx`;
- no GSUB/GPOS shaping, kerning engine, bidi, fallback, script joining, or
  language-aware positioning.

HarfBuzz's role is precisely to convert Unicode text into correctly selected and
positioned glyphs using script, language, direction, and font layout rules:
[HarfBuzz overview](https://harfbuzz.github.io/what-is-harfbuzz.html).

Recommended tiers:

- Tier 1: packaged-font, reproducible Latin text; closed solid output.
- Tier 2: HarfBuzz shaping, kerning, ligatures, and combining marks.
- Tier 3: fallback, bidirectional and complex-script layout, and vertical text.

### P2: Loft Correspondence Needs A Product-Level Confidence Contract

The current kernel contains automatic, named, ambiguity, split/merge, and
point-lifecycle machinery. The dirty square/rectangle fixtures use unnamed
automatic correspondence, and the reported softened-corner symptom is credible.
The exact bad rail still needs live artifact inspection.

Recommended contract:

- automatic correspondence is allowed when the selected mapping is measurably
  unambiguous;
- authored names are hard anchors;
- mixed named and unnamed sections are supported by inferring only between
  anchors;
- unresolved equivalent rotations or corner phases produce refusal, not a
  silent winner;
- diagnostics expose selected rails, confidence, competing assignments, and
  resampling;
- regression tests assert physical corner preservation, not only deterministic
  payloads.

Requiring names everywhere is a valid simpler product tier, but it would
deliberately remove automatic correspondence as a supported promise. That
should be a product decision, not an accidental response to one fixture.

### P2: Endcaps Exist, But The Feature Family Is Split

Surface-native loft supports `flat`, `taper`, `dome`, `slope`, and `none`.
Experimental `loft_endcaps()` returns mesh output for `FLAT`, `CHAMFER`,
`ROUND`, and `COVE`. Non-flat surface caps also have topology limitations.

Comparison with `v0.0.3a1`/`v0.0.3a2` indicates that the older cap family is not
simply missing source to copy back; it already lived in the experimental mesh
route. Reuse should mean adapting its proven profile strategy and visual
fixtures into current surface-native seam truth.

Completion gate:

- one public vocabulary with geometric definitions;
- surface-native output for supported types;
- multi-region behavior or explicit refusal;
- old/current dimensional and visual parity fixtures;
- watertight cap-to-loft seams;
- mesh route labeled compatibility/debug only.

### P2: Assemblies, Constraints, And Product Structure Are Absent

Groups and transforms are not an assembly model. General mechanical work needs:

- named parts and instances;
- joints/mates or constraints;
- hierarchy and coordinate systems;
- interference and clearance analysis;
- BOM/product metadata;
- stable references and replacement;
- multi-body export and configuration.

This is not required for a single-part loft toolkit. It is required for general
CAD and aligns with the product-structure scope of STEP AP242.

### P2: Documentation And Assurance Do Not Execute Product Truth

Documentation tests mostly inspect text and file presence. Several example
scripts fail when actually built. There is no current durable full-coverage
artifact under `project/coverage`, no benchmark gate, and no static API/type
gate.

Completion gate:

- execute all examples in bounded categories;
- assert README commands;
- separate documentation syntax from physical model validation;
- generate and retain coverage for the exact release commit;
- publish capability matrices and known refusals beside release notes.

## Capability Assessment

| Capability | Kernel depth | Product completeness |
| --- | --- | --- |
| Surface values and patch families | high | medium-low |
| Loft planning and diagnostics | high | medium |
| Automatic correspondence | high | low until visual confidence contract |
| Endcaps | medium | split across surface and mesh |
| CSG planning/policy | high | low for generic user workflow |
| Primitives | medium | low for solid output |
| Text | medium geometry, basic layout | low for manufacturing/typography |
| Heightmap/SDF | meaningful experimental paths | low integration confidence |
| Native persistence | high V1 schema detail | medium-low lifecycle completeness |
| Preview | mature mesh viewer | low canonical surface integration |
| STL | basic writer | low manufacturing safety |
| Reference review | broad developer tooling | low transactional/release readiness |
| Packaging/release | basic wheel workflow | low |
| General CAD operations | limited/retired | low |
| Industry interchange | STL only | low |

## Direct Answers To The Starting Topics

### Dirty Square/Rectangle STL

Do not promote it. The current fixture relies on unnamed automatic
correspondence, and a softened corner is a physically meaningful regression.
Inspect the chosen rails and add a corner-preservation assertion before deciding
whether the defect is resampling, cyclic phase, or ambiguity policy.

### Named, Automatic, And Mixed Correspondence

The strongest long-term contract is automatic when unambiguous, names as hard
anchors, mixed mode supported between anchors, and refusal when ambiguity
remains. Universal naming is a viable reduced-scope alpha contract if the team
prefers predictability over inference.

### Current Text State

Text is implemented as real outline-derived surface geometry with topology
profiles and mesh compatibility output. It is not complete as manufacturable
text or international typography: output topology is open, font packaging is
non-reproducible, empty text uses hidden geometry, and layout bypasses shaping.

### Current Endcap State

Endcaps were reimplemented and are present. Flat/taper/dome/slope are on the
surface-native loft path; chamfer/round/cove remain experimental mesh output.
The older code is useful as algorithm and visual evidence, but should be adapted
to current surface topology rather than copied wholesale.

## Release Gates By Product Claim

### Loft-Centric Alpha

1. Install and execute the built wheel.
2. Restore canonical `SurfaceBody` preview/export.
3. Close claimed solid primitives and enforce export validity.
4. Repair public CSG examples and publish the support matrix.
5. Resolve the square correspondence regression and define mixed-mode policy.
6. Choose endcap vocabulary and verify surface-native parity.
7. Make tests deterministic and release-gated.

### Production Additive Tool

Complete the alpha gates, then add:

1. unit-safe 3MF and explicit STL unit policy;
2. orientation, self-intersection, component, thickness, and process validation;
3. transactional reference artifacts and traceability;
4. multi-part/material/build metadata;
5. reproducible text and packaged assets.

### General Parametric CAD

Complete the previous gates, then add:

1. extrude, revolve, sweep, fillet, chamfer, shell/thicken, patterns, and robust
   selection;
2. higher-order surface continuity and editing;
3. STEP/DXF interchange;
4. assemblies, constraints, product structure, and configurations;
5. a coherent feature/history or dependency-update model.

The immediate release plan should target the first tier and stop claiming the
second or third until their end-to-end gates exist.
