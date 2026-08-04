---
created: 2026-07-23
---

# Principal Engineer Review: Code Quality

## Review Frame

This review is about the quality of the code as code: what an experienced
engineer encounters while reading, navigating, calling, changing, and debugging
the implementation.

It evaluates:

- naming and identifier length;
- function size, shape, and call depth;
- API coherence and discoverability;
- module layout and ownership;
- type boundaries and mutation semantics;
- abstraction quality and duplication;
- comments and documentation at the point of use;
- whether the code reads like deliberate human engineering or accumulated
  generated code.

Correctness, security, packaging, and release findings are preserved separately
in
[Correctness And Release Integrity Review](correctness-and-release-integrity-principal-engineer-review.md).

## Direct Verdict

Impression is not uniformly poor code. Several small numerical and geometry
modules are clear, direct, and pleasant to read. `modeling/bspline.py`,
`modeling/_ops_planar.py`, and much of `mesh.py` use recognizable domain terms,
short functions, local validation, and straightforward control flow.

The core repository as a whole, however, does not read like consistently
human-curated engineering. The strongest impression is AI-assisted accumulation:

- names restate entire contexts instead of naming stable concepts;
- every policy stage gains another record, diagnostic, report, verifier, and
  canonical payload;
- simple public calls pass through long wrapper and validation tails;
- release evidence and fixture bookkeeping live inside production geometry
  modules;
- APIs grow by adding parameters and aliases rather than by clarifying the
  object model;
- local explicitness is repeatedly purchased at the cost of global readability.

It is not classic tangled spaghetti. Most individual functions are typed and
deterministic. The problem is over-factoring, over-modeling, and insufficient
curation: an engineer can usually understand the next ten lines, but must cross
too many names, records, modules, and policy layers to understand one operation.

The code needs subtraction and ownership work more than it needs additional
abstractions.

## Measured Shape

The following measurements cover `src/impression/**/*.py`.

| Measure | Count |
| --- | ---: |
| Source modules | 89 |
| Source lines | 67,499 |
| Functions and methods | 2,801 |
| Classes | 749 |
| Functions longer than 100 lines | 50 |
| Functions longer than 200 lines | 7 |
| Functions longer than 400 lines | 2 |
| Functions with more than 8 parameters | 30 |
| Functions with more than 12 parameters | 12 |
| Function names longer than 40 characters | 176 |
| Class names longer than 40 characters | 48 |
| Unique identifiers longer than 30 characters | 785 |
| Top-level one-call forwarding wrappers | 110 |
| Modeling `canonical_payload()` methods | 378 |
| Dataclasses in CSG, surface, and loft | 387 |
| Public `impression.modeling` exports | 1,018 |

The line formatting itself is not the central problem: only 281 lines exceed
120 characters. The excessive cognitive width comes from concepts, names,
parameters, and layers rather than raw horizontal formatting.

## Findings

### P0: The Public Modeling API Is An Internal Inventory, Not A Curated Interface

`src/impression/modeling/__init__.py` is 2,129 lines and exports 1,018 names.
Those exports include:

| Export category | Count |
| --- | ---: |
| Capitalized types | 574 |
| `*Record` | 176 |
| `*Diagnostic` | 97 |
| `*Report` | 37 |
| `build_*` functions | 44 |
| `assert_*` functions | 16 |
| `verify_*` functions | 10 |

The facade exposes implementation evidence, migration scanners, support
matrices, refusal records, fixtures, and completion gates beside ordinary
modeling operations. Its opening primitive section includes
`LegacyPrimitiveMeshAssumptionInventoryReport` and repository scanning functions
before basic path and topology values
(`modeling/__init__.py:1110-1160`).

Consequences for an engineer:

- IDE autocomplete cannot communicate the intended workflow;
- every internal type looks supported;
- renaming or reorganizing implementation records becomes an API change;
- it is difficult to tell which five concepts matter for a normal model;
- documentation has to explain an inventory instead of teaching a design.

Required direction:

- define a small user facade containing stable geometry values and modeling
  operations;
- put advanced planning and diagnostics in explicit submodules;
- move compatibility APIs under a compatibility namespace;
- stop exporting fixture, evidence, release, and repository-scanning helpers;
- maintain a reviewed public API manifest.

A useful target is tens of top-level names, not hundreds.

### P0: Core Geometry Modules Contain Process Governance Instead Of Geometry

The production kernel directly implements capability promotion, release
completion, reference evidence, fixture cleanliness, and specification
retirement policy.

Examples:

- the first roughly 2,300 lines of `modeling/surface.py` are dominated by
  availability, evidence, completion, promotion, and reference contracts before
  the main surface geometry implementation;
- `modeling/primitives.py:199-338` scans Python and Markdown source text for
  legacy assumptions;
- `modeling/csg.py:6670-7185` defines a long sequence of promotion rows,
  diagnostics, feasibility reports, evidence reports, fixture rows, proof
  records, and gates;
- CSG alone has 172 top-level symbols whose names contain terms such as
  `evidence`, `fixture`, `completion`, `promotion`, `diagnostic`, `report`, or
  `gate`, occupying about 4,780 source lines;
- surface has 85 such top-level symbols occupying about 1,788 lines.

This code may represent valuable policy, but it is in the wrong ownership
boundary. It makes the domain kernel read like a specification ledger translated
into Python.

Required direction:

- keep runtime invariants in the kernel;
- move repository scans, fixture generation, evidence matrices, completion
  reports, and release assertions to tests or developer tooling;
- store static capability declarations as data where possible;
- make tests inspect the runtime instead of making the runtime embed the test
  process.

The test for whether a piece belongs in `csg.py` should be: does model execution
need it, or does project governance need it?

### P0: The Loft Interface Is A Parameter Train Repeated Through Wrapper Layers

`loft_profiles()` takes 31 parameters. `loft()` repeats the same 31-parameter
signature and spends 68 lines forwarding them
(`modeling/loft.py:2727-2893`). `Loft()` is a capitalized function, not a class,
and exposes another 24-parameter variant. `_loft_profiles_surface()` repeats
most of `loft_profiles()` with a slightly different tail.

The options span unrelated concerns:

- input sampling;
- cap construction;
- split/merge behavior;
- ambiguity selection;
- probabilistic search;
- fairness;
- skeleton behavior;
- patch-family intent.

This is difficult to call, document, test, and evolve. It also creates long
forwarding blocks where omission or default drift is easy.

The capitalized `Loft` function, lowercase `loft`, `loft_profiles`,
`loft_sections`, `loft_plan_sections`, `_loft_profiles_surface`, and
`loft_execute_plan` do not form a self-evident progression for a reader.

Required direction:

- keep one obvious public `loft(...)` entry point;
- group advanced controls into a few cohesive immutable options, such as
  sampling, correspondence, caps, and fairness;
- normalize those options once;
- make the visible implementation read:
  `normalize inputs -> plan -> execute`;
- remove aliases whose only job is to forward the complete parameter train;
- use capitalized names only for types.

This should not create dozens of new option records. Four cohesive values are
better than 31 parameters and also better than 31 one-field dataclasses.

### P1: Long Names Are Compensating For Missing Concept Boundaries

Long names are not inherently bad. `control_points` is better than `cp`, and the
compact math modules use appropriately short conventional names inside small
scopes.

The core crosses into sentence-length noun stacking:

- `assert_surface_reference_requirement_matrix_covers_capabilities`
- `verify_sampled_implicit_promotion_fixture_evidence_matrix`
- `build_sampled_implicit_dirty_evidence_completion_blocker`
- `SurfaceSampledImplicitReconstructionFeasibilityReport`
- `SurfaceSampledImplicitPromotionProvenanceDiagnostic`

There are 176 function names and 48 class names over 40 characters. The longest
function name is 63 characters. Even local state grows into phrases such as
`probabilistic_selected_candidate_ids` and
`global_optimizer_hit_iteration_cap`.

These names are precise in isolation but expensive in aggregate. They repeatedly
encode the module, input family, operation, stage, artifact type, and result type
in every identifier because the code lacks smaller namespaces and stable domain
objects.

Required direction:

- split modules so context comes from the namespace:
  `sampled_implicit.verify_promotion()` rather than a 57-character global name;
- attach behavior to cohesive concepts:
  `promotion.feasibility()` and `evidence.verify()`;
- remove redundant `Surface`, `CSG`, and family prefixes inside modules already
  dedicated to those concepts;
- reserve `Record`, `Report`, and `Diagnostic` suffixes for genuinely different
  contracts, not every intermediate step;
- keep local names proportional to scope.

The goal is not shorter names at all costs. It is fewer facts encoded in every
name.

### P1: Simple Operations Cross Too Many Conceptual Layers

A local call trace, restricted to Impression source, produced:

| Operation | Internal calls | Distinct internal functions | Maximum depth |
| --- | ---: | ---: | ---: |
| `make_box()` | 180 | 20 | 6 |
| simple two-circle loft | 812 | 137 | 13 |
| `make_text("A")` | 490 | 52 | 12 |

These counts include properties and comprehensions, so they are not performance
benchmarks. They are useful measures of how much implementation a reader may
cross while following one public operation.

The simple loft begins:

```text
loft
  -> loft_profiles
    -> loft_sections
      -> loft_plan_sections
```

before entering normalization, correspondence, planning, and execution.

Even `make_box()` routes through `_surface_primitive_result()`, which imports the
18,699-line CSG module to call a function that only rejects `Mesh` or
`MeshGroup` (`primitives.py:373-413`, `csg.py:14894-14901`).

There are 110 top-level functions whose entire implementation is a single
returned call. Some are useful factories or compatibility aliases. Together,
they show a strong tendency to add names without removing conceptual distance.

Required direction:

- use one facade layer, not several aliases;
- normalize and validate at subsystem boundaries, not again at every hop;
- keep primitive invariants with primitives and type constructors;
- make planner stages explicit data transitions rather than wrapper chains;
- remove pass-through functions that add neither policy, type conversion,
  diagnostics, nor a stable compatibility boundary.

### P1: The Core Is Over-Modeled With Near-Duplicate Record Types

CSG contains 226 top-level classes; 146 begin with `Surface` and 45 with `Loft`.
The CSG, surface, and loft modules contain 387 dataclasses and the modeling
package contains 378 hand-written `canonical_payload()` methods.

The cluster at `csg.py:6670-7185` illustrates the pattern. Separate types exist
for:

- unsupported row report;
- promotion diagnostic;
- policy row;
- decision;
- matrix report;
- lossiness record;
- provenance diagnostic;
- provenance record;
- reconstruction criteria;
- reconstruction diagnostic;
- feasibility report;
- fixture row;
- evidence report;
- refusal record;
- reference fixture row;
- promotion report;
- proof record;
- evidence gate;
- evidence state;
- dirty evidence diagnostic;
- dirty evidence report.

Typed intermediate values are normally a strength. Here, many types differ only
in a few fields and repeat the same `passed`, `diagnostics`, status, message,
and `canonical_payload()` structure. An engineer must learn the entire taxonomy
before determining which value actually changes geometry.

Required direction:

- identify the small set of durable domain values;
- use shared generic result/diagnostic structures for transient evaluation;
- keep persistence schemas separate from every in-memory intermediate;
- replace hand-written payload boilerplate with explicit schema encoders at
  serialization boundaries;
- avoid serializing internal planning trivia unless it is part of a supported
  diagnostic contract.

The answer is not one untyped dictionary. It is fewer, stronger types.

### P1: Policy Is Repeated As Prose Instead Of Encoded Once

Variants of `no mesh fallback` occur 396 times under `modeling/`. CSG repeats
`"no mesh fallback was attempted"` in scores of diagnostic messages.

The policy is important. Repeating the sentence in constructors, fixtures,
reports, refusals, and verifiers makes it noise rather than a clear invariant.
It also couples test assertions to prose and invites wording drift.

Required direction:

- represent fallback posture once as an enum or structured field;
- render user-facing text at the diagnostic boundary;
- let tests assert the code and field, not copied English;
- document the invariant at the owning interface.

This is a characteristic generated-code smell: a prompt requirement appears
verbatim everywhere instead of becoming one design rule.

### P1: Module Ownership Is Cyclic And Hidden By Local Imports

Static import inspection finds one strongly connected component spanning 12
modules, including:

- surface;
- CSG;
- loft;
- tessellation;
- primitives;
- surface primitives and operations;
- intersections;
- heightmap;
- `.impress` I/O.

Sixty-eight functions contain local imports. Some are legitimate optional
dependency boundaries, but many defer internal imports to survive cycles. Public
primitive constructors import their implementation inside each function;
surface classification imports a private tessellation helper from CSG; I/O and
modeling know about each other's specialized evidence.

Consequences:

- dependency direction is not visible at module headers;
- type identity and reload behavior become fragile;
- unit testing requires broad package initialization;
- moving one concept causes changes across the entire component.

Required direction:

- establish a one-way dependency order:
  geometry values -> topology/math -> operations -> consumers/I/O -> tooling;
- move shared protocols below both producer and consumer;
- keep I/O codecs dependent on model types, never the reverse;
- remove private cross-module imports such as CSG reaching into tessellation
  internals;
- use local imports only for optional heavy dependencies or deliberate cycle
  breaks with an explicit architecture note.

### P1: Type Signatures Are Broad Where The API Most Needs Clarity

Two hundred fifty-four functions contain `object` in an annotation. Many are
internal decoder or generic traversal functions, where that can be appropriate.
The public boundaries also accept broad unions such as
`Section | Region | Path2D | object`, raw dictionaries, and untyped `target`
values.

Mode selection is frequently stringly typed at runtime:

- `split_merge_mode`
- `ambiguity_mode`
- `ambiguity_selection_policy`
- `ambiguity_cost_profile`
- `disambiguation_mode`
- `probabilistic_fallback`
- `fairness_mode`
- `skeleton_mode`

The functions then maintain parallel `_validate_*` helpers. The type system
cannot guide an API user, and invalid combinations remain representable.

Mutation semantics also depend on runtime type. `translate()` returns a new
surface body, mutates `MeshGroup`, and mutates a mesh in place
(`modeling/transform.py:28-41`). `scale()` says it is in-place even when a
surface body returns a new value.

Required direction:

- define protocols or closed unions at public boundaries;
- use enums or validated option values for user-selected modes;
- make mutating and non-mutating operations explicit and consistent;
- avoid one function whose semantics change fundamentally by target type;
- return typed result unions only where callers can reasonably act on each
  branch.

### P1: Large Functions Mix Orchestration, Validation, And Domain Logic

The largest functions are:

| Function | Lines | Parameters | Branch nodes |
| --- | ---: | ---: | ---: |
| `loft_plan_sections()` | 428 | 27 | 31 |
| `_validate_loft_plan()` | 426 | 1 | 100 |
| `preview.show()` | 338 | 15 | - |
| `_loft_execute_plan_surface()` | 298 | 2 | 17 |
| `_pair_sections_for_transition()` | 266 | 15 | 24 |
| `decode_surface_patch_payload()` | 213 | 1 | 12 |

Large functions are not automatically wrong. A parser dispatch or numerical
algorithm can be clearer in one place. These functions cross multiple conceptual
phases without a compact top-level narrative.

For example, `loft_plan_sections()` validates more than a dozen individual
controls, derives seeds, expands stations, plans inference, handles
probabilistic selection, records evidence, computes fairness, and assembles the
plan. Its one-line docstring does not orient a maintainer through those phases.

Required direction:

- make the top-level function read as a short sequence of named phases;
- pass one normalized context through those phases;
- keep branch-heavy validation close to the value being validated;
- add comments that explain non-obvious invariants and phase boundaries, not
  comments that restate the next line;
- preserve algorithms intact when splitting would require hidden mutation.

### P2: Duplicate Paths Make It Hard To Know Which Implementation Is Real

Representative duplicates and aliases:

- `loft_profiles()` and `_loft_profiles_surface()` repeat normalization, cap
  handling, station construction, and option forwarding;
- `loft()` is almost entirely an alias for `loft_profiles()`;
- text exposes `make_text`, `text`, `text_profiles`, and `text_sections`;
- surface composition exposes `handoff_*`, `flatten_*`, `make_*`, and
  `surface_group` wrappers over overlapping concepts;
- transform matrices are implemented independently in `group.py` and
  `transform.py`;
- B-spline basis evaluation exists in `bspline.py` and `surface.py`;
- path resampling exists in `loft.py` and `sdf.py`;
- both `impression/cad.py` and `impression/cad/` occupy the same import name.

An engineer should be able to identify one implementation owner from the name
and import path. Compatibility aliases need a clear quarantine and removal
policy.

### P2: The Reference Review UI Has A Large Construction And Coordination Object

`ReferenceReviewWindow.__init__()` is 207 lines and assigns 39 `self`
attributes (`devtools/reference_review/ui/shell.py:539-745`). It:

- initializes view state;
- constructs worker infrastructure;
- creates every major widget;
- lays out the screen;
- wires signals;
- starts window behavior.

The code is readable line by line, but the class owns too many reasons to
change. State coordination and widget construction are tightly coupled, which
makes lifecycle bugs and isolated testing harder.

Required direction:

- extract queue/sidebar, preview pane, review details, and action bar widgets;
- move preview job coordination into a controller with a narrow signal surface;
- keep the window responsible for composition and high-level navigation;
- avoid splitting tiny visual details into classes unless they have state or a
  testable contract.

### P2: Documentation Density Does Not Match Abstraction Density

Only about 29% of functions, methods, and classes have docstrings. Raw coverage
is not the goal, especially for obvious private helpers. The issue is where
documentation effort goes:

- many trivial wrappers have docstrings that paraphrase their names;
- many record types document that they are a record or diagnostic;
- large planner and validator functions have only one-line summaries;
- repeated policy prose is abundant, while algorithm phase and invariant
  explanations are sparse.

Good comments should answer:

- why does this stage exist?
- what invariant enters and leaves?
- what representation owns the truth?
- which tolerances are coupled?
- why is a branch safe?

The compact B-spline and planar modules are readable with little prose because
their structure carries the explanation. The giant planners need stronger
orientation precisely because their structure does not.

## Human-Code Assessment

| Dimension | Assessment |
| --- | --- |
| Local variable clarity | generally good in small modules; overly compound in policy code |
| Function readability | good in numerical islands; poor in core orchestration |
| API discoverability | poor |
| Naming discipline | precise but excessively contextual and repetitive |
| Module cohesion | poor in CSG, surface, loft, facade, and review UI shell |
| Type design | rich but over-modeled internally and too broad publicly |
| Mutation semantics | inconsistent across surface, mesh, and groups |
| Abstraction level | too many wrappers and intermediate records |
| Comments/docstrings | abundant policy restatement, insufficient architectural orientation |
| Generated-code smell | high in core policy/evidence paths |

## Positive Examples To Preserve

### `modeling/bspline.py`

- clear progression from normalization to span lookup, basis evaluation,
  derivative construction, and sampling;
- short domain names inside small scopes;
- validation close to the represented value;
- functions do one recognizable mathematical job.

### `modeling/_ops_planar.py`

- a small public operation delegates to a proven backend;
- conversion helpers are local and easy to follow;
- control flow is direct;
- names are descriptive without restating the module.

### `mesh.py`

- `combine_meshes()` and `analyze_mesh()` are compact and unsurprising;
- data movement is visible;
- the implementation can be understood without learning a parallel taxonomy of
  reports and gates.

These modules are useful style references for the cleanup. The goal is not to
make sophisticated CSG as small as `analyze_mesh()`. It is to recover the same
clarity of ownership and progression.

## Refactoring Standard

The cleanup should follow these rules:

1. **Subtract before abstracting.** Remove unused aliases, evidence machinery,
   and exports before inventing replacement frameworks.
2. **One concept, one owner.** A reader should know where loft options,
   transform math, path resampling, and CSG diagnostics live.
3. **One public path per job.** Compatibility paths are named and isolated.
4. **Normalize once.** Public input becomes a typed internal request at the
   boundary.
5. **Plan, then execute.** Keep policy and geometry execution visibly separate.
6. **Short names through namespaces.** Do not encode the entire call path in
   every identifier.
7. **Few strong records.** Model durable domain state, not every intermediate
   sentence.
8. **Structured diagnostics, rendered prose.** Do not duplicate policy wording.
9. **Consistent value semantics.** Mutation behavior must not depend silently on
   runtime type.
10. **Tooling stays tooling.** Fixtures, completion gates, and repository scans
    do not live in the modeling kernel.

## Recommended Sequence

1. Define and enforce a small public API manifest.
2. Move evidence, fixture, completion, and repository-scanning code out of
   `surface.py`, `csg.py`, and `primitives.py`.
3. Replace the loft parameter train with a small normalized request model and
   remove duplicate loft entry paths.
4. Split CSG by domain ownership: common values, policy/planning, intersections,
   execution, diagnostics, and developer evidence.
5. Break the 12-module import cycle and eliminate internal local-import
   workarounds.
6. Consolidate transform, spline, path, and scene abstractions.
7. Standardize mutation and typing at public boundaries.
8. Decompose the review window around stateful UI components and one preview
   controller.
9. Add lint, formatting, type, import-boundary, and public-API checks after the
   intended structure is defined.
10. Review new code with an explicit generated-code gate: no new record,
    wrapper, verifier, or exported name without showing why an existing concept
    cannot own the behavior.

## Acceptance Criteria

Code-quality improvement should be measured by engineer experience and
structural outcomes:

- a new contributor can identify the public modeling path from the facade;
- a simple loft reaches one normalization function, one planner, and one
  executor;
- advanced loft controls are discoverable without reading a 31-parameter
  signature;
- fixture and completion terminology no longer appears in the runtime geometry
  modules except where it is genuine model metadata;
- the public facade exports no repository scanners or project evidence gates;
- internal modules form an acyclic dependency graph;
- compatibility aliases have explicit ownership and deprecation status;
- large planners expose clear phases and invariants;
- repeated diagnostic policy is represented structurally and rendered once;
- engineers can use concise names because modules and types provide context.

The target is not merely fewer lines. It is code that communicates its design
without forcing the reader to reconstruct a generated specification from a
thousand names.
