---
created: 2026-07-23
---

# Principal Engineer Review: Efficiency And Reuse

## Review Frame

This is an independent repository-wide pass over runtime cost, memory and object
shape, algorithmic scaling, duplicate implementation, process lifecycle, and
reuse of existing internal and external capabilities. It deliberately resets
the priorities from the correctness review.

Severity in this document reflects sustained engineering or runtime cost:

- `P0`: structural duplication directly breaks a canonical workflow
- `P1`: large or scaling cost on common workflows
- `P2`: material duplication, dead weight, or bounded scaling liability
- `P3`: optimization opportunity without current user-visible impact

## Verdict

The dominant problem is not one slow function. Impression has multiple parallel
representations and consumer paths without one shared conversion boundary:

- preview and reference preview traverse scenes differently;
- surface composition exists but is not connected to the primary CLI;
- transforms, paths, spline math, and VTK runtime policy have multiple owners;
- hot reload replaces the entire internal modeling namespace;
- text converts display sampling into authored topology;
- process cancellation marks results stale without stopping expensive work.

This duplication has already drifted into correctness failures. The highest
leverage efficiency work is consolidation of ownership and preservation of
high-level geometry, followed by measured kernel optimization.

## Measured Repository Shape

| Measure | Current checkout |
| --- | ---: |
| Python source modules | 89 |
| Python source lines | 67,499 |
| Functions | 2,801 |
| Classes | 749 |
| `canonical_payload()` methods in modeling | 378 |
| Dataclasses in CSG, surface, and loft | 387 |
| Public `impression.modeling.__all__` names | 1,018 |

Largest modules:

| Module | Lines |
| --- | ---: |
| `modeling/csg.py` | 18,699 |
| `modeling/surface.py` | 8,170 |
| `modeling/loft.py` | 7,815 |
| `io/impress.py` | 2,745 |
| `surface_intersections.py` | 2,338 |
| `modeling/__init__.py` | 2,129 |

These metrics do not prove poor design by themselves. They identify review and
change domains where reuse must be intentional rather than incidental.

## Findings

### P0: Scene Consumption Has Multiple Incompatible Implementations

`src/impression/preview.py:628-696` recursively collects mesh and path values but
does not understand surface values. The reference-review payload builder has a
separate traversal at
`src/impression/devtools/reference_review/preview_payload_builder.py:169-230`
that does handle `SurfaceBody` and `SurfaceConsumerCollection`.

`src/impression/modeling/surface_scene.py` adds another composition and
tessellation traversal for `SurfaceComposition`, but neither primary preview nor
CLI export reuses it. `.impress` loading has its own explicit tessellation path
in `src/impression/cli.py:396-400`.

This is not hypothetical duplication: the reference review can preview a
surface body that the canonical CLI rejects.

Recommended architecture:

1. Define one normalized scene protocol with explicit nodes for body,
   composition, mesh compatibility, path/drawing, style, and visibility.
2. Traverse it once into consumer records.
3. Give preview, export, reference capture, and persistence different
   tessellation intents without different type-dispatch trees.
4. Make unsupported nodes fail through one diagnostic contract.

### P1: Hot Reload Invalidates Far More Than User Code

`src/impression/cli.py:55-82` deletes all loaded
`impression.modeling` modules for every user-model load. Besides corrupting
class identity, this discards module initialization, import caches, registries,
and any future kernel caches. Reload cost grows with Impression rather than with
the user's dependency graph.

The same file already contains logic to identify local transitive model modules
while excluding Impression and site packages
(`src/impression/cli.py:93-125`). That narrower dependency graph should be the
single reload authority.

If stronger isolation is needed, reuse the process boundary already present in
reference preview. Rebuilding the framework inside a long-lived interpreter is
the least efficient and least reliable option.

### P1: Cancellation Discards Results But Not Expensive Work

The reference preview controller uses a one-worker process pool behind a
one-worker launch thread. A stale request can be ignored, but the worker keeps
running and later requests wait behind it. `shutdown(wait=False)` does not kill
running process futures.

Consequences:

- rapid editing accumulates latency behind irrelevant builds;
- CPU and memory remain occupied by stale geometry;
- shutdown can wait indefinitely for child completion;
- test processes can hang after all assertions pass.

Recommended design:

- one owned worker process per replaceable request, or a worker protocol that
  can cooperatively cancel inside bounded kernel checkpoints;
- terminate and reap on timeout;
- coalesce before process submission, not only at result delivery;
- record queue wait, execution time, cancellation latency, and child count.

### P1: Text Converts Curve Sampling Into Topological Complexity

`text_profiles()` preserves font outline curves initially, but
`_surface_text_extrude()` eventually samples them and creates one ruled patch
per sampled loop edge. Tessellation density therefore determines authored shell
size.

Fresh local baseline for default Arial, `make_text("SURFACE")`,
`font_size=10`, `depth=1`:

| Operation | Result |
| --- | ---: |
| Construction | 0.067 s |
| Surface patches | 2,365 |
| One `stable_identity` access | 0.082 s |
| Preview tessellation | 1.277 s |
| Vertices | 9,404 |
| Faces | 9,384 |

Using the repository's simpler Noto Symbols fixture font produces only 70
patches for the same letters, demonstrating that complexity is driven by outline
segmentation and sampling rather than content length.

Recommended repair:

- retain line, quadratic, cubic, and spline boundaries as curve truth;
- use one swept/ruled side surface per logical curve segment;
- make tessellation density a consumer parameter only;
- cache shaped glyph outline records by font file identity and glyph;
- establish patch-count and latency budgets for representative fonts.

### P1: Tessellation Repeatedly Re-derives Shell Facts

High-patch shells cause repeated seam participation and sampling decisions.
`_patch_requires_shell_grid_tessellation()` and
`_shell_grid_counts_for_patch()` operate from individual patches while needing
shell-wide seam truth.

Stable identities also recursively serialize and hash complete payloads on each
property access. Body tessellation reads body identity multiple times, and shell
and patch identities compose beneath it.

Recommended repair:

- build one immutable tessellation context per shell containing seam
  participation, compatible edge counts, boundary ownership, and identities;
- calculate each operation-scoped identity once;
- only memoize identity after geometry is genuinely immutable;
- distinguish geometry identity from mutable display metadata;
- benchmark high-patch text, large lofts, and mixed-family CSG.

### P2: Numerical And Transform Kernels Have Multiple Owners

Concrete duplication:

- `_resample_path()` exists in `modeling/loft.py:5574` and
  `modeling/sdf.py:158`;
- B-spline span/basis evaluation exists in `modeling/bspline.py:105-143` and
  again near `modeling/surface.py:2432-2467`;
- matrix construction and axis rotation are separately implemented in
  `modeling/group.py:22-106` and `modeling/transform.py:176-207`;
- VTK inspection and mutation is duplicated in `impression/__init__.py` and
  `_vtk_runtime.py`;
- `Path`, `Path3D`, and `Path2D` provide overlapping path concepts without one
  shared parametric path protocol.

Duplicate numerical kernels are especially expensive because endpoint,
tolerance, and degeneracy fixes must be repeated and can create route-dependent
geometry.

Recommended repair: create small internal math/path/transform modules, test
their invariants once, and keep route-specific adaptation above them.

### P2: The Modeling Core Mixes Records, Policy, Evidence, And Execution

`csg.py`, `surface.py`, and `loft.py` contain hundreds of records and functions
covering:

- canonical geometry values;
- support matrices and policy;
- execution plans;
- diagnostics and refusal evidence;
- reference-fixture verification;
- persistence payloads;
- user-facing operations.

The largest loft planning and validation functions exceed 400 lines. The
monolithic test files mirror the same ownership: `test_surface_csg.py` is 5,954
lines, `test_surface.py` 5,058, and `test_loft.py` 3,629.

This increases import cost, review cost, and the amount of code loaded for small
operations. More importantly, it prevents reuse because a narrow capability
cannot be imported without its complete policy and evidence environment.

Recommended decomposition:

- immutable geometry and topology records;
- family adapters and intersection kernels;
- planning/policy;
- execution;
- persistence schemas;
- diagnostics/evidence tooling;
- public facade and compatibility facade.

Do this behind contract tests, not as one mechanical rewrite.

### P2: Shadowed And Orphaned Modules Carry Cost Without Capability

Examples:

- `impression/cad.py` is shadowed by `impression/cad/`;
- `modeling/extrude.py` implements mesh extrusion/revolution while docs call the
  public path retired and the facade does not expose it;
- `validation.py` and `printability.py` have effectively no integrated product
  usage;
- build123d is mandatory even though the advertised adapter is unreachable;
- the packaged font is documented but not included in the wheel.

Every such path adds dependency, maintenance, or reader cost. Classify each as
supported, compatibility-only, experimental, or removable, then enforce that
classification in packaging and the facade.

### P2: Correspondence Assignment Has Explicit Exponential Ceilings

`minimum_cost_loop_assignment()` and
`minimum_cost_subset_assignment()` use bitmask dynamic programming and cap
problem size at 12. Loft ambiguity analysis also has bounded recursive
enumeration with a default branch budget of 64.

Explicit bounds are better than unbounded execution, but they create a sharp
scaling cliff for complex sections.

Recommended direction:

- use a polynomial assignment solver for the best base mapping;
- apply named anchors before inference;
- enumerate only alternatives near the ambiguity threshold;
- separate "find best" from "prove unambiguous";
- benchmark 8, 12, 20, and 50 regions/loops with deterministic budgets.

### P2: Font Discovery Repeats Host-Wide Recursive Scans

Without `font_path`, every `text_profiles()` call recursively scans five system
font roots and then opens a new `TTFont` (`modeling/text.py:268-277` and
`:367-392`). This is host-dependent and wasteful in interactive labeling.

Cache a font index with invalidation, cache parsed faces by resolved file
identity, and cache unscaled glyph outlines. A packaged default font would make
both discovery and output deterministic.

### P2: Serialization Reuse Is Incomplete

The modeling package contains 378 `canonical_payload()` methods. Explicit
schemas are valuable at persistence boundaries, but common normalization of
arrays, enums, transforms, metadata, and nested records is repeatedly
hand-authored.

Introduce shared canonical field encoders and schema helpers while keeping
versioned root/patch payloads explicit. This reduces boilerplate without hiding
format decisions.

### P3: There Is No Performance Regression Gate

Coverage scripts exist, but there is no benchmark suite or CI budget for:

- import and first preview;
- primitive tessellation;
- loft planning/execution;
- text construction and tessellation;
- CSG matrix exemplars;
- `.impress` save/load;
- reference preview cancellation and shutdown.

Without fixed fixtures and phase telemetry, optimization claims and regressions
will remain anecdotal.

## Reuse Decisions

### Reuse Existing Internal Work

- Make the reference preview's surface adapter the starting evidence for the
  unified scene consumer.
- Reuse the existing tessellation request objects for preview/export intent.
- Reuse local-module tracking for hot reload.
- Reuse atomic-write and containment helpers in all reference state changes.
- Reuse the older endcap algorithms as geometry references, not as mesh output
  copied into the surface kernel.

### Reuse External Libraries Deliberately

- Use a proven assignment implementation if it fits the dependency policy.
- Use HarfBuzz or a platform shaping engine for real typography.
- Use lib3mf or a proven 3MF implementation for standard conformance.
- Use build123d/Open CASCADE through one explicit adapter only where fidelity,
  units, and ownership are specified.

External reuse must not become another undeclared sibling or transitive
dependency.

## Required Efficiency Sequence

1. Add stable end-to-end benchmarks and phase telemetry.
2. Unify scene traversal and consumer conversion.
3. Narrow hot reload to user-owned modules and make process cancellation real.
4. Precompute shell tessellation context and operation identities.
5. Preserve curve-level text topology and cache font/glyph data.
6. Consolidate numerical, transform, path, and VTK owners.
7. Split CSG/surface/loft by ownership behind API and persistence tests.
8. Replace bounded exponential base assignment where measured workloads require
   it.
9. Remove or quarantine shadowed and orphaned code and dependencies.

The first milestone should reduce duplicated conversion paths and produce
repeatable numbers. Micro-optimizing individual NumPy calls before that would
leave the dominant structural costs intact.
