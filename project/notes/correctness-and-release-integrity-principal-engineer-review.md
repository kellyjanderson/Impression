---
created: 2026-07-23
---

# Principal Engineer Review: Correctness And Release Integrity

## Review Frame

This supplemental repository-wide review preserves the correctness and release
findings that were initially, and incorrectly, placed in the code-quality
review. It does not use the earlier discussion of correspondence, text, or
endcaps to decide priority. Those areas are treated like every other subsystem
and surface only where their evidence warrants it.

This pass evaluates runtime correctness, security, contract integrity, failure
behavior, testability, and release truth. Readability and engineer experience,
performance and reuse, and product breadth are covered independently in the
three principal-engineer review passes.

Severity:

- `P0`: release blocker, security boundary failure, or canonical workflow failure
- `P1`: high-risk correctness or architectural defect
- `P2`: material maintainability, packaging, or test-quality weakness
- `P3`: bounded cleanup

## Repository Coverage

The review inventoried 89 Python source modules, 67,499 source lines, 77 test
files, packaging and installer scripts, both GitHub workflows, documentation
examples, and the reference-review application.

| Area | Examined |
| --- | --- |
| Product boundary | installer, wheel, imports, CLI, preview, STL, `.impress` |
| Geometry kernel | primitives, transforms, topology, surface, loft, CSG, tessellation |
| Secondary modeling | text, drafting, heightmap, SDF, paths, splines, drawing |
| Developer tooling | reference discovery, async work, promotion, UI, evidence |
| Assurance | tests, CI, release workflow, docs examples, coverage configuration |

## Verdict

The checkout is not release-ready. The most serious problems are not confined
to a single modeling feature:

- loading a user model corrupts Python class identity across the modeling stack;
- the documented first preview/export workflow rejects the canonical
  `SurfaceBody` returned by public primitives;
- the docs downloader permits archive path traversal;
- documented mesh CSG examples call an API that rejects meshes;
- most representative solid primitives are intentionally emitted as open shells;
- export bypasses the repository's watertight policy and does not apply units;
- the full test signal is order-dependent, incomplete in CI, and can abort in Qt.

The repository contains unusually rich diagnostic and surface-planning work,
but its integration boundaries do not yet preserve that kernel truth.

## Findings

### P0: User-Model Loading Creates A Split-Brain Modeling Runtime

`src/impression/cli.py:55-82` deletes `impression.modeling` and every submodule
from `sys.modules` whenever it loads a Python model. Existing modules retain the
old `SurfaceBody`, `SurfaceShell`, and patch classes; newly imported code gets
new class objects with the same names.

The result is order-dependent type corruption across drafting, heightmaps,
transforms, CSG, I/O, and lofts. A minimal test-order reproduction is:

```text
.venv/bin/python -m pytest -q tests/test_cli_preview.py tests/test_drafting.py
6 passed, 1 failed

TypeError: SurfaceBody shells must all be SurfaceShell instances.
```

`tests/test_drafting.py` passes by itself. In a broader run, the same pollution
produced false-looking failures such as
`isinstance(HeightmapSurfacePatch, HeightmapSurfacePatch) == False`.

This mechanism also runs in the long-lived preview process, so it is a product
defect, not just a test fixture problem.

Required repair:

- reload only the user model and its tracked local dependencies;
- never unload Impression's own package modules;
- isolate model execution in a subprocess if full dependency replacement is
  required;
- add repeated-load tests that retain and reuse pre-load geometry values.

### P0: The Canonical First Preview And Export Contract Is Broken

The README states that public primitives return `SurfaceBody` and that preview
and export tessellate at the consumer boundary (`README.md:62-81`).
`make_box()` does return `SurfaceBody`.

However, `src/impression/preview.py:628-696` accepts mesh, path, and topology
values but not `SurfaceBody`, `SurfaceConsumerCollection`,
`SurfaceComposition`, or surface boolean results. The reference-review adapter
does accept surface bodies at
`src/impression/devtools/reference_review/preview_payload_builder.py:169-230`.

Observed canonical command:

```text
impression export docs/examples/primitives/box_example.py \
  --output /tmp/box.stl --overwrite

Invalid value: Model build() must return internal meshes...
```

The README's highlighted CSG export example fails earlier for a separate CSG
contract defect. There is no end-to-end CLI test for a public primitive
returning a `SurfaceBody`.

Required repair: define one scene-to-consumer protocol used by preview, export,
reference preview, and `.impress`; test the exact README commands against the
installed wheel.

### P0: Docs Archive Extraction Permits Path Traversal

`src/impression/cli.py:148-183` appends each archive-controlled relative path
directly to the destination. It does not reject `..`, absolute paths, symlinks,
or a resolved target outside the destination.

A local proof archive containing `docs/../escaped.txt` wrote
`escaped.txt` outside the requested docs directory. The repository URL and ref
are caller-controlled CLI inputs. Downloads at `src/impression/cli.py:190-235`
also have no timeout or response-size limit and are read fully into memory.

Required repair:

- reject unsafe member types and names before writing anything;
- resolve every target and require containment under the destination;
- bound download and expanded archive sizes;
- use network timeouts;
- extract into a temporary directory and atomically publish on success;
- add adversarial archive tests.

### P0: Public CSG Signatures And Examples Do Not Match Runtime Behavior

`boolean_union`, `boolean_difference`, and `boolean_intersection` advertise
mesh and surface operands. The CSG documentation and
`docs/examples/csg/*.py` use mesh inputs. In
`src/impression/modeling/csg.py:18621-18677`, non-surface operands fall through
to surface preparation, which raises `TypeError`.

Observed result:

```text
boolean_union([make_box_mesh(), make_box_mesh(...)])
TypeError: union operand 0 must be a SurfaceBody
```

`union_meshes()` works, but that is not the documented example API.
`docs/modeling/csg.md` also describes both mesh-primary and surface-primary
contracts.

Required repair: choose and enforce one public dispatch contract, make mesh
compatibility explicit, run every documentation example, and remove
contradictory guidance.

### P0: Release Gates Do Not Verify The Distribution Contract

The full suite currently fails collection because documented hinge symbols are
not exported. `src/impression/modeling/hinges.py` imports an undeclared sibling
package, while `pyproject.toml` does not declare it.

CI runs only five of 77 test files on Python 3.13
(`.github/workflows/ci.yml:54-61`). The release workflow builds and uploads
artifacts without tests or an installed-wheel smoke test
(`.github/workflows/release.yml:25-43`).

The canonical installer compounds this:

- it installs a hand-selected dependency subset and then the wheel with
  `--no-deps` (`scripts/dev/install_impression.sh:368-378`);
- that subset omits declared `build123d` and `scikit-image`;
- it validates `python -m impression.cli --version`, but `impression.cli` has no
  module entry point, so the command exits zero without invoking Typer
  (`scripts/dev/install_impression.sh:381`);
- it checks that the executable exists but does not execute it.

Broad local evidence:

```text
full collection: 1560 tests, 4 collection errors
bounded broad run: 30 failed, 617 passed, stopped at maxfail=30
separate broad run: fatal Python abort in Qt at 46%
```

Many of the 30 failures are consequences of the split-brain import defect, which
is itself evidence that the suite is not isolated.

Required repair:

- clean-environment dependency and `pip check` validation;
- execute the installed `impression` entry point and canonical model export;
- require full collection and a stable non-GUI suite;
- isolate Qt/process tests into explicit jobs;
- test the minimum supported Python and release Python;
- make release publication depend on verification of the exact artifact.

### P1: "Solid" Primitive Output Frequently Is Not Closed

The public surface constructors for cylinder, cone, prism, nhedron, ngon, and
polyhedron use caps that are not seam-connected to sidewalls. The cylinder
implementation acknowledges this at
`src/impression/modeling/_surface_primitives.py:238-244`.

Fresh default probes:

| Primitive | Surface classification | Watertight | Boundary edges |
| --- | --- | ---: | ---: |
| box | closed | yes | 0 |
| cylinder | open | no | 376 |
| cone | open | no | 124 |
| sphere | closed | yes | 0 |
| torus | closed | yes | 0 |
| prism | open | no | 16 |
| nhedron | open | no | 24 |

Tests currently assert that several of these results are open, so this is a
blessed contract mismatch rather than an uncovered accident. It contradicts the
README's watertight STL positioning.

### P1: Export Bypasses Watertight And Unit Contracts

`src/impression/cli.py:636-693` routes export through
`PyVistaPreviewer.collect_datasets()`, combines preview meshes, and writes STL.
It does not use `export_tessellation_request()`, whose policy requires
watertight output.

Configured unit scales exist in `src/impression/_config.py:15-17`, but the only
uses are display/logging properties. `src/impression/io/stl.py:30-76` writes raw
coordinates. An authored value of `1` in inch mode is therefore written as `1`,
not `25.4`, while the CLI reports that units are inches.

Export also does not use the `.impress`-aware scene loader. A unitless format
plus an unapplied unit setting is silent dimensional corruption.

Required repair:

- create a dedicated export consumer using export tessellation policy;
- reject open, nonmanifold, invalid, or wrongly oriented output by default;
- convert configured model units to the declared STL convention;
- include `.impress` in the same export path;
- test output coordinates by reading the written artifact.

### P1: Reference Promotion Is Not Atomic Despite Its Contract

`PromotionExecutor` says it atomically promotes artifacts
(`src/impression/devtools/reference_review/lifecycle.py:166-198`) but copies
gold files one at a time. A late copy or checksum failure leaves earlier gold
files changed. Checksums are keyed only by artifact kind, so duplicate kinds
overwrite one another.

`approve_reference_artifacts()` is more destructive
(`src/impression/devtools/reference_review/source_registry.py:668-700`): it
unlinks existing gold and moves dirty artifacts sequentially, then can return a
failure after partial promotion. Several JSON state writes use direct
`write_text()` rather than the existing atomic-write helper.

The durable lock lane serializes callbacks, but it does not make a multi-file
transaction atomic and has no stale lock ownership recovery after a crash.

Required repair: prevalidate the complete plan, stage all artifacts, fsync as
appropriate, publish atomically with rollback, record provenance in the same
transaction, and add injected-failure tests at every step.

### P1: Mirroring Mesh Output Reverses Solid Orientation

The mesh compatibility transform applies a negative-determinant matrix without
reversing face winding. A mirrored centered box produced:

```text
outward faces: 0
inward faces: 12
analyze_mesh().is_watertight: True
```

`analyze_mesh()` counts undirected edge incidence but does not validate
oriented edge pairing or signed volume. The STL writer recomputes normals from
the inverted faces, preserving the wrong orientation. The mirror test checks
bounds only.

Negative scale has the same risk. Transform code must detect orientation
reversal, repair winding, and validate outward orientation.

### P1: Preview Worker Cancellation Does Not Cancel Work

`PreviewPayloadProcessController` creates a process pool and a launch thread
(`preview_payload_controller.py:132-150`). Cancellation can mark requests stale,
but the launch thread waits on `future.result()` and `close()` calls
`shutdown(wait=False, cancel_futures=True)` on already-running work.

Running process futures are not canceled. Stale model builds continue, newer
requests queue behind them, and interpreter shutdown can wait for workers. The
exact five-file CI selection reached 79 passing tests locally but remained in
process-pool shutdown until interrupted.

The controller needs killable worker ownership, bounded cancellation latency,
and shutdown tests that inspect child processes.

### P1: Importing The Package Mutates External State

`src/impression/__init__.py:17-105` sets environment variables, runs `nm`,
renames/unlinks/symlinks files inside the VTK installation, and initializes
`~/.impression` configuration during import. The VTK logic is duplicated in
`src/impression/_vtk_runtime.py`.

Import should be observational. Installation repair and user-state creation
belong in explicit commands with diagnostics, especially in read-only,
packaged, and multi-process environments.

### P2: CAD Packaging Contains A Shadowed, Unreachable Adapter

The wheel contains both `impression/cad.py` and
`impression/cad/__init__.py`. Python resolves `impression.cad` to the package,
whose `shape_to_polydata()` always raises "disabled in this build." The working
build123d adapter in `cad.py` is unreachable.

The README advertises `examples/half_pipe.py`, which imports that symbol.
Meanwhile build123d remains a mandatory dependency. This is dead code,
namespace ambiguity, and a broken advertised integration in one boundary.

### P2: Dependency And Artifact Metadata Drift

NumPy is imported directly throughout the package but is not declared directly
in `pyproject.toml`; it is available only transitively. The documented
`assets/fonts/NotoSansSymbols2-Regular.ttf` is outside package data and absent
from the built wheel. Default text instead searches host fonts for Arial.

The project also lacks lint, formatting, static typing, and API-surface checks
in CI. These would not replace tests, but they would detect several boundary
and packaging problems earlier.

### P2: Frozen Geometry Still Contains Mutable Identity State

Surface values are frozen dataclasses, but constructor arrays can remain
caller-owned and writable, and metadata is only shallow-copied. Mutating an
input after construction can change `stable_identity`.

That undermines deterministic persistence, cache keys, CSG ordering, and any
identity memoization. Copy and freeze arrays, canonicalize kernel metadata, and
separate mutable display metadata from geometric identity.

### P2: Ownership Boundaries Are Too Large To Review Reliably

Measured source shape:

| Module | Lines | Functions | Classes |
| --- | ---: | ---: | ---: |
| `modeling/csg.py` | 18,699 | 654 | 226 |
| `modeling/surface.py` | 8,170 | 417 | 112 |
| `modeling/loft.py` | 7,815 | 333 | 69 |
| `io/impress.py` | 2,745 | 94 | 31 |
| `modeling/__init__.py` | 2,129 | - | - |

The modeling facade exports 1,018 names. Several functions exceed 400 lines.
This is not merely aesthetic: accidental public APIs, circular type ownership,
duplicated policy, and broad reload behavior become difficult to reason about.

Split by execution ownership, define an intentional public manifest, and keep
compatibility exports separate from kernel records.

## Positive Foundations

- Surface routes generally prefer explicit refusal to hidden mesh fallback.
- Structured diagnostics and canonical payloads support reproducible failure
  analysis.
- `.impress` validates schema and unsafe implicit-field payload budgets.
- The test corpus is large and often physically specific even though its global
  execution is currently unreliable.
- The reference-review stack already contains useful containment and durable
  write primitives; promotion needs to consistently use them.

## Required Correctness Sequence

1. Remove internal-package unloading from user-model loading.
2. Unify preview/export scene consumption and restore the README quickstart.
3. Fix archive extraction before shipping the docs downloader.
4. Make solid primitives, transforms, units, and export validation physically
   correct.
5. Reconcile CSG, CAD, hinge, dependency, and packaged-font contracts.
6. Make reference promotion transactional and preview workers killable.
7. Establish full collection, isolated test layers, installed-wheel smoke tests,
   and release gating.
8. Then reduce facade and module scope without mixing that work into urgent
   correctness repairs.

Release should remain blocked until steps 1 through 4 and the release-gate
portion of step 7 are complete.
