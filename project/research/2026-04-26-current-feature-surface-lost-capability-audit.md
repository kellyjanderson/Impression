# Current Feature Surface Lost-Capability Audit

## Topic

Audit Impression as it exists today for modeling capabilities that may have
been lost, stranded, or only partially ported during the surface-first
rearchitecture.

## Scope

This report is narrower than a roadmap.

It asks:

> Which concrete capability classes appear to have been intentionally removed,
> partially stranded, or accidentally left without a surfaced/public
> replacement?

The goal is to separate:

- intentional removals
- known bounded unsupported areas
- likely lost or unported capability classes

## Current Public Surface Snapshot

The public modeling namespace currently includes:

- primitives
- drawing2d
- topology
- loft
- text
- hinges
- drafting
- transforms
- groups
- bounded surfaced CSG
- `Path`
- `Path3D`

Evidence:

- [docs/index.md](../../docs/index.md)
- [src/impression/modeling/__init__.py](../../src/impression/modeling/__init__.py)

## Category 1: Intentional Removals

These do not currently look accidental.

### Morph

`morph` was explicitly removed from the supported modeling surface.

Evidence:

- [docs/modeling/morph.md](../../docs/modeling/morph.md)
- [tests/test_morph.py](../../tests/test_morph.py)
- [CHANGELOG.md](../../CHANGELOG.md)

Assessment:

- lost capability from an end-user point of view
- not an accidental omission
- should not be treated as a porting bug

### Public Extrude

Public `linear_extrude()` and `rotate_extrude()` were explicitly retired.

Evidence:

- [docs/modeling/extrusions.md](../../docs/modeling/extrusions.md)
- [CHANGELOG.md](../../CHANGELOG.md)

Assessment:

- also an intentional removal
- but unlike morph, this leaves behind a likely workflow gap because internal
  surfaced extrusion still exists

## Category 2: Internal Capability Exists, But Public Replacement Looks Missing

These are the strongest candidates for “not fully ported.”

### Profile-To-Solid Extrusion / Revolution Workflow

Internal surfaced extrusion and revolution builders exist:

- `make_surface_linear_extrude`
- `make_surface_rotate_extrude`

Evidence:

- [src/impression/modeling/_surface_ops.py](../../src/impression/modeling/_surface_ops.py)
- [tests/test_surface.py](../../tests/test_surface.py)

But they are intentionally hidden from the public modeling namespace:

- [tests/test_surface.py](../../tests/test_surface.py)

And the public docs say extrusion is retired:

- [docs/modeling/extrusions.md](../../docs/modeling/extrusions.md)

Assessment:

- this is the clearest current “stranded capability” class
- the kernel support exists
- the public surfaced replacement is missing
- text already depends on the private extrusion lane, which shows the lane is
  real and useful

Recommended interpretation:

- likely not an accident in the narrow sense
- but it does look like an incompletely rehomed feature class

### Generic Revolved/Profile-Based Surface Construction

Revolution patches are part of the required surfaced patch families:

- [src/impression/modeling/surface.py](../../src/impression/modeling/surface.py)

And internal rotate-extrude support is implemented.

But outside primitives and private ops, there is no general public surfaced
workflow for:

- revolving arbitrary profile sections
- creating user-facing surfaced bodies from authored profiles

Assessment:

- closely related to the public extrusion gap
- likely same missing surfaced replacement family rather than a separate issue

## Category 3: Capability Classes Referenced By Docs Or Planning But Not Truly Present

These look more like unported or never-finished feature classes than removals.

### Sweep / Pipe / Path-Driven Solid Modeling

Multiple places imply sweep-like capability:

- `Path` docs say the path utility is for sweeps and sampling
- `Path3D` docs say it is intended for sweeps/lofts and future tooling
- deferred surfaced patch families include `"sweep"`
- planning docs still mention extrude/sweep along path

Evidence:

- [docs/modeling/paths.md](../../docs/modeling/paths.md)
- [docs/modeling/path3d.md](../../docs/modeling/path3d.md)
- [src/impression/modeling/surface.py](../../src/impression/modeling/surface.py)
- [project/planning/README.md](../planning/README.md)

But there is no public sweep or pipe modeling feature in the active namespace.

Assessment:

- the surrounding abstractions do gesture toward this capability class
- but current planning direction now treats this as intentionally subsumed by
  loft enhancement rather than as a separate first-class product line
- so this should not be read as “we need a separate sweep feature track”

Planning correction:

- loft progression, trajectory, and transport enhancement are the intended
  replacement path for these use cases
- the missing piece is better loft semantics, not a distinct sweep/pipe branch

### First-Class Smooth Path / B-Spline Path Support

`Path` has `to_spline(...)`, but only as sampled polyline output.

`Path3D` supports:

- line
- arc
- bezier

But not B-spline or another high-control smooth path primitive.

Evidence:

- [docs/modeling/paths.md](../../docs/modeling/paths.md)
- [docs/modeling/path3d.md](../../docs/modeling/path3d.md)
- [src/impression/modeling/path3d.py](../../src/impression/modeling/path3d.py)

Assessment:

- not a regression from an earlier shipped surfaced feature
- but definitely a missing capability that makes the current path surface feel
  thinner than the docs and research now want

## Category 4: Bounded Unsupported Areas That Should Stay Visible

These are not necessarily lost features, but they are real capability gaps in
today’s product surface.

### Surfaced CSG Partial-Overlap Coverage

Surfaced CSG remains bounded. Partial box/sphere overlap is still explicitly
unsupported.

Evidence:

- [docs/modeling/csg.md](../../docs/modeling/csg.md)
- [tests/test_surface_csg.py](../../tests/test_surface_csg.py)

Assessment:

- not a lost feature
- clearly a current kernel boundary
- should remain visible in feature planning

### Loft Many-To-Many Ambiguity

Loft still explicitly rejects true many-to-many split/merge ambiguity.

Evidence:

- [docs/modeling/loft.md](../../docs/modeling/loft.md)
- [src/impression/modeling/loft.py](../../src/impression/modeling/loft.py)

Assessment:

- not accidental
- important current boundary
- relevant to future inference work, but not a missing port of an older public
  capability

## Strongest Lost-Or-Stranded Capability Candidates

If the question is “what may have been lost or not fully ported,” the strongest
answers are:

### 1. Public surfaced profile extrusion / revolution workflows

Why:

- internal implementation exists
- public feature was retired
- no new surfaced replacement is exposed
- real subsystems already depend on the private lane

This is the most concrete candidate for a feature class that may have been
over-removed before the surfaced public replacement was ready.

### 2. Sweep / pipe / path-driven body construction

Why:

- docs and planning still frame paths as feeding sweeps
- deferred patch metadata includes sweep
- no actual public surfaced sweep feature exists

Updated interpretation:

- not a separate `0.1.0.a` feature gap
- this capability should be absorbed by loft progression/path enhancement rather
  than by adding an independent sweep/pipe feature family

### 3. First-class smooth path primitives beyond bezier

Why:

- the current path layer tops out too early for the research direction
- `to_spline(...)` is not a first-class path primitive
- future trajectory-guided and inferred-curve work clearly wants more

This is not an accidental loss from a prior public feature, but it is a
meaningful missing low-level consumer capability.

## Features That Look Intentionally Gone Rather Than Accidentally Lost

These should not be treated as lost-port bugs:

- public `morph`
- legacy public mesh-primary extrude module

They may deserve replacement strategies, but the repo state shows explicit
intentional retirement rather than accidental omission.

## Recommendation

For planning purposes, the repo should distinguish three buckets:

1. intentionally removed and not to be restored as-is
   - morph
   - legacy public mesh extrude
2. stranded internal surfaced capability that likely needs a public surfaced
   replacement
   - profile extrusion
   - profile revolution
3. capability classes that the surrounding architecture already points toward
   but that are not actually present yet
   - loft-owned path/progression enhancement for sweep-like cases
   - first-class B-spline path support

If `0.1.0.a` stays focused on inference and curve fitting, the sweep/pipe-like
cases should be pulled inward into loft enhancement rather than split outward
into a separate feature line. The still-open follow-on question is narrower:

> Should profile-to-solid surfaced construction reappear as a supported public
> modeling path once the public API is redesigned around the surface-first
> posture?

## Conclusion

The rearchitecture does not appear to have accidentally lost a large number of
major public features.

The most important current gap is more specific:

- public surfaced profile extrusion and revolution look stranded
- sweep/pipe should be treated as loft-enhancement territory rather than a
  missing separate feature branch
- richer smooth path support is still missing

Those are the strongest candidates for “not fully ported” capability classes in
Impression today.
