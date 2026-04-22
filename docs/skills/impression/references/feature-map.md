# Impression Feature Map

Use this file when you need to decide which Impression feature family fits a
modeling task.

## Start With The Repo Docs

If the workspace contains these docs, read them before guessing:

- `docs/index.md`
- `docs/modeling/*.md`
- `docs/examples/**`

## Feature Families

### Primitives

Use for direct surfaced solids:

- box
- cylinder
- cone
- sphere
- torus
- prism / polyhedron families

Read:

- `docs/modeling/primitives.md`

### Drawing2D

Use for authored 2D shapes and paths that feed loft, text-adjacent profile
work, and planar modeling.

Read:

- `docs/modeling/drawing2d.md`

### Topology

Use when you need explicit sections, regions, or loops, or when another feature
expects topology-native inputs.

Read:

- `docs/modeling/topology.md`

### Loft

Use for profile-to-profile transitions, explicit station sequences, and
topology-changing section evolution.

Read:

- `docs/modeling/loft.md`
- this skill's [loft reference](loft.md)

### Text

Use for glyph outlines, text sections, and surfaced text bodies.

Read:

- `docs/modeling/text.md`

### CSG

Use for boolean composition, but read the current execution posture first.
In this project, surfaced CSG exists, but the executable scope may still be
bounded and narrower than the mesh lane.

Read:

- `docs/modeling/csg.md`

### Threading

Use for surfaced thread operands, assemblies, and standards-based thread
geometry.

Read:

- `docs/modeling/threading.md`

### Hinges

Use for traditional, living, and bistable hinge outputs.

Read:

- `docs/modeling/hinges.md`

### Drafting

Use for surfaced drafting helpers and 2.5D annotation geometry.

Read:

- `docs/modeling/drafting.md`

### Heightmaps

Use for heightfield generation and image-based displacement workflows.

Read:

- `docs/modeling/heightmaps.md`

## General Modeling Rules

- Keep geometry creation in `impression.modeling`.
- Prefer the documented API over ad hoc geometry math.
- Reuse examples whenever a close example exists.
- Stay surface-first when the feature supports it.
