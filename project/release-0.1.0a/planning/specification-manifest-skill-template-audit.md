# Specification Manifest Skill And Template Audit

## Purpose

Audit the specification-manifest skill and templates to decide whether
Impression needs project-local template overrides for the new release workspace
process.

## Audited Files

Skill bodies:

- `../.agents/skills/specification-manifest/SKILL.md`
- `~/.codex/skills/keld-specification-manifest/SKILL.md`

Shared templates:

- `../.agents/process/skills-templates-manifest.md`
- `../.agents/process/templates/manifest-entry-template.md`
- `../.agents/process/templates/implementation-spec-template.md`
- `../.agents/process/specification-manifest-scoring-policy.md`

Repo-local generated skill tree:

- `.agents/skills/`

## Findings

### 1. The updated manifest skill exists, but not in repo-local generated skills

The updated manifest-capable skill exists in:

- parent shared layer: `../.agents/skills/specification-manifest/SKILL.md`
- home layer: `~/.codex/skills/keld-specification-manifest/SKILL.md`

The repo-local generated runtime tree does not currently include:

```text
.agents/skills/specification-manifest/SKILL.md
```

That means the local generated skill set is stale relative to the shared
manifest workflow.

### 2. The current skill already supports template overrides

The skill says to prefer:

```text
project/process/skills-templates-manifest.md
```

and then fall back to the shared registry:

```text
.agents/process/skills-templates-manifest.md
```

Conceptually, that is the right behavior: project-local templates should
override shared templates.

### 3. `project/process/` conflicts with the cleaned project folder shape

The new `project/` shape intentionally separates durable project knowledge from
active release work:

```text
project/
  future-features/
  meetings/
  research/
  releases/
  release-n.n.n/
```

Adding `project/process/` would reintroduce a top-level process folder inside
`project/`, even though agent/process mechanics are better housed under
`.agents/`.

### 4. The shared templates are good enough for current use

The shared manifest-entry template already captures the fields Impression needs
for architecture-to-spec discovery:

- responsibilities by category
- reusable code plan
- implementation owner/module
- chosen defaults / parameters
- test strategy
- data ownership
- routes
- reuse/extraction decision
- UI field/control inventory
- scoring
- readiness blockers
- split decision

The shared implementation-spec template also covers the important readiness
areas:

- implementation routing
- chosen defaults
- data ownership
- dependencies and routes
- reuse/extraction plan
- required DTOs/functions/components
- performance contract
- error and state behavior
- test strategy
- acceptance criteria

No immediate Impression-specific override is required for the loft topology
point-correspondence work.

### 5. The shared templates could be clearer for release work, but should stay generic

The shared templates do not explicitly mention active release folders such as:

```text
project/release-0.1.0a/
```

That is acceptable. Release path selection is an Impression overlay concern,
not a shared template concern.

## Decision

Use the existing shared templates for now.

Do not create project-local template overrides yet.

If Impression later needs local overrides, place them under:

```text
.agents/process/
```

not under:

```text
project/process/
```

This keeps project-management process machinery out of the cleaned `project/`
folder and keeps `project/releases/README.md` as the durable description of the
release workspace lifecycle.

## Recommended Skill Update

The specification-manifest skill lookup rule has been updated so it checks for
local overrides in this order:

1. `.agents/process/skills-templates-manifest.md` in the current repository
2. `project/process/skills-templates-manifest.md` only for older projects that
   already use that convention
3. nearest parent shared `.agents/process/skills-templates-manifest.md`

The skill should say that repo-local `.agents/process/` is the preferred
override location for projects that keep process machinery outside `project/`.

Updated skill files:

- `../.agents/skills/specification-manifest/SKILL.md`
- `~/.codex/skills/keld-specification-manifest/SKILL.md`
- `~/.codex/skills/keld-spec-manifest/SKILL.md`

## Recommended Template Policy

Use shared templates unless a project-specific need appears.

Create repo-local template overrides only when:

- a required field is repeatedly added manually to manifest entries
- scoring needs project-specific categories
- implementation specs need mandatory project-specific routing fields
- release workspace paths must be encoded in generated spec templates
- a local process policy intentionally diverges from the shared one

## Acceptance Checks

This audit is complete when:

- current work can use the shared templates without missing required fields
- the agent directive consistency work records that repo-local generated skills
  need refresh
- any future local override location is `.agents/process/`, not a new top-level
  `project/process/` folder
