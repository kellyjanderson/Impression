# Agent Directive Consistency Plan

## Purpose

Align agent-facing directives with the new `project/` lifecycle:

- durable project knowledge stays in top-level durable folders
- active release work lives under `project/release-n.n.n/`
- completed release work moves under `project/releases/`
- issue tracking lives in GitHub Issues, not a local `project/issues/` folder

This plan is intentionally scoped to directive and skill cleanup. It does not
change product behavior.

## Audit Scope

Reviewed layers:

- repository-local source directives: `.agents/`
- repository-local non-skill directive notes: `.agents/index.md` and
  `.agents/potentials/**/*.md`
- repository-local generated runtime skills: `.agents/skills/`
- parent shared project-process skills: `../.agents/skills/`
- parent process templates: `../.agents/process/`
- home-level Codex skills: `~/.codex/skills/`

## Current Shape

### Repository-Local `.agents/`

The repository has both source directive folders and generated runtime output:

- source overlays:
  - `.agents/coverage-discipline/`
  - `.agents/delegation/`
  - `.agents/git-and-github-overlay/`
  - `.agents/modern-surface-first-guardrails/`
  - `.agents/reference-artifact-lifecycle/`
  - `.agents/session-handoff/`
  - `.agents/workflow-impression-overlay/`
  - `.agents/workspace-index/`
- generated runtime tree:
  - `.agents/skills/`

`.agents/skills/.system-skills-composer.json` says the runtime tree is generated
and source skills should be edited in the custom hierarchy, not directly in
`.agents/skills/`.

Additional local markdown directives exist outside the active skill set:

- `.agents/index.md`
- `.agents/potentials/index.md`
- `.agents/potentials/modeling/README.md`
- `.agents/potentials/modeling/how-to-model.md`
- `.agents/potentials/modeling/model-planning-and-specs.md`
- `.agents/potentials/modeling/detail-and-completeness.md`
- `.agents/.skillskeeper-disabled/20260517-142431/specifications-core/SKILL.md`

These are not generated runtime skills, but they are still local agent
directives or directive candidates and must be part of cleanup decisions.

### Parent `../.agents/`

The parent shared layer contains generic project-process skills:

- architecture
- coding
- documentation
- git-and-github-core
- planning
- release-definitions
- research
- specification-manifest
- specification-sizing
- test-specifications-core
- workflow-core

These are mostly project-agnostic and should remain shared. They should not
encode Impression's release-folder layout directly.

### Home `~/.codex/skills/`

Home contains namespaced `keld-*` skills that overlap with parent/shared skills:

- `keld-architecture`
- `keld-planning`
- `keld-release-definitions`
- `keld-research`
- `keld-workflow-core`
- related coding, UI, and Impression skills

These are useful as personal/global defaults, but they should not own
per-project release structure. Per-project structure belongs in the repository
or nearest shared project layer.

## Findings

### 1. Repo-local overlays still reference old top-level release paths

The following source skills contain stale paths:

- `.agents/coverage-discipline/SKILL.md`
  - references `project/coverage/`
  - references `project/planning/progression.md`
- `.agents/reference-artifact-lifecycle/SKILL.md`
  - references `project/reference-images/`
  - references `project/reference-stl/`
- `.agents/git-and-github-overlay/SKILL.md`
  - allows `project/adhoc/`
  - allows version-planning structures under `project/planning/`
- `.agents/workflow-impression-overlay/SKILL.md`
  - describes planning structure merged to `main`, but not the new
    `project/release-n.n.n/` lifecycle
- `.agents/workspace-index/SKILL.md`
  - says shared skills live under `/Users/k/Documents/Projects/agents/`; the
    actual parent shared layer in this workspace is `../.agents/skills/`
  - describes project documents generically but does not mention the active
    release workspace boundary

Generated counterparts under `.agents/skills/` repeat the same stale content.

### 2. Parent shared skills are mostly acceptable as generic policy

Parent skills describe broad concepts such as architecture, research, planning,
release definitions, and durable planning anchors.

They do mention generic project-local paths such as:

- `project/research/`
- `project/process/skills-templates-manifest.md`
- `.agents/process/templates/manifest-entry-template.md`

Those references are acceptable as generic conventions as long as they remain
fallbacks or examples, not hard requirements for every project's release layout.

### 3. Home `keld-*` skills overlap with shared and repo-local skills

Home skills duplicate much of the parent shared layer. This is not inherently
wrong, but it creates two risks:

- agents may load the personal `keld-*` version when the project-local or shared
  unprefixed version is more specific
- personal skills may accidentally become a hidden source of project-management
  policy

The home layer should remain project-management-light.

### 4. Disabled skill archives are noisy but not currently harmful

Both repo and parent layers include `.skillskeeper-disabled/` archives for old
`specifications-core` skills.

They are not active in the generated runtime skill list, but they add audit
noise. They can be removed if there is no need to preserve the history locally.

### 5. The new `project/issues/` removal is not yet represented in overlays

Shared `git-and-github-core` correctly says bug fixes can be anchored by an
issue, but the repo-local overlay should clarify that, for Impression, an
"issue" means GitHub Issue unless explicitly stated otherwise.

### 6. Local `potentials/` modeling directives are path-light but process-heavy

The potential modeling directives do not hardcode stale `project/planning/` or
`project/issues/` paths, but they do define substantial process:

- render-and-review loops
- sub-agent visual critique
- durable reference-image handling
- model-plan documents
- modeling specs
- modeling-spec atom ledgers
- detail-floor and completeness rules

That process may be useful, but it is not yet canonical. It should remain under
`potentials/` unless deliberately promoted.

If promoted, it needs a project-lifecycle pass first:

- model plans and modeling specs should live under the active release workspace
  when they are release work
- reusable findings should go to `project/research/`
- release reference images should use
  `project/release-n.n.n/reference-images/`
- release reference STL artifacts should use
  `project/release-n.n.n/reference-stl/`
- issue follow-up should use GitHub Issues

### 7. `.agents/index.md` gives `potentials/` local authority while calling it non-canonical

The current wording is understandable, but slightly sharp-edged:

- agents should follow relevant `potentials/` directives locally
- agents must not publish or treat them as canonical

That should be preserved, but clarified as "experimental local guidance" so
future agents do not promote it accidentally or ignore it entirely.

### 8. Disabled `specifications-core` is stale relative to current manifest direction

The disabled archive:

```text
.agents/.skillskeeper-disabled/20260517-142431/specifications-core/SKILL.md
```

still describes the older recursive IWU-centered specification flow. Because it
is disabled, it does not currently control runtime behavior. But it is audit
noise and can mislead a human reading `.agents/`.

Recommended treatment:

- remove it if no local rollback value remains
- otherwise move it to a clearly named historical archive outside active
  directive discovery

## Recommended Direction

Keep project-management structure as a per-project concern.

That means:

- home `~/.codex/skills/` should contain only global/personal working rules
- parent `../.agents/skills/` should contain generic project-process concepts
- repo `.agents/` should contain the Impression-specific release workspace
  layout, artifact paths, branch conventions, and GitHub Issues choice
- generated `.agents/skills/` should be refreshed from source rather than edited
  by hand

This gives each project room to choose its own release/document lifecycle while
still sharing reusable architecture, research, planning, and Git/GitHub process
concepts.

## Target Rules

### Active Release Path Resolution

Repo-local skills should refer to the active release workspace:

```text
project/release-n.n.n/
```

For the current release:

```text
project/release-0.1.0a/
```

Release-scoped paths should be expressed as:

- `project/release-n.n.n/architecture/`
- `project/release-n.n.n/specifications/`
- `project/release-n.n.n/test-specifications/`
- `project/release-n.n.n/planning/`
- `project/release-n.n.n/coverage/`
- `project/release-n.n.n/reference-images/`
- `project/release-n.n.n/reference-stl/`
- `project/release-n.n.n/prs/`
- `project/release-n.n.n/adhoc/`

### Durable Project Paths

Repo-local skills should treat only these as durable top-level project areas:

- `project/future-features/`
- `project/meetings/`
- `project/research/`
- `project/releases/`
- `project/project-dna.md`
- `project/README.md`

### Issue Tracking

Issue tracking should use GitHub Issues.

Repo-local overlays may still allow an issue as a planning anchor, but should
phrase it as:

- a GitHub Issue for bug-fix work
- a release-scoped specification or ad hoc note when the issue creates durable
  implementation knowledge

They should not recreate `project/issues/`.

## Execution Plan

### Phase 1: Update Repo-Local Source Overlays

Edit source skills only, not generated runtime output:

- `.agents/coverage-discipline/SKILL.md`
  - change coverage artifact path from `project/coverage/` to
    `project/release-n.n.n/coverage/`
  - change final progression trigger from `project/planning/progression.md` to
    `project/release-n.n.n/planning/progression.md` or the active release's
    selected progression file
- `.agents/reference-artifact-lifecycle/SKILL.md`
  - change artifact paths to `project/release-n.n.n/reference-images/` and
    `project/release-n.n.n/reference-stl/`
  - state that completed release artifacts move with the release archive
- `.agents/git-and-github-overlay/SKILL.md`
  - replace local issue anchor language with GitHub Issue language
  - change `project/adhoc/` to `project/release-n.n.n/adhoc/`
  - change `project/planning/` planning-structure references to
    `project/release-n.n.n/planning/`
  - add the rule that active release work lives under exactly one top-level
    active release folder
- `.agents/workflow-impression-overlay/SKILL.md`
  - add the active-release-folder lifecycle
  - clarify that release-scoped planning is merged or archived as a release
    workspace, not as loose top-level project directories
- `.agents/workspace-index/SKILL.md`
  - correct the shared skill path to `../.agents/skills/`
  - add the durable top-level project areas
  - add the current active release workspace lookup rule

### Phase 1A: Update Non-Skill Local Directive Markdown

Review and update local markdown files that are not generated skills:

- `.agents/index.md`
  - keep `potentials/` non-canonical
  - clarify that `potentials/` is experimental local guidance, not published
    process
- `.agents/potentials/index.md`
  - keep the non-canonical boundary
  - add a reminder that promotion requires a release-lifecycle path review
- `.agents/potentials/modeling/README.md`
  - keep as index-only unless the modeling process is promoted
- `.agents/potentials/modeling/how-to-model.md`
  - clarify that durable model research goes to `project/research/` when it is
    reusable cross-release knowledge
  - clarify that release-specific model references go under the active release
    workspace
- `.agents/potentials/modeling/model-planning-and-specs.md`
  - clarify that model plans/specs are release-scoped work unless explicitly
    promoted to durable research or future-feature material
- `.agents/potentials/modeling/detail-and-completeness.md`
  - keep path-agnostic; no immediate lifecycle change required
- `.agents/.skillskeeper-disabled/20260517-142431/specifications-core/SKILL.md`
  - remove or move out of active local directive discovery if no rollback value
    remains

### Phase 2: Regenerate Runtime Skills

After source overlay edits, regenerate `.agents/skills/` using the existing
system-skills composer or skillskeeper workflow.

Do not hand-edit `.agents/skills/*` except as a last-resort emergency patch,
because the composer metadata marks that tree as generated.

Verification:

- `.agents/skills/coverage-discipline/SKILL.md` matches source changes
- `.agents/skills/reference-artifact-lifecycle/SKILL.md` matches source changes
- `.agents/skills/git-and-github-overlay/SKILL.md` matches source changes
- `.agents/skills/workflow-impression-overlay/SKILL.md` matches source changes
- `.agents/skills/workspace-index/SKILL.md` matches source changes

### Phase 3: Keep Parent Shared Skills Generic

Do not encode `project/release-0.1.0a/` or Impression's exact folder layout in
`../.agents/skills/`.

Optional small cleanup:

- make parent `git-and-github-core` say "GitHub Issue or equivalent tracker"
  instead of implying a local issue document
- keep parent `research`, `planning`, `release-definitions`, and
  `workflow-core` path-light
- keep manifest template paths as fallback conventions only

### Phase 4: Keep Home Skills Project-Management-Light

Do not put Impression release lifecycle rules into `~/.codex/skills/keld-*`.

Recommended home policy:

- home skills may define personal standards, global safety, and broad workflow
  preferences
- home skills should not require a specific `project/` folder layout
- home skills should defer per-project release paths, issue systems, and artifact
  storage to lower-level workspace overlays

Optional later simplification:

- retire or archive overlapping `keld-*` project-management skills once the
  parent shared layer is trusted
- keep only genuinely personal/global `keld-*` skills at home

### Phase 5: Remove Audit Noise

Optional cleanup after the active directives are aligned:

- remove repo `.agents/.skillskeeper-disabled/` if the disabled
  `specifications-core` archive is no longer useful
- remove parent `../.agents/.skillskeeper-disabled/` if its disabled archives
  are no longer useful
- ignore macOS metadata files and avoid spending process time on them

Do not remove `.agents/potentials/` as audit noise. It contains active
experimental directive work. Either keep it explicitly experimental or promote
specific documents after a lifecycle review.

## Acceptance Checks

The cleanup is complete when:

- no active repo-local source skill references old top-level release-scoped paths
  such as `project/planning/`, `project/specifications/`, `project/coverage/`,
  `project/reference-images/`, `project/reference-stl/`, `project/prs/`, or
  `project/adhoc/`
- no active directive tells agents to use `project/issues/`
- no non-skill local `.agents/**/*.md` document contradicts the active release
  workspace lifecycle
- `potentials/` documents remain explicitly experimental, or promoted documents
  have been moved out of `potentials/` after being updated for the release
  lifecycle
- generated `.agents/skills/` matches the corrected source overlays
- parent shared skills remain project-agnostic
- home `keld-*` skills contain no Impression-specific release lifecycle rules
- `project/releases/README.md` remains the canonical durable description of the
  project release workspace lifecycle

## Recommended Follow-Up

Implement Phases 1 and 2 first. They are the only required changes for this
repository to behave consistently with the new process.

Treat Phases 3 and 4 as shared-skill hygiene. Do them only if the shared/home
layers are causing real conflicts, because the repo-local overlay can safely
narrow generic rules without requiring global churn.
