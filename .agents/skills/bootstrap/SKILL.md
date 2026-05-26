---
name: bootstrap
description: Project bootstrap and orientation workflow. Use when the user asks Codex to bootstrap, onboard to, review, or understand an entire project/repository/workspace before beginning substantive work, especially at the start of a new project session or when project state is unclear.
---

# Bootstrap

Use this skill to build a working understanding of the current project state before starting implementation or analysis.

## Workflow

1. Read the active instruction layers first.
   - Start with the user-level entry point if one is specified by the workspace, such as `/Users/k/Agents/index.md`.
   - Check for top-level `AIgents` or `Agents` folders.
   - Check for `project/agents` or `project/Agents` and treat them as more specific project-local instructions.

2. Survey the project structure.
   - Identify the root, major directories, generated artifacts, source files, config files, docs, tests, build outputs, and dependency manifests.
   - Use fast local tools such as `rg --files`, `find`, `ls`, and targeted file reads.
   - Avoid exhaustive reading of large generated or dependency directories unless they are the project itself.

3. Read project orientation material.
   - Prefer `README`, `AGENTS`, `AIgents`, `docs`, `project`, planning notes, changelogs, manifests, and task files.
   - For code projects, inspect package/build/test configuration and the main entry points.
   - For design/modeling projects, inspect source models, generated outputs, build scripts, and build documentation.

4. Form a current-state summary.
   - Capture what the project appears to be for.
   - Identify active files, likely workflows, build/test commands, generated outputs, and important constraints.
   - Distinguish observed facts from inferences.

5. Look for major holes or ambiguities.
   - Ask questions only for issues that materially block safe next work.
   - Treat missing core requirements, unknown target platform, unclear ownership of major files, contradictory instructions, or absent critical inputs as major.
   - Do not stop for minor naming, style, cleanup, or optimization questions. Assume those can be resolved during upcoming work.

6. Report orientation results concisely.
   - If major blockers exist, ask the smallest set of concrete questions needed to proceed.
   - If no major blockers exist, say so and proceed with the requested work or state readiness.
   - Mention non-major uncertainties only as assumptions or notes, not as blocking questions.

## Question Threshold

Ask only when the answer changes the immediate direction of the work or prevents likely wasted effort. Otherwise, make a reasonable assumption, name it briefly, and continue.
