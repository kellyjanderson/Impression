# Mesh Reference And Cruft Cleanup Checklist

Date: 2026-07-11

Purpose: remove stale language that teaches agents or contributors that Impression has, should have, or should preserve a mesh-first or parallel mesh modeling path. Mesh may remain only as an explicit boundary artifact for preview, STL/export, downstream mesh-only consumers, foreign mesh analysis/repair, or explicitly named compatibility utilities.

## Mesh Reference Removal

- [ ] Getting Started tutorial mesh-lane language
  - Path: `/Users/k/Documents/Projects/Impression/docs/tutorials/getting-started.md`
  - Change: rewrite the overview and key ideas so `build()` returns Impression modeling objects, with `SurfaceBody` as canonical authored geometry. Remove "internal meshes" and "Public CSG is still executable on the mesh lane today." Replace CSG guidance with surface-body CSG posture and explicit tessellation/export boundaries.

- [ ] Generated Getting Started tutorial mesh-lane language
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/tutorials/getting-started.md`
  - Change: bring this copy into parity with the corrected `docs/tutorials/getting-started.md`, or remove/regenerate the stale generated docs tree if `impression-docs/` is not meant to be authoritative.

- [ ] CLI build return contract
  - Path: `/Users/k/Documents/Projects/Impression/docs/cli.md`
  - Change: replace "`build()` must return internal meshes" with a surface-first contract: `build()` returns Impression modeling objects, including `SurfaceBody`; mesh is accepted only as an explicit compatibility or consumer-boundary artifact. Update workflow wording at the bottom of the file.

- [ ] Generated CLI build return contract
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/cli.md`
  - Change: mirror the corrected `docs/cli.md` language or remove/regenerate the generated copy.

- [ ] CSG documentation mesh-lane framing
  - Path: `/Users/k/Documents/Projects/Impression/docs/modeling/csg.md`
  - Change: remove "mesh-lane" and "mesh-primary" framing from the main CSG page. Present public booleans as surface-body result APIs. Move `union_meshes(...)`, `make_*_mesh(...)`, and mesh examples into an explicit compatibility/tooling subsection that cannot be read as canonical modeling.

- [ ] Generated CSG documentation stale mesh-primary framing
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/modeling/csg.md`
  - Change: remove statements that "all helpers operate on internal triangle meshes," "public boolean execution helpers remain mesh-primary," "default mesh lane returns executable mesh geometry," and "keep using the default mesh lane." Sync with the corrected `docs/modeling/csg.md` or regenerate.

- [ ] Primitive documentation stale triangle-mesh claim
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/modeling/primitives.md`
  - Change: replace "All primitives ... return internal triangle meshes" with the surface-first default contract from `docs/modeling/primitives.md`. Include explicit `make_*_mesh(...)` compatibility helpers only as named mesh-boundary routes.

- [ ] Root docs product description
  - Path: `/Users/k/Documents/Projects/Impression/docs/index.md`
  - Change: replace "toolkit for building watertight meshes" with surface-body modeling language. Mention watertight mesh only as a tessellated export/verification product.

- [ ] Generated root docs product description
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/index.md`
  - Change: mirror the corrected `docs/index.md` language or regenerate the generated copy.

- [ ] Group documentation treats `MeshGroup` as normal authored grouping
  - Path: `/Users/k/Documents/Projects/Impression/docs/modeling/groups.md`
  - Change: rewrite around surfaced composition or mark this page as explicit mesh compatibility only. Remove advice that a normal model can return `MeshGroup` and then pass `grp.to_mesh()` into CSG helpers as a standard path.

- [ ] Generated group documentation treats `MeshGroup` as normal authored grouping
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/modeling/groups.md`
  - Change: mirror the corrected `docs/modeling/groups.md` language or remove/regenerate the generated copy.

- [ ] Transform documentation says transforms reshape meshes
  - Path: `/Users/k/Documents/Projects/Impression/docs/modeling/transforms.md`
  - Change: rewrite transform guidance around surface objects and surfaced composition first. Keep `Mesh`/`MeshGroup` transform behavior only under explicit mesh compatibility or standalone mesh utility sections.

- [ ] Generated transform documentation says transforms reshape meshes
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/modeling/transforms.md`
  - Change: mirror the corrected `docs/modeling/transforms.md` language or remove/regenerate the generated copy.

- [ ] CSG examples are mesh-first by default
  - Path: `/Users/k/Documents/Projects/Impression/docs/examples/csg/`
  - Change: split examples into surface-body CSG examples and explicitly named mesh-tool examples. Any example that imports `make_*_mesh(...)`, `MeshGroup`, or `union_meshes(...)` should live under an explicit compatibility/tooling name and not be the primary CSG tutorial path.

- [ ] Generated CSG examples are mesh-first by default
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/examples/csg/`
  - Change: sync with the corrected `docs/examples/csg/` layout or regenerate. In particular, avoid examples importing `make_box`, `make_cylinder`, and `union_meshes` together in a way that implies public primitives are mesh constructors.

- [ ] Public CSG parameter naming still says `meshes`
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/csg.py`
  - Change: rename public boolean parameters and docs from `meshes` to `operands` or `bodies` where the API accepts `SurfaceBody`. Keep `meshes` only for explicitly named mesh utilities such as `union_meshes(...)`.

- [ ] Public CSG exports expose mesh utilities beside canonical APIs
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/__init__.py`
  - Change: review exports for `union_meshes`, `make_*_mesh(...)`, `MeshGroup`, and loft debug mesh helpers. Keep if needed for compatibility, but consider moving docs and examples toward a compatibility namespace or adding stronger deprecation/compatibility naming.

- [ ] `MeshGroup` public grouping surface
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/group.py`
  - Change: ensure `MeshGroup` remains explicitly classified as mesh compatibility only. Add or verify docs/tests that authored surface modules must not use it as composition truth.

- [ ] Mesh executor loft spec title and placement
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/loft-60-mesh-executor-correspondence-consumption-v1_0.md`
  - Change: archive, move, or rename this retired spec so agents do not see "mesh executor" as an active implementation target. Keep the content only as historical/debug/tessellation-boundary guidance.

- [ ] Interactive loft spec still accepts "watertight mesh" as execution result
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/loft-20-interactive-branch-picking-v1_0.md`
  - Change: rewrite acceptance criteria so valid manual selection yields canonical `SurfaceBody`; mesh watertightness may be a downstream tessellation/export verification only.

- [ ] Primitive API migration spec says default remains mesh
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/surface-44-primitive-api-surface-return-migration-v1_0.md`
  - Change: supersede or revise the compatibility-phase wording that says the default remains `backend="mesh"`. Mark historical if the implementation has moved to surface defaults.

- [ ] Modeling op migration spec says default remains mesh
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/surface-45-modeling-op-surface-return-migration-v1_0.md`
  - Change: supersede or revise any compatibility-phase wording that preserves mesh defaults. Align with surface-body-only authored modeling posture.

- [ ] Mesh-first decommission spec title keeps stale concept active
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/surface-19-surface-promotion-and-mesh-decommission-v1_0.md`
  - Change: mark as superseded by current surface-body-only policy or archive under historical migration material. Avoid presenting "mesh-first internals" as an active design track.

- [ ] Mesh-first rollback spec title keeps stale concept active
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/surface-57-mesh-first-decommission-rollback-v1_0.md`
  - Change: mark as superseded or historical. If retained, make clear this is not permission for new mesh paths.

- [ ] Mesh execution inventory report is stale as active guidance
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/surface-159-mesh-execution-inventory-report-v1_0.md`
  - Change: update classifications to current implementation state or move to historical audit. Remove stale public-risk claims that say current defaults produce mesh where they no longer do.

- [ ] Architecture tracker already notes older specs need audit
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/architecture/architecture-work-tracker.md`
  - Change: expand the existing mesh-primary audit item into concrete links to this cleanup checklist, or replace it once this checklist becomes the active source of truth.

- [ ] Agent feature map says surface CSG is bounded and narrower than mesh lane
  - Path: `/Users/k/Documents/Projects/Impression/docs/skills/impression/references/feature-map.md`
  - Change: remove "mesh lane" phrasing. Describe explicit mesh tools separately from canonical surface-body modeling.

- [ ] Generated agent feature map says surface CSG is bounded and narrower than mesh lane
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/skills/impression/references/feature-map.md`
  - Change: mirror the corrected `docs/skills/impression/references/feature-map.md` or regenerate.

## Cruft Cleanup

- [ ] Stale generated documentation tree
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/`
  - Change: decide whether this tree is generated output, packaged docs source, or active docs. If generated, remove it from source-of-truth review paths and regenerate from `docs/`. If active, sync every stale surface/mesh page with `docs/`.

- [ ] Accidental `.agents` tree inside reference STL artifacts
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/reference-stl/.agents/`
  - Change: remove from reference artifacts unless there is a documented reason for skills to live inside `reference-stl`. This directory can mislead agents and pollute artifact scans.

- [ ] macOS `.DS_Store` files
  - Path: `/Users/k/Documents/Projects/Impression/`
  - Change: delete tracked/untracked `.DS_Store` files across the repo and ensure `.gitignore` excludes them. Current scan found `.DS_Store` under root, `.agents`, `.cache`, `dist`, `docs`, `ide`, `impression-docs`, `project`, `scripts`, `src`, and `tests`.

- [ ] Preview control file
  - Path: `/Users/k/Documents/Projects/Impression/.impression-preview`
  - Change: remove local preview control state from the repo workspace and ensure it is ignored.

- [ ] Egg-info build metadata in source tree
  - Path: `/Users/k/Documents/Projects/Impression/src/impression.egg-info/`
  - Change: remove generated packaging metadata from source control/workspace unless intentionally versioned. Ensure build metadata is ignored or generated only during packaging.

- [ ] VSIX binary artifact
  - Path: `/Users/k/Documents/Projects/Impression/ide/vscode-extension/impression-vscode-0.1.0.vsix`
  - Change: decide whether the packaged VSIX belongs in source. If not, remove it and document package generation in the extension README or release process.

- [ ] Active dirty worktree should not be mixed with cleanup
  - Path: `/Users/k/Documents/Projects/Impression/`
  - Change: before cleanup edits, isolate current modified files and untracked CSG STL outputs. Keep mesh-language cleanup in a separate change set from active CSG implementation/reference artifact work.

- [ ] Untracked CSG dirty STL outputs
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/reference-stl/dirty/surfacebody/csg/`
  - Change: decide whether the untracked files are intended reference artifacts. Promote, track, or delete them through the reference artifact lifecycle rather than leaving loose untracked outputs.

- [ ] Duplicate skill mirrors
  - Path: `/Users/k/Documents/Projects/Impression/.agents/skills/`
  - Change: verify whether local skills are generated mirrors or durable source. If generated, document the update path and avoid manual edits that drift from the actual skill source.

- [ ] Historical research that still uses mesh-primary framing
  - Path: `/Users/k/Documents/Projects/Impression/project/research/`
  - Change: leave historical research intact when useful, but add a short historical-status note or index warning so old "mesh lane" research is not mistaken for current architecture.

- [ ] Mesh tools docs placement
  - Path: `/Users/k/Documents/Projects/Impression/docs/modeling/mesh-tools.md`
  - Change: keep this page only if it is clearly labeled as foreign mesh analysis/repair/debug tooling. Cross-link from CSG/primitive docs only as an explicit compatibility boundary, not as a modeling path.

- [ ] Generated mesh tools docs placement
  - Path: `/Users/k/Documents/Projects/Impression/impression-docs/modeling/mesh-tools.md`
  - Change: mirror the corrected `docs/modeling/mesh-tools.md` or regenerate.

- [ ] Release specification README indexes stale mesh-first specs as active leaves
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/specifications/README.md`
  - Change: mark superseded/historical specs in the index or move them under a retired section so agents do not select them as live implementation instructions.

- [ ] Reference STL artifact directory contains non-STL support files
  - Path: `/Users/k/Documents/Projects/Impression/project/release-0.1.0a/reference-stl/`
  - Change: after removing `.agents` and `.DS_Store`, verify the directory contains only expected reference artifacts and documented metadata.
