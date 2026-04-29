# Future-Feature External Research Index

## Topic

External literature and tooling review for the current loft- and repair-oriented
future-feature ideas.

## Findings

This note collects the external companion research created after the initial
repo-grounded future-feature research pass.

The goal of this tranche is different from the earlier internal notes:

- earlier notes answered the questions from the point of view of the current
  Impression codebase
- this tranche checks those same questions against external papers and official
  geometry-tool documentation

## External Research Notes

- `project/research/2026-04-23-external-control-station-inference.md`
- `project/research/2026-04-23-external-spanwise-loft-consolidation.md`
- `project/research/2026-04-23-external-trajectory-guided-loft.md`
- `project/research/2026-04-23-external-curve-intent-from-dense-stations.md`
- `project/research/2026-04-23-external-model-assisted-mesh-repair.md`
- `project/research/2026-04-23-external-patchwise-loft-repair.md`

## Coverage

These notes cover the research agenda topics already listed in:

- `project/research/2026-04-23-future-feature-research-agenda.md`

The mapping is:

- control-station semantics + workflow
  -> `external-control-station-inference`
- spanwise qualification + branch strategy
  -> `external-spanwise-loft-consolidation`
- trajectory-guided representation
  -> `external-trajectory-guided-loft`
- curve-intent inference from dense stations
  -> `external-curve-intent-from-dense-stations`
- model-assisted mesh repair workflow
  -> `external-model-assisted-mesh-repair`
- patchwise loft repair structure
  -> `external-patchwise-loft-repair`

## Implications

The external material reinforces a few broad themes:

- ordered cross-sections, compatibility, and explicit path handling are common
  in industrial loft/sweep tools
- fitting and simplification problems are usually parameterization / knot /
  tolerance problems, not only raw point-reduction problems
- topology-preserving guarantees are much stronger when they are encoded in the
  algorithm rather than repaired after the fact
- repair and reconstruction literature repeatedly separates:
  - structural recovery
  - local boundary extraction
  - final geometric filling / remeshing

Those themes line up well with the future directions already emerging inside the
repo.

## References

- `project/research/2026-04-23-future-feature-research-agenda.md`
