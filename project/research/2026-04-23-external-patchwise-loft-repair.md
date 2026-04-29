# External Research — Patchwise Loft Repair

## Topic

External research relevant to local patchwise reconstruction / repair rather
than whole-band or whole-body reconstruction.

## Findings

### External literature is stronger on local boundary extraction and remeshing than on explicit patchwise loft repair

This is an important honest result.

The sources reviewed do support:

- extracting meaningful hole boundaries
- reconstructing from slices
- preserving structure during reconstruction

But there is much less directly matching literature on a feature exactly like
"patchwise loft repair" as phrased in the future-feature note.

That means this future branch will likely need to be more Impression-specific
than some of the others.

### Border rings and local boundary completeness are recurring prerequisites

The LoD2 hole-filling paper is again useful here. Its insistence on detecting
and extracting complete border rings before reconstruction is directly relevant
to patchwise repair.

For Impression, the lesson is:

- a local patch repair must start from explicit local boundary truth

not from a vague local point neighborhood.

### Structural models that represent patches and adjacencies explicitly are valuable precedents

*ComplexGen* is not a repair paper, but it is still highly relevant because it
models:

- vertices
- edges
- surface patches
- their relationships

as a chain complex.

That is a strong external precedent for thinking about patchwise repair as a
structural reconstruction problem, not merely a geometric interpolation problem.

### Topology-preserving frameworks strengthen the case for integrating patchwise repair into the surfaced shell model

The topology-first meshing paper also matters here. Its broader lesson is that
topology and adjacency should be encoded as invariants if possible.

That supports the repo-grounded conclusion that patchwise repair should target
the existing `SurfaceBody` / `SurfaceShell` / seam model rather than inventing a
parallel patch-graph abstraction that later needs ad hoc translation.

### Slice-based reconstruction remains a useful local evidence source

The orthogonal-slice reconstruction paper remains relevant here too, because it
shows that local section evidence can meaningfully constrain reconstruction.

For patchwise loft repair, this suggests a plausible hybrid:

- local boundaries define the missing patch neighborhood
- local slices or probes supply additional shape evidence
- the repaired patch set is then integrated back into the surfaced shell

## Implications

External research suggests that a future patchwise repair system in Impression
should:

- recover explicit local boundaries first
- use the surfaced seam/adjacency model as the integration target
- allow additional section evidence to constrain the repair
- be framed as one mode of a broader surface reconstruction system, not as an
  isolated one-off primitive

The external literature also leaves a clear gap:

- there is strong support for local boundary extraction and structural
  reconstruction
- there is less direct precedent for exactly "patchwise loft repair"

So this branch is a good candidate for project-specific experimentation once the
broader repair and surface reconstruction branches are more mature.

## References

- Weixiao Gao, Ravi Peters, Hugo Ledoux, Jantien Stoter, *Filling holes in LoD2
  building models* (2024)
  - https://arxiv.org/abs/2404.15892
- Radek Svitak, Vaclav Skala, *Robust Surface Reconstruction from Orthogonal
  Slices* (2004 / arXiv rehost 2023)
  - https://arxiv.org/abs/2301.01713
- Haoxiang Guo et al., *ComplexGen: CAD Reconstruction by B-Rep Chain Complex
  Generation* (2022)
  - https://arxiv.org/abs/2205.14573
- YunFan Zhou et al., *Topology-First B-Rep Meshing* (2026)
  - https://arxiv.org/abs/2604.02141
