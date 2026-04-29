# External Research — Model-Assisted Mesh Repair

## Topic

External research relevant to mesh repair and surface reconstruction workflows
that could inform Impression's future model-assisted repair direction.

## Findings

### Reconstruction pipelines repeatedly separate structure recovery from surface generation

Open3D's reconstruction documentation is useful here because it exposes several
families of surface reconstruction with clearly stated assumptions:

- alpha shapes
- ball pivoting
- Poisson reconstruction

It also makes an important distinction:

- some methods produce non-smooth results tied closely to the input points
- Poisson solves a regularized problem and can produce smoother surfaces

That reinforces the basic architectural lesson for Impression:

- raw mesh repair and final surfaced reconstruction are different stages

### Slice-based reconstruction is a real external problem, not just an internal project idea

*Robust Surface Reconstruction from Orthogonal Slices* is directly relevant to
the future idea of sectioning damaged meshes and reconstructing spans from
cross-sections. Its core claim is that the final reconstruction depends on
correct estimation of the original structure and that orthogonal slice sets
improve the problem.

For Impression, the exact algorithm is less important than the lesson:

- section sets are a legitimate reconstruction basis
- structure determination is the hard part

That fits very well with the current retained tool
`section_mesh_with_plane(...)`.

### Recent cross-section reconstruction work emphasizes compact parametric representations

*Curvy* also matters here because it uses a compact parametric polyline
representation and reconstructs from sparse cross-sections.

That supports a future Impression repair pipeline where section extraction does
not stop at raw polylines. Instead, extracted sections may need:

- cleanup
- canonicalization
- compact representation

before surfaced reconstruction is trustworthy.

### B-Rep reconstruction literature reinforces the value of explicit structural validity

*ComplexGen* and *Topology-First B-Rep Meshing* both reinforce a similar theme:

- geometry is not enough
- valid structural relationships matter

For Impression, this strongly supports a repair workflow that does not merely
patch triangles locally. Instead, it should aim to recover:

- section structure
- adjacency
- seam intent
- or broader surfaced validity

before declaring the repair complete.

### Hole-filling literature shows the practical importance of border-ring extraction and bounded remeshing

The LoD2 hole-filling paper is particularly practical. Its stages:

1. preprocess topological errors
2. detect and extract complete border rings
3. remesh to reconstruct complete geometry

line up closely with the likely future Impression repair sequence:

1. analyze / clean foreign mesh locally
2. extract meaningful boundaries or slices
3. reconstruct the missing region
4. verify the result

That is strong confirmation that the future repair idea is well formed.

## Implications

External research supports a future Impression repair architecture that:

- uses retained mesh tools for diagnosis and section extraction
- converts extracted evidence into more canonical surfaced reconstruction inputs
- treats structure recovery as a first-class step
- treats watertightness and topological consistency as acceptance gates, not
  optional polish

It also suggests that a strong future repair stack will need both:

- local slice / boundary extraction
- structural reconstruction logic

not just one of them.

## References

- Open3D Surface Reconstruction documentation
  - https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
- Radek Svitak, Vaclav Skala, *Robust Surface Reconstruction from Orthogonal
  Slices* (2004 / arXiv rehost 2023)
  - https://arxiv.org/abs/2301.01713
- Aradhya N. Mathur, Apoorv Khattar, Ojaswa Sharma, *Curvy: A Parametric
  Cross-section based Surface Reconstruction* (2024)
  - https://arxiv.org/abs/2409.00829
- Haoxiang Guo et al., *ComplexGen: CAD Reconstruction by B-Rep Chain Complex
  Generation* (2022)
  - https://arxiv.org/abs/2205.14573
- YunFan Zhou et al., *Topology-First B-Rep Meshing* (2026)
  - https://arxiv.org/abs/2604.02141
- Weixiao Gao, Ravi Peters, Hugo Ledoux, Jantien Stoter, *Filling holes in LoD2
  building models* (2024)
  - https://arxiv.org/abs/2404.15892
