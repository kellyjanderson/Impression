# External Research — Spanwise Loft Consolidation

## Topic

External research relevant to spanwise loft consolidation, especially exact or
approximate simplification of over-segmented surfaces and the question of where
topology guarantees should live.

## Findings

### Mesh simplification literature separates cost, placement, and stop conditions

CGAL's surface mesh simplification documentation is useful as a mature,
practical reference point. Its edge-collapse pipeline is built around explicit:

- cost functions
- placement functions
- stop criteria
- topological constraints

That maps well onto the future Impression question:

- what is the consolidation cost
- what is the replacement span / patch placement
- when should simplification stop
- which topological constraints are non-negotiable

Even though CGAL works on triangle meshes rather than surfaced loft spans, its
architecture is a strong reminder that simplification needs explicit optimization
contracts, not only heuristic "merge if it looks okay" behavior.

### Proxy-based approximation is a useful analogy for larger-span grouping

CGAL's surface mesh approximation package is also relevant because it
approximates a mesh via clusters and proxies. The important lesson for
Impression is not the exact planar-proxy algorithm. The useful lesson is that a
good simplification pipeline often has two distinct parts:

- partition the input into meaningful groups
- fit or assign a simpler representation to each group

That aligns closely with the future spanwise consolidation problem:

- detect which consecutive loft spans belong together
- then decide how to represent the grouped run

### Topological guarantees are stronger when encoded as invariants, not after-the-fact repairs

*Topology-First B-Rep Meshing* is particularly relevant to the inline vs
postprocess question. Its core argument is that topology should be treated as an
invariant of meshing, not something patched later with heuristic repairs.

That strongly supports a long-term architectural preference inside Impression:

- exact spanwise consolidation is most trustworthy when the planner knows about
  it

At the same time, the paper does not invalidate a postprocess phase. Instead, it
clarifies what the postprocess phase cannot honestly claim:

- it may be useful
- but it does not have the same strength as a topology-aware native plan

### External CAD loft tooling still exposes span decisions structurally

Open CASCADE's loft/sweep APIs again provide a practical lesson here. Their
path/section tools expose:

- cross-section simulation
- transition modes
- continuity / approximation controls
- compatibility checks

They do not pretend that larger-span interpretation is free or automatic. When
they do approximate, they expose that approximation.

That is the right honesty posture for Impression too.

## Implications

External research reinforces the following strategy:

- first exact postprocess optimizer:
  - explicit cost / error / stop model
  - topology-preserving eligibility
- later inline planner promotion of the exact cases that prove reliable
- keep approximation, seam relocation, and proxy choice explicit

It also supports the repo-grounded conclusion that the first exact eligibility
lane should be narrow:

- stable topology
- stable correspondence
- no structural events

because that is where topological invariants are easiest to preserve honestly.

## References

- CGAL Surface Mesh Simplification User Manual
  - https://doc.cgal.org/latest/Surface_mesh_simplification/
- CGAL Surface Mesh Approximation User Manual
  - https://doc.cgal.org/latest/Surface_mesh_approximation/index.html
- YunFan Zhou et al., *Topology-First B-Rep Meshing* (2026)
  - https://arxiv.org/abs/2604.02141
- Open CASCADE `GeomFill_Pipe`
  - https://dev.opencascade.org/doc/refman/html/class_geom_fill___pipe.html
- Open CASCADE `BRepOffsetAPI_MakePipeShell`
  - https://dev.opencascade.org/doc/refman/html/class_b_rep_offset_a_p_i___make_pipe_shell.html
