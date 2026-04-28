# External Research — Trajectory-Guided Loft

## Topic

External research relevant to explicit trajectory guidance for loft-like surface
construction.

## Findings

### Industrial CAD tooling already distinguishes between section-only and guided path/sweep construction

Open CASCADE's geometry stack exposes several flavors of path-driven surface
construction:

- constant section along a path
- evolving section from first to last section
- multiple sections along a path
- guide-curve assisted pipes
- two-guide-line variants

That matters for Impression because it shows a mature external precedent for the
idea that vertical evolution can be specified with more than just isolated
cross-sections.

In other words, the external ecosystem already recognizes a ladder of control:

- plain sections
- path plus sections
- path plus guide information

This supports the future direction of adding explicit trajectory guidance rather
than forcing all vertical intent to hide inside station density.

### External tools usually start with global path guidance before richer local guidance

The Open CASCADE APIs also suggest a practical staging principle:

- first whole-path / whole-loft guidance
- then guide-curve or richer sweep variants

That aligns with the repo-grounded conclusion that whole-loft shared trajectory
guidance is the right first milestone before region- or track-level trajectories.

### Guided-surface literature supports structured interpolation over curve networks

Two external papers are particularly relevant:

- *Curve network interpolation by C1 quadratic B-spline surfaces*
- *Interpolation of a spline developable surface between a curve and two rulings*

The details differ from Impression's future idea, but both reinforce the same
high-level lesson:

- surfaces can be constructed from richer curve constraints than just a serial
  list of sections

This supports the idea that trajectory-guided loft does not need to be seen as
an odd special case. It sits inside a broader family of curve-constrained
surface-construction methods.

### Explicit trajectory guidance should remain a control layer, not a replacement for structural planning

The external sources do not suggest a reason to collapse structural planning into
trajectory fitting. Even in guided path/sweep tooling, there is still a strong
separation between:

- path / guide input
- section input
- build / approximation settings

That supports keeping trajectory guidance in Impression as:

- a path-aware influence on how sections travel

not:

- a replacement for topology interpretation and ambiguity handling

## Implications

External research strongly supports:

- starting with shared whole-loft trajectory guidance
- treating path / guide data as a first-class influence on in-between evolution
- keeping stations as structural anchors
- adding richer region- or track-level attachment only after the basic
  path-guided contract proves useful

It also suggests that external CAD precedent favors explicitness:

- if guidance changes the build, the API should say so
- if approximation is involved, the API should expose it

## References

- Open CASCADE `GeomFill_Pipe`
  - https://dev.opencascade.org/doc/refman/html/class_geom_fill___pipe.html
- Open CASCADE `BRepOffsetAPI_MakePipeShell`
  - https://dev.opencascade.org/doc/refman/html/class_b_rep_offset_a_p_i___make_pipe_shell.html
- Catterina Dagnino, Paola Lamberti, Sara Remogna, *Curve network
  interpolation by C1 quadratic B-spline surfaces* (2013)
  - https://arxiv.org/abs/1312.5533
- A. Cantón, L. Fernández-Jambrina, *Interpolation of a spline developable
  surface between a curve and two rulings* (2015)
  - https://arxiv.org/abs/1503.06995
