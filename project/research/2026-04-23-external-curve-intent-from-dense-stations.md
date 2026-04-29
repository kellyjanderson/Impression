# External Research — Curve Intent From Dense Stations

## Topic

External research relevant to inferring smooth or higher-order curve intent from
dense faceted section data.

## Findings

### Parameterization and knot placement are the external literature's closest analogue to "curve intent inference"

The most relevant external literature does not literally ask:

- "when does a dense station stack really mean a curve?"

Instead, it asks nearby questions:

- how should parametric values be assigned?
- how should knot number and knot placement be chosen?
- how can a dense sample set be approximated compactly within tolerance?

That is still highly useful for Impression, because it reframes curve-intent
inference as a structured approximation problem rather than as a vague visual
guess.

### Dense sample reduction is usually treated as a tolerance-bound fitting problem

The B-spline approximation literature repeatedly treats dense data as a fitting
problem under tolerance, not as a raw decimation problem. In particular:

- parametrization quality changes the quality of the resulting fit
- knot placement determines where local flexibility is retained
- iterative fitting algorithms can converge toward compact approximations

This strongly supports a future Impression approach where:

- station frequency over distance is one signal
- but the real decision is made through fit quality and error metrics

### Cross-section reconstruction literature also treats sparse/dense section sets as representational choices

Two cross-section reconstruction papers are relevant:

- *Robust Surface Reconstruction from Orthogonal Slices*
- *Curvy: A Parametric Cross-section based Surface Reconstruction*

These papers differ in method, but they reinforce a useful point for Impression:

- section density and section arrangement are representational choices, not
  merely raw data quantity

The *Curvy* paper is especially relevant because it uses a compact parametric
polyline representation with adaptive splitting. That is close in spirit to the
future Impression question of whether a dense stack should be re-expressed as a
smaller, more meaningful structure.

### External literature does not appear to offer a turnkey "station frequency implies curve" rule

This is an important negative result.

The external sources support:

- fitting
- knot/parameter inference
- compact reconstruction from sections

But they do not provide a simple canonical rule like:

- "if station frequency rises here, that is definitely a curve."

So Impression will likely need its own project-specific inference contract that
combines:

- station density
- descriptor continuity
- correspondence stability
- tolerance-bounded fit quality

rather than expecting one external algorithm to solve that semantic inference
out of the box.

## Implications

External research suggests the right future design is:

- treat dense-station curve intent inference as a tolerance-bound fitting and
  representation problem
- use station frequency as supporting evidence, not sole evidence
- evaluate candidate simplified curve explanations against loop / section
  descriptors and fit residuals

This is a strong match for the current Impression direction of keeping the
problem topology-aware and planner-aware.

## References

- Pascal Laube, Matthias O. Franz, Georg Umlauf, *Deep Learning
  Parametrization for B-Spline Curve Approximation* (2018)
  - https://arxiv.org/abs/1807.08304
- Jiaqi Luo, Zepeng Wen, Hongmei Kang, Zhouwang Yang, *A Deep Neural Network
  for Knot Placement in B-spline Approximation* (2022 / CAD 2024)
  - https://arxiv.org/abs/2205.02978
- Nian-Ci Wu, Chengzhi Liu, *Randomized progressive iterative approximation for
  B-spline curve and surface fittings* (2022 / AMC 2024)
  - https://arxiv.org/abs/2212.06398
- Radek Svitak, Vaclav Skala, *Robust Surface Reconstruction from Orthogonal
  Slices* (2004 / arXiv rehost 2023)
  - https://arxiv.org/abs/2301.01713
- Aradhya N. Mathur, Apoorv Khattar, Ojaswa Sharma, *Curvy: A Parametric
  Cross-section based Surface Reconstruction* (2024)
  - https://arxiv.org/abs/2409.00829
