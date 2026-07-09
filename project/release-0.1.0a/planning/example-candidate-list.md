# 0.1.0.a Example Candidate List

## Purpose

This document lists the example candidates that can best demonstrate the jump
from the released `0.0.3a2` surface-first loft stack to the current
`0.1.0.a` working branch.

The intent is not to fully spec the examples yet. The intent is to choose the
examples that most clearly prove:

- what the released version could already do
- what the current version can now do better
- why those changes matter for real modeling work

## Ranked Summary List

1. **Airframe Shell From Sparse Former Stations**
   Sparse winglet or fuselage-style sections demonstrate curve fitting,
   reduction, hidden control structure, and topology-preserving refusal.
   This is the best overall proof that `0.1.0.a` turns dense loft evidence
   into a cleaner and more explainable result than `0.0.3a2`.

2. **Tight-Shoulder Bottle Without Rail Explosion**
   A consumer-package bottle demonstrates shared trajectory inference,
   guidance attachment, and fit drift reporting in a shape almost everyone
   can judge instantly. This is the strongest hero-image candidate.

3. **Orientation-Safe Motor Mount Or Fairing Spine**
   A path-sensitive lofted fairing proves that progression is now a semantic
   object rather than just a scalar ordering. It shows transport, attachment,
   twist, and guidance behavior without splitting loft into a separate sweep
   product line.

4. **Contour Stack Simplifier For Reverse-Engineered Parts**
   A deliberately over-sampled contour stack demonstrates reduced progression
   bundles, retained topology/control classification, and replayable internal
   inference output. This is the most engineering-forward example.

5. **Loft Triage Dashboard For Uncertain Or Failed Inference**
   A diagnostic-first example shows accepted, uncertain, and refused inference
   outcomes side by side. This is the trust example and the best proof that
   the new stack is inspectable rather than magical.

## What Each Later Example Plan Must Contain

Each future detailed example plan should include:

- the modeling objective in plain language
- the specific visual or geometric moment that demonstrates the delta from
  `0.0.3a2` to `0.1.0.a`
- the exact `0.1.0.a` features being exercised
- a clear statement of what the released version would require instead
  such as denser stations, more rails, more manual cleanup, or no durable
  explanation of failure
- the real-world user need or forum pain point the example answers
- the presentation angle
  still image, parameter sweep, diagnostic printout, side-by-side comparison,
  or hero documentation example

## Detailed Candidate Notes

### 1. Airframe Shell From Sparse Former Stations

**Summary**

Build a small aircraft fuselage or winglet shell from a sparse set of section
profiles and show that `0.1.0.a` can explain a smooth loft with fitted curve
intent, hidden control structure, retained topology stations, and explicit
reduction/refusal posture. The released version can already loft the shell, but
 it cannot compact or explain the section stack this way.

**Why it matters**

- Airframe and fairing work is one of the clearest cases where users currently
  fall back to “add more sections” or “add more guides.”
- It is visually impressive because the result is elegant and obviously not a
  toy vase or cube demo.

**Feature coverage**

- curve fitting from dense loft evidence
- non-user-facing control stations
- control-station inference
- structural preservation and refusal posture
- shared diagnostics and developer inspection

**Released vs current**

- `0.0.3a2`: author many sections and manually tune the loft
- `0.1.0.a`: start dense if needed, infer a reduced progression, keep topology
  truth explicit, and report what was retained or dropped

**Real-world need links**

- [Fusion 360 Lofting Trouble](https://forums.autodesk.com/t5/fusion-support-forum/fusion-360-lofting-trouble/td-p/11614488)
- [Designing a winglet for an airplane wing, using loft and guide curves](https://forum.onshape.com/discussion/22566/designing-a-winglet-for-an-airplane-wing-using-loft-and-guide-curves)

### 2. Tight-Shoulder Bottle Without Rail Explosion

**Summary**

Build a packaging or decanter-style bottle with a tight shoulder transition and
demonstrate that `0.1.0.a` can reason about shared trajectory intent, expose
fit drift, and attach explicit guidance records where needed instead of forcing
the user into a brittle proliferation of rails and intermediate sections.

**Why it matters**

- Bottle and package geometry is easy to recognize and easy to judge.
- Users immediately understand when a shoulder looks wrong, pinched, or lumpy.

**Feature coverage**

- shared trajectory candidate fitting
- whole-loft shared trajectory assessment
- explicit shared guidance attachment
- planner consumption boundaries for guidance
- fit residual and refusal reporting

**Released vs current**

- `0.0.3a2`: loft the shape, but rely on manual station density and rail tuning
- `0.1.0.a`: keep a surfaced loft workflow while adding durable trajectory and
  guidance semantics plus explicit uncertainty/refusal posture

**Real-world need links**

- [How to achieve this loft on this bottle?](https://forums.autodesk.com/t5/fusion-design-validate-document/how-to-achieve-this-loft-on-this-bottle/td-p/9609170)
- [Tight curve lofting improvment](https://forums.autodesk.com/t5/fusion-360-ideastation-archived/tight-curve-lofting-improvment/idi-p/8647961)
- [Loft issue with Bottle design](https://forums.autodesk.com/t5/autocad-forum/loft-issue-with-bottle-design/td-p/8127558)

### 3. Orientation-Safe Motor Mount Or Fairing Spine

**Summary**

Build a lofted part whose sections must stay meaningfully oriented along travel,
such as a motor mount fairing, intake fairing, or aerodynamic transition. Use
it to demonstrate the difference between old scalar progression and the new
path-backed progression, transport semantics, twist/scale slots, and explicit
guidance consumption boundaries.

**Why it matters**

- This is the cleanest example for proving that progression is no longer “just
  numbers.”
- It also lets us show why path-driven behavior is being absorbed into loft
  rather than split into a separate sweep tool line.

**Feature coverage**

- path-backed progression
- station attachment to progression
- transport semantics
- twist and scale semantic slots
- explicit shared guidance planner consumption

**Released vs current**

- `0.0.3a2`: progression mainly orders stations
- `0.1.0.a`: progression becomes a semantic carrier for path, provenance,
  attachment, transport, and later trajectory/guidance interpretation

**Real-world need links**

- [Keep profile orientation during a loft](https://forum.onshape.com/discussion/28189/keep-profile-orientation-during-a-loft)
- [Guide curve or more profiles for lofting.](https://forum.onshape.com/discussion/8422/guide-curve-or-more-profiles-for-lofting)
- [Loft Not Following Guide Curves](https://forums.autodesk.com/t5/fusion-design-validate-document/loft-not-following-guide-curves/td-p/9370557)

### 4. Contour Stack Simplifier For Reverse-Engineered Parts

**Summary**

Start from an intentionally over-sampled stack of sections that represents
reverse-engineered, scanned, or otherwise contour-derived shape evidence. Show
that `0.1.0.a` can reduce the stack into a replayable reduced progression
bundle with retained station classification, while also refusing reduction when
topology-critical structure would be lost.

**Why it matters**

- This is the purest demonstration of control-station inference rather than
  merely “better loft behavior.”
- It gives us a strong technical example for engineers and CAD power users who
  think in terms of imported evidence and cleanup.

**Feature coverage**

- reduced progression bundle
- retained station classification
- hidden control-station provenance
- structural preservation assessment
- refusal as a first-class output

**Released vs current**

- `0.0.3a2`: keep the dense stack or manually simplify it without durable
  machine-readable provenance
- `0.1.0.a`: produce a reduced internal representation with replayable output,
  retained topology/control distinction, and explicit refusal triggers

**Real-world need links**

- [Self intersecting loft when using offset contour](https://forums.autodesk.com/t5/fusion-design-validate-document/self-intersecting-loft-when-using-offset-contour/td-p/10766237)
- [Creased edges after lofting](https://forums.autodesk.com/t5/fusion-design-validate-document/creased-edges-after-lofting/td-p/9349250)
- [Loft generates surface but not solid](https://forum.onshape.com/discussion/25049/loft-generates-surface-but-not-solid)

### 5. Loft Triage Dashboard For Uncertain Or Failed Inference

**Summary**

Create a diagnostic-first example that runs several loft cases through the new
inference stack and shows the shared diagnostic bundle, developer inspection
record, and downstream report side by side. This should be the “trust” example:
the user sees not only a good result, but also why a reduction or trajectory
was accepted, marked uncertain, or refused.

**Why it matters**

- This is the most important example for credibility with advanced users.
- It proves that the new system is not just adding opaque heuristics.

**Feature coverage**

- shared inference diagnostic bundle schema
- bundle population and reuse across branches
- developer-facing explainability
- downstream uncertainty/refusal reporting
- exact-vs-inferred provenance communication

**Released vs current**

- `0.0.3a2`: surfaced loft output and planner artifacts exist, but the new
  inference stack and its reporting layers do not
- `0.1.0.a`: accepted, uncertain, and refused outcomes become inspectable and
  communicable instead of implicit

**Real-world need links**

- [Loft fails when guide curve is too severe](https://forum.onshape.com/discussion/23829/loft-fails-when-guide-curve-is-too-severe)
- [Loft did not generate properly: Current selections would create a self-intersecting body](https://forum.onshape.com/discussion/29436/loft-did-not-generate-properly-current-selections-would-create-a-self-intersecting-body)
- [Loft failure when I use a third guide](https://forum.onshape.com/discussion/27044/loft-failure-when-i-use-a-third-guide)

## Shortlist Recommendation

If we only build three examples first, the strongest opening set is:

1. **Airframe Shell From Sparse Former Stations**
   Best overall proof of curve fitting, reduction, and retained topology truth.
2. **Tight-Shoulder Bottle Without Rail Explosion**
   Best visual consumer-product demo and easiest hero image.
3. **Loft Triage Dashboard For Uncertain Or Failed Inference**
   Best proof that the new system is explainable rather than magical.

## Impression Test

The examples most likely to impress difficult users are the ones that show:

- a shape they already know is hard in other CAD tools
- fewer authored sections or rails
- a visibly cleaner outcome
- a readable explanation when the system refuses to guess

That is why the strongest mix is not “five pretty models.” It is:

- one hero airframe or shell
- one hero bottle or package
- one trust-building diagnostic example
