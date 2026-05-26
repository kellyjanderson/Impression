# Model Planning And Specs

## Purpose

This directive defines how model-planning documents and modeling specs should
work before substantial model execution begins.

The goal is to prevent ambitious modeling tasks from collapsing into a single
blob of under-specified work.

## Core Rule

A model-plan document should behave more like an architecture document than
like a casual idea note.

It should define:

- what object or object family is being modeled
- what the model is supposed to demonstrate
- what the visible success criteria are
- what major parts or sub-bodies exist
- what operations are likely required
- what level of detail the model is expected to reach
- what counts as "enough" planning before execution starts

## Model Plan Role

The model-plan document is the high-level design artifact for the model.

It should describe:

- objective
- release or feature delta being demonstrated
- visual target
- object decomposition
- probable modeling primitives and operations
- review strategy
- expected level of detail
- modeling risks

The model plan should not pretend to be executable if it still hides major
shape decomposition decisions.

## Modeling Specs

After the model plan, create one or more modeling specs.

Modeling specs should break work into small digestible units of modeling effort.

Examples of useful model-spec branches:

- forebody shell
- mid-body / cabin body
- tailcone / empennage root body
- wing-root blending body
- canopy break or payload fairing
- dense evidence visualization
- reduced progression visualization
- fitted curve overlay
- diagnostic comparison panel composition

## Spec Branching Rule

If a model would benefit from multiple lofts, CSG joins, or separate feature
lanes, the specs should surface that explicitly.

Do not hide multi-part modeling inside one oversized "single loft" concept if
the object naturally decomposes into:

- multiple lofts
- loft plus booleans
- loft plus secondary fairings
- separate visual proof assets

For example:

- a whole aircraft is often too ambitious for one loft
- a believable aircraft example may be:
  - main fuselage loft
  - canopy or payload blister loft
  - tailboom or tail-root loft
  - wing-root cue geometry
  - CSG combination or controlled overlap handling

## What A Modeling Spec Should Contain

Each modeling spec should describe at least:

- the part or modeling concern it owns
- the object read it must produce
- the operations expected
- the important sections / stations / profiles
- the feature-level success criteria
- what it depends on
- what level of detail it must reach
- how it will be visually reviewed
- its declared spec ledger and required atom count

See [Detail And Completeness](detail-and-completeness.md) for the discrete atom
model.

## Feature Demonstration Requirement

Modeling specs must not only describe geometry. They must also say how the
result demonstrates the feature or release delta.

Examples:

- which part of the model proves curve fitting
- which part proves reduced progression
- which part shows retained topology stations
- which part shows hidden control stations
- which part produces a comparison viewers can understand immediately

If no viewer can point to something in the final output and say "that required
the new version," then the model-plan/spec set is incomplete.

## Completion Rule For Planning

Model-plan work is complete enough to move into execution only when:

- the model decomposition is explicit
- the operations are plausible
- the visible success criteria are explicit
- the level of detail is defined
- the specs are broken into bounded modeling units
- the feature-demonstration surfaces are identified
- the modeling specs have declared atom ledgers

If any of those remain vague, continue refining the plan/spec set before
treating execution as the next step.

## Authority Boundary

This directive is a potential directive.

Within the workspace containing this `.agents/` folder, treat it as local
directive guidance for model planning and model-spec generation when relevant.

Do not add it to a skill set, do not publish it as canonical guidance, and do
not treat it as canonical unless it is explicitly moved out of
`.agents/potentials/`.
