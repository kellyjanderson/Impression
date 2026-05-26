# How To Model

## Purpose

This directive defines a modeling workflow that uses render-and-review loops to
replicate the way a human modeler tries a shape, looks at it, and refines it.

The intent is not merely to create geometry that technically satisfies a prompt.
The intent is to create models that actually read as the intended real-world
thing and survive critical visual inspection.

## Core Rule

When creating a model, leverage agent perceptive models by sending renderings
into sub-agents and then reviewing the output of those sub-agents.

Treat this as a normal part of modeling work, not as an exceptional step.

## Visual Review Loop

Use a repeated loop:

1. model a meaningful increment
2. render the result
3. send one or more renderings to a sub-agent for visual critique
4. review the sub-agent feedback critically
5. modify the model
6. repeat until the modeling goal is reached or the current blocker is clearly
   understood

This is intended to replicate the human ability to try something, look at it,
and make adjustments.

## Sub-Agent Review Posture

Sub-agents used for visual review should be:

- critical
- constructive
- specific
- willing to say when the model does not yet read correctly

The goal is not approval. The goal is useful criticism.

Sub-agents should be asked to comment on things like:

- silhouette quality
- proportion
- topology read
- whether the model resembles the intended object class
- where the form looks awkward, generic, swollen, pinched, underdefined, or
  accidentally stylized
- whether the model demonstrates the intended feature or release delta clearly

## Multiple Angles

You may and should ask sub-agents to review multiple angles when the object
benefits from it.

Useful views often include:

- front
- side
- top
- isometric
- close detail views of the most important transitions

Choose views that expose the actual modeling risk, not just views that look
good.

## Working With Sub-Agent Feedback

You may respond to sub-agents immediately with:

- changes to the model
- follow-up renderings
- discussion of why a form is not working yet
- requests for comparison between successive iterations

Use your own judgment together with the sub-agent feedback to decide what to
change.

Do not blindly obey any one sub-agent response.

## Reference Images

You may and should find visual examples of what you are modeling.

When reference images are useful:

- store them as durable reference images
- keep them associated with the model or example being developed
- send those reference images together with rendered output for visual
  comparison

Reference images are especially useful for:

- proportion
- silhouette
- transition quality
- believable topology
- object-category recognition

## Research During Modeling

If you are having difficulty generating some aspect of the model, you may and
should do research.

That research may help identify:

- missing product features
- missing low-level modeling primitives
- better workflows
- concrete modeling answers
- real-world object constraints
- known CAD pain points for similar forms

## Durable Modeling Research

Model-specific research should be stored in a durable local research folder near
the modeling work.

That research document should capture:

- what problem was being investigated
- what sources were consulted
- what conclusions were reached
- what those conclusions imply for the current model
- whether the research points to a missing feature, workflow gap, or modeling
  decision

Do not let useful modeling research remain only in transient conversation.

## Planning Before Modeling

Do not jump from prompt directly into full model execution when the shape or
feature demonstration is complex.

Start from:

1. a model-plan document
2. then one or more modeling specs
3. then implementation / iteration work

See:

- [Model Planning And Specs](model-planning-and-specs.md)
- [Detail And Completeness](detail-and-completeness.md)

## Expected Output Quality

Avoid:

- AI-shaped blobs
- forms that technically demonstrate a feature but do not read as anything
  believable
- settling for a model before silhouette and proportion are convincing

Aim for:

- intentional shape
- believable object read
- clear feature demonstration
- documented reasoning about the important modeling decisions

A model that is merely "watertight" or "technically lofted" is not enough if a
viewer cannot tell what object class it belongs to or what new capability it is
supposed to demonstrate.

## Authority Boundary

This directive is a potential directive.

Within the workspace containing this `.agents/` folder, treat it as local
directive guidance for modeling work when relevant.

Do not add it to a skill set, do not publish it as canonical guidance, and do
not treat it as canonical unless it is explicitly moved out of
`.agents/potentials/`.
