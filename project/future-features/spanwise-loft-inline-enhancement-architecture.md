# Spanwise Loft Inline Enhancement Architecture

## Status

Future feature branch.

## Idea

Instead of planning only one station interval at a time, the loft planner could
recognize that a larger run of stations forms one coherent surface span.

In that case, the planner would emit a larger-span realization directly rather
than forcing the executor to realize many small local transitions.

## Architectural Placement

This branch lives inside the loft planning path:

```text
placed topology sequence
-> spanwise planner recognition
-> larger-span transition representation
-> executor
-> consolidated surface body
```

## Why It Matters

- preserves a canonical construction path
- avoids generating over-segmented local patch structure only to simplify it
  later
- allows the planner to make larger-span surface decisions while structural
  context is still available

## Main Challenge

This is the cleanest architectural path, but also the hardest one.

It requires the planner to reason about:

- more than one immediate interval at a time
- larger-span compatibility
- when local topology truth still allows larger-span surface realization

without turning the planner into an unconstrained surface guesser.

## Open Questions

- What larger-span evidence is sufficient for inline consolidation?
- How does the planner represent a larger-span result without breaking the
  current planner/executor boundary?
- How are local topology events preserved inside a longer-span realization?

