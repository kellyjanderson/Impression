# Detail And Completeness

## Purpose

This directive proposes a way to define modeling detail and spec completeness
so model-spec work has a meaningful stopping point.

## Core Idea

Modeling detail should be expressed relative to scale.

At minimum, define:

- the total size or envelope of the object being modeled
- the smallest feature scale that the model is expected to represent

Detail is then expressed as a ratio between those two.

## Working Terms

### Total Model Scale

The characteristic overall size of the modeled object.

Examples:

- overall length of an aircraft fuselage
- overall height of a bottle
- overall diameter of a vessel

### Detail Floor

The smallest feature class that the example is expected to model explicitly.

Examples:

- canopy break
- wing-root shoulder
- payload blister
- panel break
- root fairing

Features smaller than the declared detail floor may be intentionally omitted,
abstracted, or deferred.

### Detail Ratio

`detail_ratio = smallest_explicit_feature_size / total_model_scale`

This is not meant to be mathematically perfect. It is meant to force the model
plan/spec set to declare how fine the example is trying to go.

## Why This Matters

Without a declared detail floor:

- spec work can continue forever
- agents do not know whether to keep refining
- "complete" becomes subjective
- viewers may judge the model against a different implied scope than the plan

## Detail Declaration In Model Plans

Every non-trivial model plan should declare:

- overall scale
- target detail floor
- the kinds of features above that floor that must be represented
- the kinds of features below that floor that may be omitted

Example:

- total scale: `1200 mm fuselage length`
- detail floor: `40 mm feature class`
- required above-floor features:
  - payload forebody
  - canopy break
  - wing-root shoulder
  - tailcone transition
- omitted below-floor features:
  - fasteners
  - inspection hatches
  - skin seams

## Discrete Spec Completeness

Spec completeness should be treated as a counted planning measure, not just a
feeling.

The unit of completeness is a **spec atom**.

A spec atom is one required planning statement that can be counted as either:

- `0` = missing
- `1` = present

There are no partial atom values.

## Spec Atom Types

Each modeling spec must declare atoms in the following buckets.

### 1. Feature Atoms

One atom per above-floor feature the spec owns.

Examples:

- payload forebody
- canopy break
- wing-root shoulder
- tailcone transition

### 2. Transition Atoms

One atom per critical transition the spec must explain.

Examples:

- forebody to cabin
- cabin to wing-root shoulder
- aft body to tailcone

### 3. Operation Atoms

One atom per feature or transition that must be assigned to an operation
family.

Examples:

- loft
- multi-loft
- loft plus CSG union
- loft plus subtraction
- overlay / diagnostic lane

### 4. Review Atoms

One atom per required visual review target.

Examples:

- side silhouette read
- top planform read
- front-section read
- feature-delta comparison panel

### 5. Omission Atoms

One atom per intentionally omitted below-floor feature class or deferred part
class.

Examples:

- fasteners omitted
- skin seams omitted
- full wing surfaces deferred
- full empennage deferred

### 6. Demonstration Atoms

One atom per release-delta claim that must be made visible.

Examples:

- dense evidence lane is visible
- reduced progression lane is visible
- retained stations are visible
- fitted curve is visible
- refusal / uncertainty posture is visible

## Spec Ledger

Each modeling spec should carry a ledger in this shape:

- `feature_atom_count`
- `transition_atom_count`
- `operation_atom_count`
- `review_atom_count`
- `omission_atom_count`
- `demonstration_atom_count`
- `required_atom_count`
- `completed_atom_count`
- `completeness_ratio = completed_atom_count / required_atom_count`

Where:

- `required_atom_count` is the total number of declared required atoms
- `completed_atom_count` is the number of those atoms actually resolved in the
  spec

This makes completeness a discrete value over an explicit finite set.

## Atom Completion Rule

An atom counts as completed only if it is fully named and resolved at the
planning level.

Examples:

- a feature atom is complete only when the feature is named and owned by the
  spec
- a transition atom is complete only when the transition is named and described
- an operation atom is complete only when the operation family is assigned
- a review atom is complete only when the review target is named
- an omission atom is complete only when the omission is declared intentional
- a demonstration atom is complete only when the visible proof surface is named

A vague sentence does not count as multiple atoms.

## Minimum Completion Condition

A modeling spec is not complete enough for execution until:

1. `required_atom_count` is declared
2. every required atom is assigned to one of the atom buckets
3. `completed_atom_count == required_atom_count`
4. `completeness_ratio == 1.0`

Until then, the spec is still in refinement.

## Scale And Detail Coupling

The atom system only works if scale and detail floor are declared first.

That is because:

- the detail floor determines which features become feature atoms
- the total scale determines what counts as above-floor vs below-floor
- omitted below-floor items become omission atoms instead of hidden work

## Operation Assignment Rule

A feature is not fully specified until the spec says what kind of operation is
expected to create it.

Examples:

- single loft
- multi-loft body
- loft plus CSG union
- loft plus subtraction
- secondary fairing loft
- explicit diagnostic overlay lane

This helps reveal hidden work early.

## Discrete Example

Example airframe forebody spec:

- total scale: `1200 mm`
- detail floor: `40 mm`

Required atoms:

- feature atoms: `4`
  - payload nose
  - sensor bay
  - canopy break
  - wing-root shoulder cue
- transition atoms: `3`
  - nose to sensor bay
  - sensor bay to canopy break
  - canopy break to shoulder cue
- operation atoms: `4`
  - nose loft
  - sensor blister loft
  - canopy break loft
  - CSG union / blending decision
- review atoms: `3`
  - side silhouette
  - top planform
  - front section
- omission atoms: `2`
  - fasteners omitted
  - surface panel lines omitted
- demonstration atoms: `2`
  - dense evidence lane visible
  - reduced progression visible

Then:

- `required_atom_count = 18`
- if `completed_atom_count = 15`
- `completeness_ratio = 15 / 18 = 0.8333`

That spec is still incomplete.

## Refinement Trigger

A model part probably still needs refinement if:

- `required_atom_count` is not declared
- atom buckets are missing
- any required atom is still unresolved
- it owns multiple object-identifying features but too few operation atoms
- it combines multiple scales of detail without declaring a detail floor
- it hides major decompositions such as "whole airframe in one loft"
- it does not say what a viewer should be able to recognize at the chosen scale

## Completion Trigger For Spec Work

Model-spec work can stop when:

- the chosen scale is declared
- the detail floor is declared
- `required_atom_count` is declared
- all required atoms are assigned to buckets
- `completed_atom_count == required_atom_count`
- `completeness_ratio == 1.0`

At that point, execution can start without pretending that the planning still
needs hidden architecture work.

## Authority Boundary

This directive is a potential directive.

Within the workspace containing this `.agents/` folder, treat it as local
directive guidance for defining modeling detail and spec completeness when
relevant.

Do not add it to a skill set, do not publish it as canonical guidance, and do
not treat it as canonical unless it is explicitly moved out of
`.agents/potentials/`.
