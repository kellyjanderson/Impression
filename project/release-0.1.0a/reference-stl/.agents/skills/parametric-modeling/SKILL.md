---
name: parametric-modeling
description: Use when creating, editing, reviewing, or refactoring parametric model code, especially geometry functions, reusable model components, CSG/cutters, local coordinate construction, feature composition, placement, exact geometric relationships, or modeling-code organization. Apply before framework-specific modeling skills such as Impression.
---

# Parametric Modeling

Use this skill when model code should be structured as reusable, exact,
composable geometry. Framework-specific skills may add API rules, but this skill
owns the modeling discipline.

## Core Directives

### 1. Composition Ladder

Build the smallest reasonable bits first. Compose those bits inside the next
most complex object. Then compose those objects into more complex assemblies.

Each function should isolate its own concern and return a finished piece that
its caller can compose.

The model should grow through a ladder:

- primitive shape;
- named local sub-object;
- named feature;
- named assembly;
- final printable or renderable part;
- print or scene layout.

Do not skip directly from primitives to final assembly when intermediate objects
can be named and reasoned about. At each level, the composition function should
read like a description of the object it returns.

### 2. Cutters Are Internal Geometry Concerns

Cutters belong to the geometry function that needs them.

If a function models an object that requires subtractive operations, the function
should create the cutter, apply it to the object it owns, and return the
finished modeled geometry.

Do not casually return loose cutters for a distant caller to apply over a larger
assembled body. If a cut must happen at a higher composition level, model that
cutter as its own named object and make the target explicit in the function name
and call site.

The default rule is: cutters are contained inside the function that owns the
geometry being cut.

### 3. Encapsulation And Composition Repair Gate

Before editing a parametric model, review the relevant code for geometry that
does not satisfy encapsulation and composition.

Here, encapsulation and composition mean:

- build geometry through the composition ladder, where each function owns one
  concern and returns finished modeled geometry for its caller to compose;
- keep concerns contained inside the model function that owns them. Cutters,
  geometry, transforms, placement assumptions, and intermediate construction are
  local concerns. A model function takes inputs, contains its machinery, and
  returns modeled geometry. Higher-level functions compose returned geometry,
  add their own contained geometry, and return the result.

Encapsulation also defines the boundary between local construction and
placement. A reusable model function should build geometry in local canonical
coordinates first. Placement belongs at the caller or composition boundary.

Local construction answers: what is this object?

Placement answers: where does this object go in the larger model?

Look for:

- unnamed shape fragments buried in larger functions;
- cutters escaping the function that owns the geometry being cut;
- placement mixed into local shape construction;
- named, measurable, reusable, transformable, or visually inspectable concerns
  that lack a containing model function;
- repeated or copied geometry instead of reusable generators;
- assembled printable/renderable parts reused as unrelated cutters;
- feature code that bypasses the composition ladder;
- accumulated spot changes that should be rolled up into named functions.

If you find code that violates encapsulation or composition on the path of the
requested change, fix that structure before continuing the feature change.

If the repair is broader than the requested modeling change can safely include,
document it as a `codeimprovement` issue using the `coding` skill's Code
Improvement Issues process. The issue must name the affected model files and
line-number blocks, the violated modeling concern, and the proposed composition
boundary.

### 4. Build And Maintain A Shared Lexicon

Build the model's vocabulary interactively with the user.

The agent and user should agree on what each physical thing is called. Use soft
correction and direct naming questions to maintain a consistent lexicon.

Once a name is agreed, reinforce it in function names, parameter names,
metadata, comments, and discussion. When the user's wording reveals a better
name, realign the code vocabulary instead of preserving stale names.

### 5. Look For Existing Composable Models

Before modeling a new concern, look for existing composable models that can
satisfy part or all of that concern.

If an object already exists, reuse the function that generates it instead of
copying its geometry.

If an existing model is close but has the wrong size, clearance, orientation, or
placement assumptions, parameterize the function so it can satisfy the new
concern through composition.

### 6. Decompose To First-Principles Geometry

Break a modeling problem into smaller concerns until each concern can be
understood in first-principles geometric terms.

Ask what physical problem the geometry is solving: clearance, support,
retention, alignment, insertion, strength, printability, wire protection, force
transfer, or another named concern.

Model the simplest decomposition that satisfies the current concern before
adding secondary geometry.

### 7. Express Models Through Exact Parametric Geometry

Express models through math, geometry, coordinate systems, and algorithms.

Geometry should be exact and computable. The model should define enough
relationships that faces, points, edges, distances, directions, and tangents can
be calculated directly with minimal code.

### 8. Keep Model Functions Composable

Do not define model functions inside other functions unless there is a good
reason to make them private to that function.

Model functions should usually be defined at a scope where they can be reused,
parameterized, tested, previewed, and composed by other model functions. Hiding
a model definition inside another function defeats composition unless the nested
definition is truly an implementation detail that should not be composed
elsewhere.

## SkillsKeeper Directives

<!-- skillskeeper-directive: promote-substantial-objects-to-model-modules -->
### Promote Substantial Objects To Model Modules

Treat each substantial named physical object or assembly as an Impression model module, not merely as a helper function. If an object is complex enough to preview, tune, test, reuse, discuss by name, or compose into a larger object, it should usually live in its own module with its own build(params=...) function. Each model module owns its parameter type or parameter slice, local geometry functions, internal cutters, build(params=...) boundary, and optional preview or export helpers. Parent model modules compose child model modules by importing them, calling the child module build(...) function, and transforming the returned finished model at the parent assembly boundary.
<!-- /skillskeeper-directive: promote-substantial-objects-to-model-modules -->

<!-- skillskeeper-directive: do-not-name-implicit-properties -->
### Do Not Name Implicit Properties

Name functions and modules after the physical object or assembly they model, not after implicit implementation properties. Canonical local construction, parameterization, and composability are default properties of model modules. Do not encode those defaults in names with suffixes such as _local, _generated, _helper, or _model unless the distinction is truly part of the domain or resolves a real ambiguity. Prefer names like support_ramp, usb_c_port, module_board, pin_row, and usb_assembly.
<!-- /skillskeeper-directive: do-not-name-implicit-properties -->

<!-- skillskeeper-directive: shared-nouns-trigger-reuse-extraction -->
### Shared Nouns Trigger Reuse Extraction

When two model modules share the same physical noun, treat it as an opportunity for reuse. Decompose the modules, identify the shared concern, create a new module for that shared concern, and compose both original modules from the new composable module. For example, if screw_tab_support_ramp and usb_c_support_ramp both contain ramp, extract a shared support_ramp module and have both the screw tab module and USB assembly module compose support_ramp.build(...).
<!-- /skillskeeper-directive: shared-nouns-trigger-reuse-extraction -->

<!-- skillskeeper-directive: extraction-must-beat-inline-cost -->
### Extraction Must Beat Inline Cost

Do not extract a function when the function definition plus call site requires more code than the inline expression and does not add a reusable named domain concern. As a deterministic lowest bar, if a function's implementation is shorter and clearer inline than the function's name, signature, call, and body, inline it. Use extraction only when it reduces total reading cost or creates a boundary that future callers can meaningfully compose.
<!-- /skillskeeper-directive: extraction-must-beat-inline-cost -->

<!-- skillskeeper-directive: functions-must-earn-their-names -->
### Functions Must Earn Their Names

A model function must earn its name by owning a concern: a model object, cutter, placement rule, composition step, validation rule, reusable calculation, or domain concept. Simple arithmetic and one-off parameter plumbing do not earn a function boundary unless repeated or needed to clarify a real geometric relationship. Do not create tiny helpers just to make code look organized; keep obvious arithmetic at the owning concern boundary.
<!-- /skillskeeper-directive: functions-must-earn-their-names -->

<!-- skillskeeper-directive: deferred-model-code-improvements -->
### Deferred Model Code Improvements

When model-code cleanup is too broad or risky for the current task, create or update a `codeimprovement` issue using the `coding` skill's Code Improvement Issues process. Include `code-location` blocks for the model files and line ranges, identify the broken modeling rule, and describe the proposed reusable model, composition boundary, cutter ownership, placement boundary, or naming repair.
<!-- /skillskeeper-directive: deferred-model-code-improvements -->
