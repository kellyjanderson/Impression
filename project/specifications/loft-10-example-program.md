> Status: Deprecated historical spec.
> Its intent is preserved by the project DNA and later loft/surface planning,
> but it is no longer an active implementation specification. Retained for
> project history only.

# Loft Spec 10: Example Program (Impress + Teach)

This specification defines the loft example strategy for Impression.

Examples are not only API demonstrations; they are both:

- teaching tools (how and when to use loft)
- marketing assets (first impression of capability and quality)

The standard is: examples must be context-rich, visually strong, and useful.

---

## 1. Principle

Avoid contrived shape-morph blobs as primary examples.

Preferred examples:

- solve recognizable real modeling problems
- show why loft is the correct tool vs simple extrude/revolve
- communicate outcomes users care about (fit, flow, ergonomics, aesthetics, manufacturability)

---

## 2. Required Coverage Matrix

The example suite must collectively exercise:

1. path-oriented lofting with station frames
2. endcap modes (`FLAT`, `CHAMFER`, `ROUND`, `COVE`)
3. topology transitions (hole/region birth and death)
4. multi-region section correspondence
5. deterministic behavior under reordered section content
6. printable/mechanical output quality (watertight)

---

## 3. Example Categories

### A. Mechanical Utility

Purpose: show practical engineering transitions.

Candidates:

- round-to-rect duct adapter
- hose barb/nozzle transition
- cable gland strain-relief body
- ergonomic tool-handle core with cap options

What to highlight:

- exact section dimensions
- cap selection tradeoffs
- alignment and orientation behavior

### B. Organic / Product Form

Purpose: show smooth form development where loft bridges mechanical and organic design.

Candidates:

- toothbrush/electric-handle shell transition
- bottle neck-to-body profile family
- wearable contour band segment

What to highlight:

- section choreography along path
- curvature continuity impression
- manufacturable shell outcomes

### C. Topology Narrative

Purpose: make topology transitions understandable and compelling.

Candidates:

- vent body where hole appears then disappears
- dual-island feature that joins to single body (merge behavior once supported)
- boss/island emergence from a base profile

What to highlight:

- “before/after” meaning, not abstract math
- explicit labels for supported vs rejected transitions
- deterministic repeatability

### D. Hero Demo

Purpose: a single flagship scene that feels premium and memorable.

Requirements:

- staged sequence with titles/subtitles
- multiple real scenarios in one curated run
- camera-ready composition and palette
- final “takeaway” message about capability, not just animation

---

## 4. Per-Example Contract

Every shipped loft example must include:

1. one-sentence real-world problem statement
2. why loft is used here
3. key parameters exposed at top of file
4. at least one parameter set users can change to immediately see tradeoffs
5. build output that renders clearly in preview without custom setup

---

## 5. Naming and Organization

Place examples under:

- `docs/examples/loft/real_world/`

Naming pattern:

- `loft_<domain>_<purpose>_example.py`

Examples:

- `loft_hvac_round_to_rect_example.py`
- `loft_handle_ergonomic_example.py`
- `loft_topology_vent_transition_example.py`
- `loft_hero_showcase_example.py`

---

## 6. Visual Quality Rules

Examples should be intentionally art-directed:

- clear object scale and camera framing
- readable color separation by component role
- no overlapping clutter that obscures geometry behavior
- captions or label meshes for mode/feature clarity where needed

“Technically correct but visually unclear” is a failing example.

---

## 7. Documentation Integration

For each example:

- add a short section in `docs/modeling/loft.md`
- include a screenshot generated from the example
- explain “what problem this solves” in plain language

The docs page should read like a capabilities narrative, not a function catalog.

---

## 8. Acceptance Criteria

The loft example program is complete when:

1. at least 6 real-world loft examples exist across the categories above
2. at least 1 hero demo exists and runs as a standalone showcase
3. topology transitions are demonstrated in context (not isolated test geometry)
4. examples are referenced from loft docs with purpose-first explanations
5. first-time users can identify at least three practical use-cases immediately
