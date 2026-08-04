---
name: specification-sizing
description: Apply specification cohesion, split-coverage, canonicalization, and readiness checks when review-specs judges whether a specification leaf is too broad, when parent specs must split into child specs, and when canonical child specs can replace fully covered parent specs. This is a low-level structural rule skill; use review-specs for independent spec review actions.
---

# Specification Sizing

Use this skill from `review-specs` when deciding whether a specification is a
cohesive independently deliverable leaf, whether it needs child specifications,
and whether children fully cover a split parent.

The numeric `Review Score` defined by the shared scoring policy is the sole
score. It is valid only when the current implementation-spec template was loaded
through the selected process registry, its total appears in the front matter,
and every category in its complete final `## Review Score Calculation` section
was evaluated. The two totals must match exactly.

## Cohesive Leaf Rule

A final implementation leaf must describe one independently deliverable,
reviewable change boundary with:

* one primary outcome
* one coherent responsibility boundary
* one reviewable artifact or change set
* explicit verification
* declared inputs and outputs
* resolved or explicitly named assumptions and decisions

These are qualitative readiness requirements, not a second score. Split or
revise a specification when the boundary is plural, ambiguous, unnamed, or can
fail and be delivered independently.

The shared Review Score thresholds remain authoritative:

* `25+`: split required before implementation
* `16-24`: explicit split review and a strong cohesion reason required
* `0-15`: may remain whole when the cohesive leaf rule and readiness fields are
  satisfied

The shared scoring policy's unresolved deferral/gap marker tripwire is also
authoritative. Any unresolved reference to future specs, blockers, `to be
defined`/TBD/TBA, not-done work, incomplete work, unfinished work, deferred
work, or later work is a 100-point scoring event and prevents final status until
resolved.

## Split Coverage Rule

When a parent spec is split, the split is not complete until child specs cover
100% of parent responsibilities. Do not leave uncovered responsibilities parked
only in the parent. Move each uncovered responsibility into an existing child
or create a new child, then verify coverage again.

During an incomplete split, child specs must keep the parent spec as primary
ancestor. After coverage reaches 100%, children become canonical specs with the
architecture document or ACD as primary ancestor and the parent as split
provenance.

Use a coverage matrix with these statuses:

* `Covered`: a child fully owns the parent responsibility.
* `Moved this pass`: a child was updated and needs verification.
* `Partial`: invalid as a stopping state.
* `Missing`: invalid as a stopping state.
* `Parent-only`: invalid as a stopping state.

Do not archive or demote a parent until the child re-review loop has completed
and every parent responsibility is `Covered` by children.

## Fixed-Point Refinement Rule

Do not rely on a fixed pass count. Continue until a complete review of the
active specs produces:

* no new child specs
* no Review Score at or above the forced-split threshold for an intended final
  leaf
* no unresolved split coverage gaps
* no failed cohesive leaf rule
* no missing template or readiness fields that would force implementation
  guessing

Each review iteration must independently recalculate the Review Score from the
current spec using the current implementation-spec template. Prior scores are
claims to disprove, not baselines.

The review ledger is controlled by `review-specs`. Do not write ledger entries
for score changes, split creation, child writes, parent archival, coverage
updates, or readiness changes. A request-round ledger entry whose new leaf list
is `none` is the terminal fixed-point signal.

## Reporting Rule

At the end of a structural sizing check, report:

* the implementation-spec template source used for scoring
* final Review Scores and split decisions
* cohesion decisions for `16-24` leaves
* specs still requiring splits
* unresolved parent coverage
* parents fully covered and ready to archive
* remaining readiness blockers
* the request-specific review ledger path and terminal new-leaf entry
