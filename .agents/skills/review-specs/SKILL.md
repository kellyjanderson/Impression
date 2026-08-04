---
name: review-specs
description: Use when the user says "review specs", "refine specs", or asks for an independent critical check of existing implementation specs. This review-only skill must not create the initial specs; it reviews already-written specs, updates defects, splits oversized leaves, verifies parent coverage, and repeats until a fixed point is reached with a per-request review-pass new-leaf ledger.
---

# Review Specs

Use this skill as an independent check action after spec creation or promotion.

## Review Boundary

Do not create the initial implementation specs during review. If specs do not
exist yet, stop and ask for `do specs` first.

Treat every spec as suspect, especially if it was created earlier in the same
conversation. The review action is not a continuation of creation; it is a
separate critical pass over written artifacts.

Immediately load and follow `specification-sizing` for parent and child split
coverage, canonical status, cohesion, and write-barrier requirements.

Load `architecture` and `architectural-change-document` when a review discovers
missing, stale, or aspirational architecture that must be captured before specs
can be ready.

Load the selected process registry, its current `implementation-spec` template,
and the shared or project-local specification review scoring policy before
scoring. The numeric `Review Score` is the sole score. A score is invalid unless
the current template was loaded, the total appears in the front matter, and
every category in the final `## Review Score Calculation` section was evaluated.
The front-matter score and final calculation total must match exactly. Normalize
outdated or incomplete score sections to the current template before calculating
a score, and remove obsolete secondary sizing metadata that the current template
no longer defines.

## Architecture Feedback Loop

Spec review must resolve architecture gaps as artifacts, not comments.

When a reviewed spec depends on architecture that is missing, stale, too vague,
or not linked:

1. Create or update an ACD for the missing, stale, desired, transitional,
   uncertain, or not-yet-implemented architecture.
2. Link the ACD from the spec's ancestry, prerequisites, source field carryover,
   and readiness fields as applicable.
3. Keep the spec not ready until the linked ACD defines enough
   target structure, ownership, routes, data boundaries, and follow-on specs to
   remove implementation guessing.

This feedback loop is allowed during spec review even though review must not
create the initial implementation specs. ACD updates are review findings
converted into durable architecture-tracking artifacts.

Never edit canonical architecture from `review specs`. Architecture files are
sacrosanct after specs start; ACDs exist so every architecture change discovered
after that point remains traceable until post-implementation reconciliation.

## Fixed-Point Review Loop

Continue until a full post-readback iteration finds no new child specs, no
Review Score at or above the forced-split threshold for intended final leaves,
no unresolved split coverage gaps, and no missing template/readiness field that
would force implementation guessing.

Every iteration must rescore adversarially from the current spec text using the
current implementation-spec template. Treat previous Review Scores, ledger
scores, and creation scores as claims to disprove. Do not copy a previous score
forward. Start by looking for undercounted responsibilities, hidden coupling,
missing readiness fields, omitted UI/control inventory, uncounted reuse work,
unrecorded routes, unresolved prerequisites, unresolved deferral markers, and
ambiguous verification. If the fresh score matches the prior value, record
which undercount risks were checked and why the prior value survived.

Apply the shared scoring policy's deferral marker tripwire before deciding that
no split or follow-up artifact is needed. Any unresolved reference to future
specs, blockers, `to be defined`/TBD/TBA, not-done work, incomplete work,
unfinished work, deferred work, or later work counts as a 100-point scoring
event unless it is clearly resolved or explicitly marked `none` or `not
applicable`.

At the start of each user request to `review specs`, create a new review ledger
for that request. Do not append to a ledger from an earlier request. Use a
request-specific path such as
`project/spec-refinement-history/<scope>-<YYYYMMDD-HHMMSS>.md` or a local
equivalent in the spec tree.

Each iteration must complete these steps:

1. Re-read the current specs from disk.
2. Review the complete template-governed Review Score, scope, implementation routing, dependencies/routes,
   application integration, reuse/extraction, required DTOs/functions/UI
   fields, performance, error/state behavior, test strategy, readiness
   checklist, parent coverage, and source field carryover.
3. Update defects found by the review. Do not preserve creator wording when it
   hides missing fields, overbroad scope, weak verification, or optimistic
   scores.
4. Create or update required ACD feedback artifacts before clearing
   architecture readiness blockers.
5. Adversarially rescore the numeric Review Score through the current template
   and record the split decision.
6. If a split is required, write child specs, verify parent coverage, then run
   this same review loop on those children before marking the parent iteration
   complete.
7. Re-read the active specs from disk and derive the list of new leaf specs
   created by this review round.
8. Write exactly one ledger entry that records only that the active specs were
   reviewed and the new leaf list for this round.
9. Re-read that ledger entry. If the new leaf list is `none`, stop the
   iterative review loop; another loop is not required. If the new leaf list is
   not `none`, continue with the new active spec set after ledger readback.

Do not prefill future iterations. Do not record multiple completed iterations
in one ledger write.

## Ledger

The review ledger is not an action log. It exists only to prove sequential
review passes over the active spec set and to expose the new leaf specs created
by each completed review round.

Create a new ledger for every user request to `review specs`. Do not reuse or
append to a previous request's ledger.

Only one event may trigger a ledger write: completing a fresh read and review
of the active specs and deriving the new leaf list for that round.

Do not write ledger entries for splitting, child spec creation, score changes,
ACD creation, prerequisite edits, readiness fixes, parent archive
updates, file writes, or any other action. Those actions may be reflected in the
spec files, ACDs, or final response, but they do not increment the ledger.

Each entry must include:

- request ledger path;
- review pass number for this request;
- exact active specs re-read for this pass;
- new leaf specs created by this review round, or `none`;
- parent specs that remain active only because split coverage is incomplete;
- fixed-point status: `reached` or `continue`;
- next active spec set to review after ledger readback.

The stop signal is explicit: an entry whose new leaf specs field is `none`
means the review round produced no new leaves and another loop is not required.
Do not run another review iteration merely to confirm the zero-new-leaf result.

## Reporting

Report the reviewed specs, template sources, final Review Scores, split
decisions, parent coverage, ACD feedback artifacts, remaining blockers, and the
per-request ledger path with the new-leaf entries for each review round.
If the fixed point was not reached, report the concrete blocker.
