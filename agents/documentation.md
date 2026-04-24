# Documentation Guidance

Documentation is a first-class deliverable in this repository.

It should be:

- comprehensive
- accurate
- beautiful
- easy to scan
- easy to trust
- pleasant to return to repeatedly

## Standard

When writing documentation, aim for the level of clarity and polish commonly associated with strong public developer docs such as:

- Stripe
- MDN
- React
- Tailwind CSS

The goal is not to imitate their branding.

The goal is to match their strengths:

- clear hierarchy
- strong page structure
- concise but sufficient prose
- practical examples
- easy navigation
- obvious next steps
- consistent terminology

## Required Qualities

Good documentation in this repository should:

1. explain what something is before diving into implementation details
2. state why the reader should care
3. define terms before relying on them
4. separate durable rules from temporary notes
5. include examples when examples reduce ambiguity
6. avoid sprawling walls of text with no visual structure
7. keep language precise without becoming sterile

## Preferred Structure

For most technical documents, prefer:

- short overview
- explicit backlinks or related documents
- flat sections with strong titles
- tables or lists when they genuinely improve scanning
- examples near the rule they clarify
- crisp acceptance or completion criteria

## Beauty Rules

Beautiful documentation here means:

- strong information hierarchy
- even pacing
- good whitespace
- no accidental redundancy
- no clutter pretending to be thoroughness
- examples that feel intentional

Beauty is part of usability.

## Accuracy Rules

Documentation must not describe behavior the code does not support unless the document is clearly marked as architecture, research, or future specification work.

When implementation changes invalidate docs, updating the docs is part of completing the work.

## Completion Rule

A feature or system area is not fully complete if:

- the implementation exists
- the tests exist
- but the durable documentation remains missing or stale

Documentation completion is part of the delivery contract, not optional polish.

## Durable Planning Note

When work is proceeding on the ad hoc path instead of the feature/specification
path, documentation still needs a durable project-facing record.

That record should live under:

```text
project/adhoc/
```

Ad hoc documentation should capture the developer/project perspective on:

* what changed
* why it changed
* what boundaries the change intentionally did not cross
* how the change was verified

Ad hoc notes are not public feature docs. They are the lightweight durable
planning/implementation record for bounded work that is not using the full
feature path.
