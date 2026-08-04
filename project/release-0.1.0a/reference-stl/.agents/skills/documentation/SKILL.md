---
name: documentation
description: Write or update durable project documentation with strong structure, accuracy, and completion discipline.
---

# Documentation

Documentation is a first-class deliverable.

## Standard

Aim for documentation that is:

* comprehensive
* accurate
* easy to scan
* pleasant to return to
* trustworthy

Optimize for clear hierarchy, precise language, and examples that remove ambiguity without creating clutter.

## Required Qualities

Documentation should:

* explain what something is before diving into implementation detail
* define terms before relying on them
* separate durable rules from temporary notes
* include examples when they reduce ambiguity
* avoid large unstructured walls of text

## Preferred Shape

Prefer:

* short overview
* strong section titles
* explicit backlinks or related documents
* lists or tables when they improve scanning
* crisp completion or acceptance language

## Accuracy Rule

Documentation must not describe implementation behavior the code does not support unless the document is clearly architecture, research, or future planning work.

Updating stale docs is part of completing the work.

## Completion Rule

A feature or system area is not fully complete when:

* implementation exists
* tests exist
* durable documentation is still missing or stale

Documentation completion is part of delivery, not optional polish.

## SkillsKeeper Directives

<!-- skillskeeper-directive: honest-completion-language -->
### Honest Completion Language

## Honest Completion Language

Durable documentation must distinguish product reality from implementation artifacts. Prefer explicit states when describing feature status:

- `Designed`: architecture or spec exists, but implementation is not complete.
- `Implemented in isolation`: code or focused tests exist, but the app does not call it.
- `Wired`: the app route calls it, but integrated behavior has not been validated.
- `Integrated`: the intended route works and validation passes.
- `User-accessible`: a user can discover and use it through the product surface.
- `Complete`: user-accessible, validated, documented, and progression/spec state updated honestly.

Docs must not imply a feature is complete when it is only implemented as unreachable code or validated only through helper-level tests.
<!-- /skillskeeper-directive: honest-completion-language -->
