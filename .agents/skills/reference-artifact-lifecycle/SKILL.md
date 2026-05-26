---
name: reference-artifact-lifecycle
description: Manage Impression reference images and STL artifacts as part of completion for model-outputting work, including dirty or clean lifecycles, bootstrap, invalidation, and promotion rules.
---

# Reference Artifact Lifecycle

Use this Skill for model-outputting work in Impression.

## Completeness Rule

Any capability that outputs a model must have durable reference-artifact coverage before the work is considered complete.

By default that means:

* a rendered reference image
* an exported reference STL

## Storage

Reference artifacts live under:

```text
project/reference-images/
project/reference-stl/
```

Organize dirty and clean baselines separately.

## Dirty And Clean References

Dirty references are unreviewed change-detection baselines.

Clean references are explicitly reviewed and accepted.

Dirty references may prove output changed.
They do not prove the new output is aesthetically or geometrically correct.

## Bootstrap And Invalidation

On the first run for a fixture with no baseline:

* create dirty references
* do not silently promote them to clean

When the reference-test contract changes, invalidate the old dirty and clean baselines before trusting new runs.

## Promotion Rule

Dirty to clean promotion requires explicit human review.

Agents must not silently promote dirty references.

## Test Rule

Reference-artifact tests must:

* generate fresh artifacts
* prove they are non-empty and model-related
* compare against the selected baseline
* fail clearly when only part of the required artifact set exists

Prefer clean references when present.
Otherwise fall back to dirty references.
