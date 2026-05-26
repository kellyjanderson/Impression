# Surface Spec 40: Open-Surface Classification and Mesh QA Contract (v1.0)

## Overview

This specification defines how tessellated outputs representing open surfaces
are classified and what mesh QA is expected to report for them.

## Backlink

Parent specification:

- [Surface Spec 13: Seam-Consistent Tessellation and Watertight Output Rules (v1.0)](surface-13-seam-consistent-tessellation-watertightness-v1_0.md)

## Scope

This specification covers:

- open-surface mesh labeling
- expected QA results for open outputs
- distinction between intentionally open output and defective closed output

## Behavior

This branch must define:

- how open outputs are marked or classified
- which QA checks still apply to open outputs
- how QA distinguishes intentional open boundaries from invalid gaps

## Constraints

- open-output classification must be explicit
- QA must not mislabel intentional open surfaces as closed-body failures
- open/closed distinction must remain deterministic

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
covers one classification contract and one QA expectation set.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- open-output classification is explicit
- QA expectations for open outputs are explicit
- intentional-open versus defective-closed distinction is explicit

