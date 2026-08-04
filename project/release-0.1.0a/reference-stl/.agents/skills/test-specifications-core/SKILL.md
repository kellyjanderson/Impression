---
name: test-specifications-core
description: Create one durable paired test specification for each final feature leaf and define manual smoke, automated smoke, and automated acceptance expectations.
---

# Test Specifications Core

Test specifications define how a final feature leaf should be verified manually and automatically.

## Core Rule

There should be one test specification for each final feature leaf specification whose acceptance describes a durable feature contract.

Low-level support leaves do not automatically need standalone test specifications unless they are being treated as first-class feature leaves.

## Timing

When a feature specification is created as a final leaf, its test specification
should be created in the same `do specs` pass.

Implementation should not start with feature behavior fully specified but verification shape still unwritten.

If a feature spec is a child of an incomplete parent split, the paired test spec
is temporary until split coverage is complete. After the child becomes
canonical, update the test spec backlink to the canonical child and architecture
ancestor. Do not use a parent or umbrella test spec as final coverage for
canonical children.

## Recommended Structure

Test specifications should usually include:

* overview
* backlink to the feature specification
* manual smoke check
* automated smoke tests
* automated acceptance tests
* optional implementation-facing notes

Include feature spec canonical status and architecture ancestor when the source
spec came from a parent split or ACD.

## Verification Emphasis

Manual guidance may stay light.

Automated guidance should carry more detail, especially around:

* smoke tests that fail quickly and clearly
* acceptance tests that prove the feature contract
* stable fixtures
* explicit regression cases

## Relationship To Code Tests

Test specifications do not replace automated tests in the codebase.

They define:

* what coverage should exist
* what manual fallback check should exist
* what fixtures, doubles, and observable outcomes matter

## SkillsKeeper Directives

<!-- skillskeeper-directive: integrated-route-acceptance -->
### Integrated Route Acceptance

## Integrated Route Acceptance

Every user-facing feature test specification should include at least one acceptance item that proves the feature is reachable through its intended app surface, command, API, or workflow.

For GUI features, acceptable proof may include a Qt signal/slot integration test, widget event smoke, offscreen launch plus state inspection, or manual smoke where UI automation is impractical.

Async feature specs should include stale-result or adjacent-path validation so helper tests cannot pass while the integrated route still corrupts newer UI or app state.
<!-- /skillskeeper-directive: integrated-route-acceptance -->

<!-- skillskeeper-directive: route-specific-integration-proof -->
### Route-Specific Integration Proof

## Route-Specific Integration Proof

Test specifications must choose integration proof based on app type and fail work that exists only as isolated classes, services, adapters, registries, or helpers.

Require at least one route-level proof when applicable:

- GUI route smoke: visible entrypoint/event, state change, UI-thread handoff, stale result behavior, signal/slot integration, widget event smoke, offscreen launch, or manual GUI smoke.
- Console command smoke: executable/subcommand, flags/args/stdin/config, stdout/stderr, exit code, side effects, golden output, temp-directory integration test, or manual command transcript.
- API/service route smoke: endpoint/method, event topic, RPC method, queue message, scheduled job, auth/permission behavior, request/response, status/error shape, side effects, and observability.
- Mixed-surface smoke: separate proof for each user/caller route that can fail independently.

Helper-level tests remain useful, but they do not satisfy route-level acceptance for feature behavior.
<!-- /skillskeeper-directive: route-specific-integration-proof -->

<!-- skillskeeper-directive: test-specification-template-registry -->
### Test Specification Template Registry

## Test Specification Template Registry

When creating paired feature test specifications, load the `test-specification` template from the selected process registry before drafting the document.

Selection order:

1. `project/process/skills-templates-manifest.md` key `test-specification`, when present.
2. `.agents/process/skills-templates-manifest.md` key `test-specification`, when present.
3. `.agents/process/templates/test-specification-template.md` from the nearest shared ancestor.

The selected template is authoritative for route-specific GUI, console, API/service, workflow, mixed, and library-only proof sections.
<!-- /skillskeeper-directive: test-specification-template-registry -->
