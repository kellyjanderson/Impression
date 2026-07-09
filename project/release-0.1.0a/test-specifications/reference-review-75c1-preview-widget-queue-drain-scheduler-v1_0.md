# Reference Review Test Spec 75c1: Preview Widget Queue Drain Scheduler (v1.0)

## Paired Specification

- [Reference Review Spec 75c1](../specifications/reference-review-75c1-preview-widget-queue-drain-scheduler-v1_0.md)

## Automated Tests

- enqueue schedules one drain
- repeated enqueue while scheduled does not schedule nested drains
- drain clears pending scheduled state
- close clears pending commands before renderer disposal

## Acceptance

- widget drain-scheduler tests pass
