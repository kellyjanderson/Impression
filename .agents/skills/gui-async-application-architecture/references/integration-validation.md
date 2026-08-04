# Integration Validation

GUI async work is not complete until the integrated route has been exercised.

## Acceptable Proof

- UI event invokes the intended controller/service/worker.
- Qt signal/slot integration test proves handoff and UI-thread mutation.
- Widget event smoke proves visible state change.
- Offscreen launch plus state inspection proves the app route works.
- Manual smoke proves the route when automation is impractical.
- Subprocess crash, timeout, cancellation, or stale-result smoke proves failure isolation.

## Required Evidence

Name the user action or automatic trigger, the wiring module, the task lane, the completion route, and the validation that crossed the app boundary.

Helper tests are useful but insufficient when they do not exercise the route that users or callers actually use.
