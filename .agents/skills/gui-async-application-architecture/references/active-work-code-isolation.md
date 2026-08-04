# Active-Work Code Isolation

Active-work code is code the user or agent is editing while the app may execute it for previews, tests, plugins, notebooks, renderers, model builds, or automation.

## Core Rule

Active work files are expected to be broken. Broken active code must not crash, hang, poison, or exit the host app.

Prefer supervised process boundaries for user-authored, generated, plugin, model, preview, or native-heavy code. Same-process execution must be rare and justified.

## Required Checks

- `BaseException`, `SystemExit`, and process exit cannot terminate the host.
- Infinite loops and sleeps have timeout or kill behavior.
- Native crashes are contained outside the host process.
- Memory blowups have process-level limits or recovery expectations.
- Import behavior is deterministic across repeated previews.
- Import-cache poisoning cannot make later runs use stale modules silently.
- stdout and stderr are captured and routed to diagnostics.
- Last good state survives current broken edits when that is the correct UX.
- Stale result guards reject results from obsolete code versions.

## Completion Proof

Validation should include at least one broken-code scenario such as syntax error, timeout, crash/exit, stale result, or repeated edit/run cycle.
