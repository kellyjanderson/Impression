---
name: coverage-discipline
description: Run focused and full coverage at the required times in Impression and refresh durable XML and HTML coverage artifacts under project coverage paths.
---

# Coverage Discipline

Coverage is part of finishing the change.

## Required Coverage Pass

When an implementation pass finishes, run coverage for at least the code that changed.

At minimum:

* run a focused coverage command for the touched subsystem when one exists
* refresh XML and HTML artifacts under `project/coverage/`

## Full Coverage Trigger

When the final remaining item in `project/planning/progression.md` is checked off, also run a full repository coverage pass and refresh the top-level coverage artifacts.

## Artifact Rule

Coverage output should produce more than terminal text.

Keep durable XML and HTML artifacts fresh enough that humans and tooling can inspect them without rediscovering paths or rerunning coverage blindly.
