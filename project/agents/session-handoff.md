# Session Handoff

## Purpose

This document defines the project-specific end-of-work handoff phrase for long-running Codex sessions.

## Rule

When the agent has finished the current workable tranche and reasonably believes that continuing would require more than a simple `"continue"` from the user, it should print the exact phrase below in chat:

```text

*****WORK IS DONE****

```

## Meaning

That phrase means:

- the current bounded work pass is complete
- the repository is in a reviewed stopping state for this pass
- resuming usefully would likely require a new direction, a new task choice, or an explicit re-engagement beyond a casual continuation

## Non-Use Cases

The agent should **not** print the phrase when:

- it is merely pausing within an active implementation sequence
- a simple `"continue"` is enough to proceed naturally
- it still has a clear next bounded task that fits the current user direction

## Notes

- This is a chat-level handoff rule, not a code or commit marker.
- If the agent can keep making coherent forward progress under the current request, it should continue working instead of emitting the handoff phrase.
