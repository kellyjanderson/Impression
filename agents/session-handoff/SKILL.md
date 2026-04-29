---
name: session-handoff
description: Use the Impression-specific end-of-work handoff phrase only when the current work pass is complete and continuing would require more than a simple continue.
---

# Session Handoff

This Skill defines the project-specific end-of-work handoff phrase.

## Rule

When the current workable tranche is complete and continuing would require more than a simple `continue`, print the exact phrase:

```text
*****WORK IS DONE****
```

## Do Not Use It

Do not print the phrase when:

* you are only pausing within an active implementation sequence
* a simple `continue` is enough to proceed naturally
* there is still a clear next bounded task under the current request

This is a chat-level handoff rule, not a code or commit marker.
