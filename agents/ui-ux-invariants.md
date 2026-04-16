# UI/UX Invariants

This document defines mandatory structural rules for user-interface work.

These are not aesthetic suggestions.

A user interface is a spatial promise.

Do not break that promise across states.

## UI/UX Invariants and Standards

### 1. Preserve Layout Stability

- Controls must not jump, shift, collapse, or reorder between loading, empty, partial, error, and populated states unless the change is explicitly required by direct user action.
- If content may appear later, reserve its space in advance.
- Missing data must not cause surrounding layout to collapse.
- Prefer placeholders, skeletons, disabled values, or empty-state content inside a stable container.
- Avoid cumulative layout shift at all times.

### 2. Preserve Spatial Memory

- Keep controls in consistent positions across states.
- Do not move primary actions because secondary content is absent.
- Do not insert content above existing controls in a way that pushes them downward unless absolutely necessary.
- A user should be able to build muscle memory for where things are.

### 3. Design For All Core States

Every meaningful UI region must be intentionally designed for:

- loading
- empty
- partial data
- full data
- error
- disabled
- overflow or long content
- narrow viewport or responsive layout

Never implement only the happy path.

### 4. Keep Containers Stable

- Do not remove a panel, card, section, toolbar, or control group merely because its current content is unavailable.
- The structure should remain stable while the contents change.
- Swap content inside the frame rather than removing the frame.

### 5. Prefer Explicit Empty States Over Disappearance

- If information is unavailable, show an intentional empty state.
- Empty states should explain what is missing, why, and what the user can do next if action is possible.
- Do not silently hide missing sections if the user would reasonably expect them to exist.

### 6. Preserve Action Hierarchy

- Primary actions must remain visually and spatially consistent.
- Secondary actions must not visually overpower primary actions.
- Destructive actions must be clearly differentiated.
- Do not change the meaning, label, or placement of a primary action across states unless required by the workflow.

### 7. Clear And Stable Control Semantics

- Every interactive region must have a clear, consistent, and discoverable purpose.
- A single visual control may contain multiple interaction zones if:
  - each zone has a distinct and stable meaning
  - the zones are visually or spatially distinguishable
  - the mapping between interaction and outcome does not change implicitly over time
- Do not overload a single interaction region with multiple unrelated meanings.
- Do not change the behavior of a control based on hidden modes or implicit state.
- If a control changes behavior, that change must be clearly signaled in the UI.
- Composite controls such as segmented buttons, multi-zone buttons, and rocker switches are allowed and encouraged when:
  - they mirror real-world interaction patterns
  - they reduce UI clutter without reducing clarity
  - they preserve spatial consistency and user expectation
- The user must be able to predict what will happen before interacting, based on visible structure alone.

A control may be complex.

Its meaning may not be.

### 8. Use Standard Interaction Patterns

- Prefer established UI conventions over novelty.
- Buttons should look clickable.
- Inputs should look editable.
- Disabled controls should look inactive but still legible.
- Links should look like links.
- Avoid cleverness that reduces clarity.

### 9. Minimize Cognitive Load

- Reduce the number of decisions required on first view.
- Group related controls together.
- Keep visual hierarchy obvious.
- Avoid unnecessary fragmentation across too many cards, boxes, or nested panels.
- Do not require the user to remember information from one area in order to use another if it can be shown directly.

### 10. Make State Visible

The user should always be able to tell:

- what is happening
- what is selectable
- what is disabled
- what is loading
- what succeeded
- what failed

System status should be visible without forcing the user to infer it.

### 11. Respect Reading And Scanning Flow

- Important information should appear where the eye naturally lands first.
- Supporting detail should not interrupt primary tasks.
- Do not create ragged or chaotic scanning paths with inconsistent spacing, alignment, or hierarchy.
- Align labels, values, and controls consistently.

### 12. Use Consistent Spacing And Sizing

- Spacing must come from a small, repeated scale.
- Similar components must use similar padding, gaps, border radius, and typography.
- Do not let content presence alter component padding or outer dimensions in a way that causes jitter.

### 13. Prevent Accidental Interaction Errors

- Do not place destructive actions adjacent to frequent safe actions without separation.
- Use confirmation only for meaningful risk, not as a crutch for bad layout.
- Touch and click targets must be large enough to hit reliably.
- Do not make users aim precisely at tiny controls.

### 14. Accessibility Is Required

- Maintain sufficient contrast.
- Support keyboard navigation and visible focus states.
- Do not rely on color alone to communicate meaning.
- Use semantic structure and proper labels.
- Screen-reader-relevant names and roles must exist for interactive elements.

### 15. Responsive Behavior Must Preserve Meaning

- Responsive design must not merely fit; it must remain understandable.
- When adapting to smaller screens, preserve hierarchy, action clarity, and state visibility.
- Do not hide critical actions or information without a strong reason.
- Stacking is preferable to shrinking controls into unusability.

### 16. Handle Long And Real-World Content

- Design for long names, long numbers, long paragraphs, and unexpected user content.
- Prevent overflow from breaking layouts.
- Use truncation only when the full content can still be accessed.
- Never assume ideal data length.

### 17. Feedback Must Be Timely And Local

- User actions should produce visible feedback near the point of interaction.
- Loading indicators should appear where the result is expected.
- Success and error messages should be specific and actionable.
- Do not make users guess whether a click worked.

### 18. Avoid False Polish

- Removing "empty" space at the cost of stability is a failure.
- Dynamic motion that harms clarity is a failure.
- Minimalism that hides structure, labels, or affordance is a failure.
- Clean-looking UI that behaves inconsistently is bad UI.

### 19. Favor Resilience Over Cleverness

- Build interfaces that remain understandable when data is delayed, missing, malformed, or excessive.
- The UI should degrade gracefully under imperfect conditions.
- Good UI is robust before it is stylish.

### 20. Before Finalizing Any UI, Check These Invariants

The agent must verify:

- Do controls stay in the same place across states?
- Does missing content preserve layout instead of collapsing it?
- Are loading, empty, error, and full states all intentionally designed?
- Is the primary action always obvious?
- Can the interface be understood at a glance?
- Does responsive behavior preserve hierarchy?
- Are accessibility basics present?
- Would this interface still make sense with ugly real-world data?

If any answer is `no`, the UI is not finished.

### 21. Controls Must Honor Their Local Promise

- If a control is presented as available, invoking it should usually perform the expected action immediately.
- Do not expose controls that appear usable but fail because of ordinary background state the system could handle automatically.
- If an action has prerequisites, satisfy them automatically when reasonable.
- Do not make the user manually perform setup or cleanup steps the system can safely perform on their behalf.
- A control in context should behave as that context implies.

### 22. Resolve Routine State Conflicts Automatically

- When a user-triggered action conflicts with an existing reversible state, prefer automatic resolution over rejection.
- Pause, stop, suspend, switch focus, or replace the previous transient action when that is the obvious user intent.
- Prefer seamless handoff over error messaging for ordinary interaction conflicts.
- Use errors only when the system cannot safely infer intent or automatic handling would risk data loss, destructive change, or serious confusion.

Examples:

- A preview button in a picker should start previewing.
- A retry button should retry.
- A pause button should pause the active process.
- A voice preview control should not require the user to separately stop unrelated playback first if the system can pause it automatically.

## UI Implementation Rules

When generating UI code:

- Never conditionally remove structural containers solely because data is absent.
- Use fixed-height or minimum-height regions where later content would otherwise cause layout shift.
- Prefer in-place skeletons, placeholders, and empty states over conditional disappearance.
- Keep button rows, headers, control bars, and navigation regions in fixed positions.
- Design all major components as stateful shells with interchangeable internal content.
- Do not optimize for visual compactness at the expense of stability.
