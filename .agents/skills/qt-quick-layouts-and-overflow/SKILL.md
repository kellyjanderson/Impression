---
name: qt-quick-layouts-and-overflow
description: Use when building or reviewing Qt Quick or QML interfaces where layout sizing, wrapped text, dialogs, list or grid delegates, ScrollView, Popup, or clipping and overflow behavior matter.
---

# Qt Quick Layouts And Overflow

Use this skill for Qt Quick and Qt Quick Controls work where content size and container behavior are part of the problem.

## Core Rule

In Qt Quick, overflow bugs are usually layout-contract bugs.

Decide the overflow policy for each surface, then implement it with the layout system instead of ad hoc geometry.

## Pick One Overflow Policy Per Surface

* dense browser or tabular row: fixed row or cell size, no wrapping, explicit elide
* detail panel: wrapped text, container grows with content
* bounded editor or modal: wrapped text, container scrolls
* thumbnail grid: fixed cell geometry, clipped visuals only when text labels remain readable

Do not leave overflow to default behavior.

## Qt Quick Rules

1. If an item is inside a `RowLayout`, `ColumnLayout`, or `GridLayout`, let the layout manage geometry.
2. Use `Layout.fillWidth`, `Layout.fillHeight`, `Layout.preferredWidth`, `Layout.preferredHeight`, and constraints instead of mixing child `anchors`, `width`, or `height`.
3. Wrapped `Text` or `Label` needs an explicit width.
4. If text wraps, the parent must either grow vertically or live inside a scroll surface.
5. If a surface must stay fixed-height, use line limits and elision instead of silent clipping.
6. Do not use raw clipping as the default answer for meaningful text or controls.

## Surface Patterns

### Dense file browsers and result browsers

* keep delegates structurally stable
* prefer one or two lines max for labels
* use elide for filenames and metadata
* avoid wrapped text that changes delegate height unless variable rows are intentional

### Dialogs and popups

* size from content intentionally
* use popup content sizing and margins deliberately
* if content can exceed the available space, put the body in a `ScrollView`
* for vertical-only scrolling, bind `contentWidth` to `availableWidth`

### Lists and grids

* `GridView` assumes fixed cell geometry
* `ListView` variable delegate heights can destabilize content-size estimation
* if variable-height delegates are necessary, treat that as an explicit design choice and verify scroll behavior carefully

## Review Checklist

* Does every surface have an explicit overflow policy?
* Can any wrapped label push controls outside its parent?
* Can any dialog cut off actions or fields instead of growing or scrolling?
* Do list and grid delegates remain legible with long filenames and localized strings?
* Are there any children inside layouts that still set conflicting geometry directly?

## References

* Read `references/qt-quick-overflow-rules.md` for the Qt-specific implementation notes and source links.
* Pair this with `../ui-ux-invariants/SKILL.md` for the general invariant rules.
