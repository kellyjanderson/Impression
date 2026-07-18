# Reference Review Preview Display Controls Product Definition

## Overview

This supplemental product definition describes the display-control button bar
for the Reference Review Workbench preview pane.

The preview pane is a diagnostic review surface. Its controls should help a
reviewer intentionally make geometry easier or harder to inspect so rendering
and modeling issues are exposed rather than hidden. Triangle wireframe display
is therefore a first-class diagnostic mode even though object-edge display
remains the default review mode.

## Related Documents

- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Preview Engine Sharing Architecture](reference-review-preview-engine-sharing-architecture.md)
- [Reference Review Preview Qt Wrapper Architecture](reference-review-preview-qt-wrapper-architecture.md)
- [Reference Review Preview Payload Boundary Architecture](reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Preview Remediation Plan](reference-review-preview-remediation-plan.md)

## Placement

The preview pane should remove the large visible title text currently shown as
`Selected Fixture` above the preview area.

In its place, the top edge of the preview pane should contain a compact
horizontal display-control button bar. The bar sits above the preview surface
inside the preview region and belongs visually to the preview, not to the
fixture context tabs.

Layout rules:

- The bar is one row high.
- The bar is left aligned inside the preview pane.
- Buttons are icon-only controls with hover tooltips and accessible names.
- Related mutually exclusive controls are grouped together.
- A vertical separator `|` visually separates each exclusive group from the
  following controls.
- The bar must not resize the preview surface when button states change.
- If the pane becomes too narrow, the bar may horizontally scroll or collapse
  lower-priority toggles into an overflow menu; it must not wrap into multiple
  rows over the preview.

## Control Types

The bar uses two behavior types.

### Toggle Button

A toggle button independently turns one rendering feature on or off.

Visual states:

- Off: neutral icon, transparent or low-emphasis button background.
- Hover: button background lifts one emphasis step and tooltip appears.
- On: persistent selected background, selected border or accent underline, and
  icon color changes to the active accent.
- Disabled: muted icon, no hover activation, tooltip explains why unavailable.

Selecting a toggle does not change any other toggle unless a later
implementation explicitly defines dependency behavior.

### Exclusive Option Group

An exclusive option group behaves like a radio-button list rendered as adjacent
icon buttons. Exactly one option in the group is selected when preview content
is available.

Visual states:

- Selected option: same selected treatment as an active toggle.
- Unselected options: neutral icon controls.
- Hover: hovered option shows preview/tooltip but does not change selection
  until clicked.
- Disabled group: all options disabled together when no preview is loaded.

Selecting one option immediately deselects the previously selected option in
that group.

## Initial Button Set

The first version should expose all currently planned display controls, grouped
only where the choices are mutually exclusive.

### Color Mode Group

Purpose:
- Choose whether the preview uses authored object colors from the payload or
  the workbench's selected/default inspection color.

Options:

| Option | Type | Default | Behavior |
| --- | --- | --- | --- |
| Authored Colors | exclusive | off | Uses all authored mesh colors, including face colors when payloads provide them. |
| Inspection Color | exclusive | on | Uses the workbench selected/default object color for all mesh faces. |

State rules:

- The group is disabled when no preview payload is loaded.
- If authored colors are selected but the payload has no authored color data,
  the option remains selected and falls back visibly to the inspection color;
  the tooltip should mention that no authored colors are present.
- The selected/default inspection color may become user-selectable later; this
  definition only requires the mode switch.

### Lighting Mode Group

Purpose:
- Choose how surface lighting is evaluated.

Options:

| Option | Type | Default | Behavior |
| --- | --- | --- | --- |
| Flat | exclusive | off | Draws faces without lighting variation. This helps inspect raw color and silhouette without shaded cues. |
| Face Normals | exclusive | on | Shades each face from cached per-face normals. This is the current software-preview lighting behavior. |
| Camera Light | exclusive | off | Uses a fixed fill plus point-source light that is fixed relative to the camera, so rotation changes visible highlights predictably. |

State rules:

- The group is disabled when no preview payload is loaded.
- Switching lighting mode must not rebuild mesh topology.
- Lighting changes should repaint the current prepared scene only.

### Independent Toggles

| Control | Type | Default | Behavior |
| --- | --- | --- | --- |
| Object Fill | toggle | on | Shows or hides filled mesh faces. Turning it off creates an edge/diagnostic-only view. |
| Object Edges | toggle | on | Shows or hides the complete rendered object wireframe. |
| Triangle Wireframe | toggle | off | Shows or hides the explicit triangle-wireframe diagnostic overlay. This is intentionally available because triangle seams can expose rendering and tessellation issues. |
| Bounds Grid | toggle | on | Shows or hides the floor/bounds grid. |
| Axis Triad | toggle | on | Shows or hides the RGB X/Y/Z axis triad. |
| Gradient Background | toggle | on | Switches between the preview's dark gradient background and a flat dark background. |
| Polylines | toggle | on | Shows or hides path, curve, or section polyline overlays included in the preview payload. |

State rules:

- Toggles are disabled when no preview payload is loaded.
- Toggling a control updates only the visible preview presentation. It must not
  reload the fixture, rebuild the preview payload, rerun tessellation, or
  recreate the render surface.
- `Object Fill` off and both edge toggles off is allowed; the result may show
  only grid, axes, and polylines.
- `Object Edges` and `Triangle Wireframe` may both be on. This is useful when
  comparing model feature edges against tessellation seams.

## Proposed Icon Set

Icons should be selected from the app's existing icon source when available.
If the implementation uses a library such as Lucide, prefer the named library
icon. If a named icon is unavailable, create a simple line icon matching the
same metaphor.

| Control | Proposed Icon | Tooltip |
| --- | --- | --- |
| Authored Colors | palette | Use authored object and face colors |
| Inspection Color | swatch-book or square | Use inspection color |
| Flat Lighting | square | Flat color lighting |
| Face Normals Lighting | facets or box | Shade by face normals |
| Camera Light | flashlight or sun | Camera-fixed fill and point light |
| Object Fill | box or cube | Show filled faces |
| Object Edges | scan-line or route | Show object feature edges |
| Triangle Wireframe | triangle | Show triangle wireframe |
| Bounds Grid | grid-3x3 | Show bounds grid |
| Axis Triad | move-3d or axes-3d | Show axis triad |
| Gradient Background | panel-top or gradient | Use gradient background |
| Polylines | spline or route | Show curve and path overlays |

Icon guidance:

- Use icon-only buttons in the row.
- Every icon button must have a tooltip and accessible name matching the
  command's reviewer-facing meaning.
- Avoid text labels in the row unless an overflow menu is introduced.
- Keep icons visually distinct at small sizes. In particular, `Object Edges`
  and `Triangle Wireframe` must not look interchangeable.
- The selected state must be visible without relying only on color.

Generated asset path:
- `src/impression/devtools/reference_review/ui/qml/icons/preview-display/`

Generated icon files:

| Control | Asset |
| --- | --- |
| Authored Colors | `authored-colors.svg` |
| Inspection Color | `inspection-color.svg` |
| Flat Lighting | `lighting-flat.svg` |
| Face Normals Lighting | `lighting-face-normals.svg` |
| Camera Light | `lighting-camera.svg` |
| Object Fill | `object-fill.svg` |
| Object Edges | `object-edges.svg` |
| Triangle Wireframe | `triangle-wireframe.svg` |
| Bounds Grid | `bounds-grid.svg` |
| Axis Triad | `axis-triad.svg` |
| Gradient Background | `gradient-background.svg` |
| Polylines | `polylines.svg` |

## Default State

When a preview first becomes ready:

- Color mode: `Inspection Color`
- Lighting mode: `Face Normals`
- Object Fill: on
- Object Edges: on
- Triangle Wireframe: off
- Bounds Grid: on
- Axis Triad: on
- Gradient Background: on
- Polylines: on

When the preview is empty, loading, or failed:

- The button bar remains visible for spatial stability.
- All controls are disabled.
- Tooltips explain the current preview state, such as `Load a fixture to change display options`.

## Visual Effect Summary

The button bar should make display changes immediately obvious:

- Switching color mode changes mesh color source without camera movement.
- Switching lighting mode changes face shading without changing geometry,
  colors, grid, axes, or camera.
- Toggling object fill removes or restores filled polygons.
- Toggling object edges removes or restores the complete rendered object wireframe.
- Toggling triangle wireframe overlays or removes all triangle seams.
- Toggling grid, axes, background, and polylines affects only those layers.

Layer order:

1. Background
2. Bounds grid
3. Filled object faces
4. Object edges
5. Triangle wireframe
6. Polylines
7. Axis triad

## Acceptance Criteria

- The `Selected Fixture` preview-pane title is removed.
- A one-row icon button bar appears above the preview surface.
- Color mode and lighting mode behave as exclusive option groups.
- Independent rendering features behave as toggles.
- Separators visually divide exclusive groups from the rest of the controls.
- Hover tooltips describe the button behavior.
- Selected/active states are visible and do not resize controls.
- Disabled states are used when no preview payload is ready.
- Display changes do not rebuild payloads, reload fixtures, recreate renderers,
  or recompute mesh topology.

## Specification Review History

2026-07-08 five-pass review:

- Pass 1: The initial four-spec split still bundled too much behavior. The
  reusable icon/control and button-bar work was split into icon asset
  packaging, icon metadata, icon toggle visual state, icon toggle interaction,
  exclusive group selection, exclusive group composition, display option state,
  display command routing, renderer option application, and bar layout leaves.
- Pass 2: Renderer option application still bundled layer visibility with
  color and lighting behavior. It was split into layer visibility and
  color/lighting application leaves.
- Pass 3: The final bar layout leaf still bundled preview-pane chrome with
  control-row composition. It was split into preview-pane display-control slot
  and preview display-control row composition leaves.
- Pass 4: Specs 63-74 and paired test specs were reviewed for score, backlink,
  progression, and file-name consistency. No remaining implementation leaf
  scored at or above the split threshold.
- Pass 5: Final review confirmed all preview display-control specs are final
  leaves, all have paired test specifications, and all are listed unchecked in
  the progression document.
