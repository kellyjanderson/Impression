# Qt Quick Overflow Rules

This file captures the Qt-specific guidance behind the skill.

## Official Qt Rules That Matter

* Qt Quick Layouts resize their children and are intended for resizable interfaces.
* Inside a layout, children should not also set geometry-driving properties like `anchors`, `width`, or `height`.
* Layout preferred size comes first from `Layout.preferredWidth` or `Layout.preferredHeight`, then from `implicitWidth` or `implicitHeight`.
* `Layout.fillWidth` and `Layout.fillHeight` let items grow or shrink between their minimum and maximum constraints.
* `Text.wrapMode` only works when the text item has an explicit width.
* `Text.elide` also depends on explicit width.
* `Text.clip` simply chops text and should not be the default answer for meaningful content.
* `ScrollView` automatically sizes itself from one child item with an implicit size, but with multiple children you must set `contentWidth` and `contentHeight`.
* For vertical-only scrolling in `ScrollView`, binding `contentWidth` to `availableWidth` makes wrapped content use the real inner width.
* `Popup` exposes content sizing through `contentWidth`, `contentHeight`, `implicitContentWidth`, `implicitContentHeight`, plus padding and margins.
* `GridView` uses fixed `cellWidth` and `cellHeight`.
* `ListView` warns that variable delegate sizes can destabilize content-size estimation and scrollbar behavior.

## Practical Policy Mapping

### Fixed browser rows and grids

Use:

* fixed delegate height or fixed grid cell size
* short metadata lines
* `elide`
* no wrapping unless the delegate is intentionally variable-height

### Inspectors and detail panes

Use:

* wrapped labels
* `Layout.fillWidth`
* containers that can grow vertically

### Dialogs and popups

Use:

* content-driven sizing
* explicit margins
* `ScrollView` when content can exceed the available window

## Sources

* Qt Quick Layouts Overview: https://doc.qt.io/qt-6/qtquicklayouts-overview.html
* Layout QML Type: https://doc.qt.io/qt-6/qml-qtquick-layouts-layout.html
* Text QML Type: https://doc.qt.io/qt-6/qml-qtquick-text.html
* ScrollView QML Type: https://doc.qt.io/qt-6/qml-qtquick-controls-scrollview.html
* Popup QML Type: https://doc.qt.io/qt-6/qml-qtquick-controls-popup.html
* ListView QML Type: https://doc.qt.io/qt-6/qml-qtquick-listview.html
* GridView QML Type: https://doc.qt.io/qt-6/qml-qtquick-gridview.html
