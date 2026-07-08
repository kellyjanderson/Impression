"""Reusable preview display controls for the Reference Review workbench."""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QWidget

from .packaging import (
    PreviewDisplayControlIconRecord,
    preview_display_control_icon_record,
    qml_resource_root,
)

COLOR_MODE_AUTHORED = "authored"
COLOR_MODE_INSPECTION = "inspection"
LIGHTING_MODE_FLAT = "flat"
LIGHTING_MODE_FACE_NORMALS = "face_normals"
LIGHTING_MODE_CAMERA = "camera"

COLOR_MODE_COMMANDS = {
    "authored-colors": COLOR_MODE_AUTHORED,
    "inspection-color": COLOR_MODE_INSPECTION,
}
LIGHTING_MODE_COMMANDS = {
    "lighting-flat": LIGHTING_MODE_FLAT,
    "lighting-face-normals": LIGHTING_MODE_FACE_NORMALS,
    "lighting-camera": LIGHTING_MODE_CAMERA,
}
TOGGLE_COMMAND_FIELDS = {
    "object-fill": "show_object_fill",
    "object-edges": "show_object_edges",
    "triangle-wireframe": "show_triangle_wireframe",
    "bounds-grid": "show_bounds_grid",
    "axis-triad": "show_axis_triad",
    "gradient-background": "show_gradient_background",
    "polylines": "show_polylines",
}


@dataclass(frozen=True)
class IconToggleCommandRecord:
    command: str
    enabled: bool
    checked: bool


@dataclass(frozen=True)
class ExclusiveIconOptionRecord:
    id: str
    icon_id: str
    command: str


@dataclass(frozen=True)
class ExclusiveIconGroupState:
    options: tuple[ExclusiveIconOptionRecord, ...]
    selected_id: str
    enabled: bool = True

    def __post_init__(self) -> None:
        option_ids = tuple(option.id for option in self.options)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("duplicate-exclusive-icon-option")
        if self.options and self.selected_id not in option_ids:
            raise ValueError(f"unknown-exclusive-icon-option:{self.selected_id}")


@dataclass(frozen=True)
class PreviewDisplayOptions:
    color_mode: str = COLOR_MODE_INSPECTION
    lighting_mode: str = LIGHTING_MODE_FACE_NORMALS
    show_object_fill: bool = True
    show_object_edges: bool = True
    show_triangle_wireframe: bool = False
    show_bounds_grid: bool = True
    show_axis_triad: bool = True
    show_gradient_background: bool = True
    show_polylines: bool = True

    def __post_init__(self) -> None:
        if self.color_mode not in {COLOR_MODE_AUTHORED, COLOR_MODE_INSPECTION}:
            raise ValueError(f"unsupported-preview-color-mode:{self.color_mode}")
        if self.lighting_mode not in {
            LIGHTING_MODE_FLAT,
            LIGHTING_MODE_FACE_NORMALS,
            LIGHTING_MODE_CAMERA,
        }:
            raise ValueError(f"unsupported-preview-lighting-mode:{self.lighting_mode}")

    def updated(self, **changes: object) -> "PreviewDisplayOptions":
        return replace(self, **changes)


@dataclass(frozen=True)
class PreviewDisplayCommandRecord:
    command: str
    executed: bool
    options: PreviewDisplayOptions
    diagnostic: str | None = None


def select_exclusive_icon_option(
    state: ExclusiveIconGroupState,
    option_id: str,
) -> ExclusiveIconGroupState:
    if not state.enabled:
        return state
    if option_id not in {option.id for option in state.options}:
        raise ValueError(f"unknown-exclusive-icon-option:{option_id}")
    return replace(state, selected_id=option_id)


def route_preview_display_command(
    options: PreviewDisplayOptions,
    command: str,
    *,
    ready: bool,
) -> PreviewDisplayCommandRecord:
    if not ready:
        return PreviewDisplayCommandRecord(
            command,
            False,
            options,
            "preview-display-controls-disabled",
        )
    if command in COLOR_MODE_COMMANDS:
        return PreviewDisplayCommandRecord(
            command,
            True,
            options.updated(color_mode=COLOR_MODE_COMMANDS[command]),
        )
    if command in LIGHTING_MODE_COMMANDS:
        return PreviewDisplayCommandRecord(
            command,
            True,
            options.updated(lighting_mode=LIGHTING_MODE_COMMANDS[command]),
        )
    if command in TOGGLE_COMMAND_FIELDS:
        field = TOGGLE_COMMAND_FIELDS[command]
        return PreviewDisplayCommandRecord(command, True, options.updated(**{field: not getattr(options, field)}))
    return PreviewDisplayCommandRecord(
        command,
        False,
        options,
        "unsupported-preview-display-command",
    )


class WorkbenchIconToggleButton(QToolButton):
    commandTriggered = Signal(object)

    def __init__(
        self,
        icon: PreviewDisplayControlIconRecord,
        *,
        command: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.command = command or icon.id
        self.setObjectName(f"previewDisplayControl-{icon.id}")
        self.setCheckable(True)
        self.setFixedSize(30, 30)
        self.setIconSize(QSize(18, 18))
        self.setToolTip(icon.tooltip)
        self.setAccessibleName(icon.accessible_name)
        self.setText("")
        self.setIcon(QIcon(str(qml_resource_root() / icon.resource_path)))
        self.setFocusPolicy(self.focusPolicy())
        self.clicked.connect(self._emit_command)
        self.setStyleSheet(
            """
            QToolButton {
                background: #1b2430;
                border: 1px solid #65788e;
                border-radius: 5px;
                color: #dbe8f5;
                padding: 0;
            }
            QToolButton:hover {
                background: #29384a;
                border-color: #9bb3cc;
            }
            QToolButton:pressed {
                background: #27364a;
            }
            QToolButton:checked {
                background: #23425c;
                border: 2px solid #9fd3ff;
                color: #ffffff;
            }
            QToolButton:focus {
                border: 2px solid #f2c14e;
            }
            QToolButton:disabled {
                background: #161b22;
                border-color: #3a4654;
                color: #87919e;
            }
            """
        )

    def command_record(self) -> IconToggleCommandRecord:
        return IconToggleCommandRecord(self.command, self.isEnabled(), self.isChecked())

    def _emit_command(self) -> None:
        if self.isEnabled():
            self.commandTriggered.emit(self.command_record())


class ExclusiveIconOptionGroup(QWidget):
    optionSelected = Signal(str)

    def __init__(
        self,
        state: ExclusiveIconGroupState,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._buttons: dict[str, WorkbenchIconToggleButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for option in state.options:
            button = WorkbenchIconToggleButton(
                preview_display_control_icon_record(option.icon_id),
                command=option.command,
                parent=self,
            )
            button.commandTriggered.connect(
                lambda _record, selected_id=option.id: self.select_option(selected_id)
            )
            self._buttons[option.id] = button
            layout.addWidget(button)
        self.set_state(state)

    @property
    def state(self) -> ExclusiveIconGroupState:
        return self._state

    def option_ids(self) -> tuple[str, ...]:
        return tuple(self._buttons.keys())

    def set_state(self, state: ExclusiveIconGroupState) -> None:
        self._state = state
        self.setEnabled(state.enabled)
        for option_id, button in self._buttons.items():
            button.blockSignals(True)
            button.setEnabled(state.enabled)
            button.setChecked(option_id == state.selected_id)
            button.blockSignals(False)

    def select_option(self, option_id: str) -> None:
        previous = self._state.selected_id
        self.set_state(select_exclusive_icon_option(self._state, option_id))
        if self._state.enabled and self._state.selected_id != previous:
            self.optionSelected.emit(self._state.selected_id)


class PreviewDisplayControlBar(QWidget):
    commandTriggered = Signal(str)

    def __init__(
        self,
        *,
        options: PreviewDisplayOptions | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("previewDisplayControlBar")
        self._options = options or PreviewDisplayOptions()
        self._buttons: dict[str, WorkbenchIconToggleButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.color_group = ExclusiveIconOptionGroup(
            _color_group_state(self._options),
            parent=self,
        )
        self.color_group.setObjectName("previewDisplayColorGroup")
        self.color_group.optionSelected.connect(self._emit_color_command)
        layout.addWidget(self.color_group)
        layout.addWidget(_separator(self))
        self.lighting_group = ExclusiveIconOptionGroup(
            _lighting_group_state(self._options),
            parent=self,
        )
        self.lighting_group.setObjectName("previewDisplayLightingGroup")
        self.lighting_group.optionSelected.connect(self._emit_lighting_command)
        layout.addWidget(self.lighting_group)
        layout.addWidget(_separator(self))
        for command in TOGGLE_COMMAND_FIELDS:
            button = WorkbenchIconToggleButton(preview_display_control_icon_record(command), parent=self)
            button.commandTriggered.connect(lambda record: self.commandTriggered.emit(record.command))
            self._buttons[command] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self.set_options(self._options)

    def control_ids(self) -> tuple[str, ...]:
        return (
            "authored-colors",
            "inspection-color",
            "separator",
            "lighting-flat",
            "lighting-face-normals",
            "lighting-camera",
            "separator",
        ) + tuple(TOGGLE_COMMAND_FIELDS)

    def set_options(self, options: PreviewDisplayOptions) -> None:
        self._options = options
        self.color_group.set_state(_color_group_state(options, enabled=self.isEnabled()))
        self.lighting_group.set_state(_lighting_group_state(options, enabled=self.isEnabled()))
        for command, field in TOGGLE_COMMAND_FIELDS.items():
            self._buttons[command].blockSignals(True)
            self._buttons[command].setChecked(bool(getattr(options, field)))
            self._buttons[command].blockSignals(False)

    def set_ready(self, ready: bool) -> None:
        self.setEnabled(ready)
        self.color_group.set_state(_color_group_state(self._options, enabled=ready))
        self.lighting_group.set_state(_lighting_group_state(self._options, enabled=ready))
        for button in self._buttons.values():
            button.setEnabled(ready)

    def _emit_color_command(self, option_id: str) -> None:
        self.commandTriggered.emit("authored-colors" if option_id == COLOR_MODE_AUTHORED else "inspection-color")

    def _emit_lighting_command(self, option_id: str) -> None:
        command = {
            LIGHTING_MODE_FLAT: "lighting-flat",
            LIGHTING_MODE_FACE_NORMALS: "lighting-face-normals",
            LIGHTING_MODE_CAMERA: "lighting-camera",
        }[option_id]
        self.commandTriggered.emit(command)


def _color_group_state(
    options: PreviewDisplayOptions,
    *,
    enabled: bool = True,
) -> ExclusiveIconGroupState:
    return ExclusiveIconGroupState(
        (
            ExclusiveIconOptionRecord(COLOR_MODE_AUTHORED, "authored-colors", "authored-colors"),
            ExclusiveIconOptionRecord(COLOR_MODE_INSPECTION, "inspection-color", "inspection-color"),
        ),
        options.color_mode,
        enabled,
    )


def _lighting_group_state(
    options: PreviewDisplayOptions,
    *,
    enabled: bool = True,
) -> ExclusiveIconGroupState:
    return ExclusiveIconGroupState(
        (
            ExclusiveIconOptionRecord(LIGHTING_MODE_FLAT, "lighting-flat", "lighting-flat"),
            ExclusiveIconOptionRecord(
                LIGHTING_MODE_FACE_NORMALS,
                "lighting-face-normals",
                "lighting-face-normals",
            ),
            ExclusiveIconOptionRecord(LIGHTING_MODE_CAMERA, "lighting-camera", "lighting-camera"),
        ),
        options.lighting_mode,
        enabled,
    )


def _separator(parent: QWidget) -> QFrame:
    separator = QFrame(parent)
    separator.setObjectName("previewDisplayControlSeparator")
    separator.setFrameShape(QFrame.Shape.VLine)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setFixedWidth(9)
    separator.setStyleSheet("color: #3d4654;")
    return separator
