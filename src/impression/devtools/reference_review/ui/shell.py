"""QML launcher/bootstrap for the Reference Review Workbench."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bridge import BridgeRecord, BridgeRegistry
from .packaging import qml_resource_root

_ACTIVE_LAUNCH: "WorkbenchLaunchResult | None" = None
_USAGE = """usage: impression-reference-review [--check] [--offscreen]

Launch the Impression Reference Review Workbench.

options:
  --check       validate that the QML shell can load, then exit
  --offscreen   use Qt's offscreen platform plugin
  -h, --help    show this help message and exit
"""


@dataclass(frozen=True)
class WorkbenchLaunchResult:
    launched: bool
    diagnostics: tuple[str, ...] = ()
    engine: object | None = None


def _ensure_qt_app(argv: Sequence[str], *, offscreen: bool) -> object:
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is not None:
        return app
    return QGuiApplication(list(argv))


def launch_workbench(
    argv: Sequence[str] | None = None,
    *,
    bridges: BridgeRegistry | None = None,
    qml_path: Path | None = None,
    offscreen: bool = False,
) -> WorkbenchLaunchResult:
    argv = argv or ("impression-reference-review",)
    bridges = bridges or BridgeRegistry()
    diagnostics = [item.code for item in bridges.diagnostics()]
    try:
        _ensure_qt_app(argv, offscreen=offscreen)
        from PySide6.QtCore import QObject, QUrl
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuickControls2 import QQuickStyle
    except Exception as exc:
        return WorkbenchLaunchResult(False, (f"qt-unavailable:{exc}",))

    QQuickStyle.setStyle("Basic")
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    for name, record in bridges.records.items():
        context.setContextProperty(name, record.bridge)
    context.setContextProperty("startupDiagnostics", diagnostics)
    path = qml_path or (qml_resource_root() / "Main.qml")
    if not path.is_file():
        return WorkbenchLaunchResult(False, (f"missing-qml:{path.name}",), engine)
    engine.load(QUrl.fromLocalFile(str(path)))
    if not engine.rootObjects():
        return WorkbenchLaunchResult(False, ("qml-root-not-loaded",), engine)
    return WorkbenchLaunchResult(True, tuple(diagnostics), engine)


def default_bridge_registry() -> BridgeRegistry:
    from PySide6.QtCore import QObject

    registry = BridgeRegistry()
    for name in ("queueBridge", "selectionBridge", "codexBridge", "notesBridge", "artifactsBridge"):
        registry = registry.register(BridgeRecord(name=name, bridge=QObject()))
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_LAUNCH

    argv = tuple(argv or sys.argv)
    args = argv[1:]
    if any(arg in {"-h", "--help"} for arg in args):
        print(_USAGE.strip())
        return 0
    unknown = tuple(arg for arg in args if arg not in {"--check", "--offscreen"})
    if unknown:
        print(f"unknown argument: {unknown[0]}", file=sys.stderr)
        print(_USAGE.strip(), file=sys.stderr)
        return 2
    result = launch_workbench(
        argv,
        bridges=default_bridge_registry(),
        offscreen="--offscreen" in args,
    )
    if not result.launched:
        for diagnostic in result.diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    _ACTIVE_LAUNCH = result
    if "--check" in args:
        print("Reference Review Workbench launch check passed")
        return 0
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
