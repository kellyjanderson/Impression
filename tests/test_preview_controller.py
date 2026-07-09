from __future__ import annotations

import numpy as np
from pathlib import Path

from impression._config import UnitSettings
from impression.mesh import Mesh, Polyline
from impression.preview import (
    PreviewControllerOptions,
    PreviewInteractionPolicy,
    PreviewSceneApplyOptions,
    PreviewSceneController,
    PreviewStyle,
    PyVistaPreviewer,
)
from impression.preview_qt import (
    QtPreviewSurfaceConfig,
    apply_qt_preview_scene,
    configure_qt_preview_surface_format,
    configure_qvtk_backend,
    preview_scene_options_for_camera_state,
    qt_preview_supported_environment,
)


class FakePlotter:
    def __init__(self) -> None:
        self.background_calls: list[tuple[str, str | None]] = []
        self.eye_dome_calls = 0
        self.eye_dome_disabled_calls = 0
        self.axes_calls: list[dict[str, object]] = []
        self.hide_axes_all_calls = 0
        self.hide_axes_calls = 0
        self.bounds_calls: list[dict[str, object]] = []
        self.remove_bounds_axes_calls = 0
        self.remove_bounding_box_calls = 0
        self.clear_calls = 0
        self.mesh_calls: list[dict[str, object]] = []
        self.actors: list[FakeActor] = []
        self.camera_position = None

    def set_background(self, background: str, *, top: str | None = None) -> None:
        self.background_calls.append((background, top))

    def enable_eye_dome_lighting(self) -> None:
        self.eye_dome_calls += 1

    def disable_eye_dome_lighting(self) -> None:
        self.eye_dome_disabled_calls += 1

    def add_axes(self, **kwargs: object) -> None:
        self.axes_calls.append(dict(kwargs))

    def hide_axes_all(self) -> None:
        self.hide_axes_all_calls += 1

    def hide_axes(self) -> None:
        self.hide_axes_calls += 1

    def show_bounds(self, **kwargs: object) -> None:
        self.bounds_calls.append(dict(kwargs))

    def remove_bounds_axes(self) -> None:
        self.remove_bounds_axes_calls += 1

    def remove_bounding_box(self) -> None:
        self.remove_bounding_box_calls += 1

    def clear(self) -> None:
        self.clear_calls += 1

    def add_mesh(self, mesh: object, **kwargs: object) -> "FakeActor":
        call = dict(kwargs)
        call["mesh"] = mesh
        self.mesh_calls.append(call)
        actor = FakeActor()
        self.actors.append(actor)
        return actor


class FakeActor:
    def __init__(self) -> None:
        self.property = FakeActorProperty()

    def GetProperty(self) -> "FakeActorProperty":
        return self.property


class FakeActorProperty:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def LightingOn(self) -> None:
        self.calls.append("LightingOn")

    def LightingOff(self) -> None:
        self.calls.append("LightingOff")

    def SetInterpolationToFlat(self) -> None:
        self.calls.append("SetInterpolationToFlat")

    def SetInterpolationToPhong(self) -> None:
        self.calls.append("SetInterpolationToPhong")


class FakePyVista:
    def __init__(self) -> None:
        self.polydata_calls: list[tuple[tuple[int, ...], list[int]]] = []

    def PolyData(self, points=None, cells=None, deep: bool = False):
        if points is None:
            points = np.empty((0, 3))
        if cells is None:
            cells = []
        self.polydata_calls.append((tuple(np.asarray(points).shape), list(np.asarray(cells, dtype=int))))
        return {"points": np.asarray(points), "cells": np.asarray(cells), "deep": deep}


class FakeFeatureEdges:
    n_cells = 1


class FakePvMesh:
    def __init__(self) -> None:
        self.edge_angles: list[float] = []

    def extract_feature_edges(self, *, angle: float):
        self.edge_angles.append(angle)
        return FakeFeatureEdges()


class FakeLegacyPvMesh:
    def __init__(self) -> None:
        self.feature_angles: list[float] = []

    def extract_feature_edges(
        self,
        *,
        boundary_edges: bool = False,
        feature_edges: bool = True,
        non_manifold_edges: bool = True,
        manifold_edges: bool = True,
        feature_angle: float = 30.0,
    ):
        self.feature_angles.append(feature_angle)
        return FakeFeatureEdges()


def test_preview_scene_controller_configures_default_plotter_style() -> None:
    plotter = FakePlotter()
    controller = PreviewSceneController(
        unit_settings=UnitSettings(name="millimeters", label="mm", scale_to_mm=1.0)
    )

    diagnostic = controller.configure_plotter(plotter)

    assert diagnostic.background == "#090c10"
    assert diagnostic.background_top == "#1b2333"
    assert diagnostic.show_bounds
    assert diagnostic.show_axes
    assert diagnostic.eye_dome_lighting
    assert plotter.background_calls == [("#090c10", "#1b2333")]
    assert plotter.eye_dome_calls == 1
    assert plotter.axes_calls == [{"interactive": True}]
    assert plotter.bounds_calls == [
        {
            "grid": "front",
            "color": "#5a677d",
            "xtitle": "X (mm)",
            "ytitle": "Y (mm)",
            "ztitle": "Z (mm)",
        }
    ]


def test_preview_scene_controller_supports_workbench_style_defaults() -> None:
    plotter = FakePlotter()
    controller = PreviewSceneController(
        unit_settings=UnitSettings(name="millimeters", label="mm", scale_to_mm=1.0),
        options=PreviewControllerOptions(style=PreviewStyle.workbench_default()),
    )

    diagnostic = controller.configure_plotter(plotter, show_bounds=False, show_axes=False)

    assert diagnostic.background == "#07111f"
    assert diagnostic.background_top is None
    assert diagnostic.show_bounds is False
    assert diagnostic.show_axes is False
    assert controller.style.color_cycle == ("#f4a261",)
    assert plotter.background_calls == [("#07111f", None)]
    assert plotter.eye_dome_calls == 1
    assert plotter.axes_calls == []
    assert plotter.bounds_calls == []


def test_preview_scene_controller_safe_mode_disables_decorations(monkeypatch) -> None:
    monkeypatch.setenv("IMPRESSION_PYVISTA_SAFE", "1")
    plotter = FakePlotter()
    controller = PreviewSceneController(
        unit_settings=UnitSettings(name="millimeters", label="mm", scale_to_mm=1.0)
    )

    diagnostic = controller.configure_plotter(plotter)

    assert diagnostic.safe_mode
    assert diagnostic.eye_dome_lighting is False
    assert diagnostic.show_bounds is False
    assert diagnostic.show_axes is False
    assert plotter.background_calls == [("#090c10", "#1b2333")]
    assert plotter.eye_dome_calls == 0
    assert plotter.axes_calls == []
    assert plotter.bounds_calls == []


def test_preview_scene_controller_honors_interaction_policy_defaults() -> None:
    plotter = FakePlotter()
    controller = PreviewSceneController(
        unit_settings=UnitSettings(name="millimeters", label="mm", scale_to_mm=1.0),
        options=PreviewControllerOptions(
            interaction=PreviewInteractionPolicy(show_bounds=False, show_axes=True)
        ),
    )

    diagnostic = controller.configure_plotter(plotter)

    assert diagnostic.show_bounds is False
    assert diagnostic.show_axes is True
    assert plotter.axes_calls == [{"interactive": True}]
    assert plotter.bounds_calls == []


def test_preview_scene_controller_applies_polyline_scene_without_renderer_creation() -> None:
    fake_pv = FakePyVista()
    plotter = FakePlotter()
    controller = PreviewSceneController(
        unit_settings=UnitSettings(name="millimeters", label="mm", scale_to_mm=1.0),
        pyvista_provider=lambda: fake_pv,
    )
    polyline = Polyline(np.array([[0, 0, 0], [1, 0, 0]]), color=(1.0, 0.5, 0.0, 1.0))

    controller.apply_scene(plotter, [polyline], show_bounds=False, show_axes=False)

    assert plotter.clear_calls == 1
    assert fake_pv.polydata_calls == [((2, 3), [2, 0, 1])]
    assert len(plotter.mesh_calls) == 1
    mesh_call = plotter.mesh_calls[0]
    assert np.array_equal(mesh_call.pop("mesh")["points"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert mesh_call == {
        "name": "mesh-0",
        "color": (1.0, 0.5, 0.0, 1.0),
        "line_width": 2.0,
        "render_lines_as_tubes": False,
    }


def test_preview_scene_controller_applies_mesh_scene_and_feature_edges(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )

    controller.apply_scene(plotter, [mesh], show_edges=True, face_edges=True, show_bounds=False, show_axes=False)

    assert plotter.clear_calls == 1
    assert pv_mesh.edge_angles == [60.0]
    assert plotter.mesh_calls[0] == {
        "mesh": pv_mesh,
        "name": "mesh-0",
        "show_edges": True,
        "color": "#6ab0ff",
        "opacity": 1.0,
        "smooth_shading": True,
        "lighting": True,
        "specular": 0.2,
    }
    assert plotter.actors[0].property.calls == ["LightingOn", "SetInterpolationToPhong"]
    edge_call = plotter.mesh_calls[1]
    assert isinstance(edge_call.pop("mesh"), FakeFeatureEdges)
    assert edge_call == {
        "name": "mesh-0-edges",
        "color": "#cdd7ff",
        "line_width": 1.0,
        "render_lines_as_tubes": False,
    }


def test_preview_scene_controller_supports_legacy_feature_edge_signature(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakeLegacyPvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )

    controller.apply_scene(plotter, [mesh], show_edges=False, face_edges=True, show_bounds=False, show_axes=False)

    assert pv_mesh.feature_angles == [60.0]
    assert len(plotter.mesh_calls) == 2


def test_preview_scene_controller_can_render_edges_without_object_fill(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )

    controller.apply_scene(
        plotter,
        [mesh],
        show_edges=True,
        face_edges=True,
        show_bounds=False,
        show_axes=False,
        show_object_fill=False,
    )

    assert plotter.mesh_calls[0] == {
        "mesh": pv_mesh,
        "name": "mesh-0-wireframe",
        "color": "#cdd7ff",
        "style": "wireframe",
        "line_width": 1.0,
        "lighting": False,
    }
    assert plotter.actors[0].property.calls == ["LightingOff", "SetInterpolationToFlat"]
    assert isinstance(plotter.mesh_calls[1]["mesh"], FakeFeatureEdges)


def test_preview_scene_controller_resets_persistent_axes_and_bounds(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )

    controller.apply_scene(
        plotter,
        [mesh],
        show_bounds=False,
        show_axes=False,
        lighting=True,
        lighting_profile="face_normals",
        smooth_shading=False,
        specular=0.0,
    )

    assert plotter.hide_axes_all_calls == 1
    assert plotter.hide_axes_calls == 1
    assert plotter.remove_bounds_axes_calls == 1
    assert plotter.remove_bounding_box_calls == 1
    assert plotter.eye_dome_disabled_calls == 1
    assert plotter.axes_calls == []
    assert plotter.bounds_calls == []
    assert plotter.actors[0].property.calls == ["LightingOn", "SetInterpolationToFlat"]


def test_preview_scene_controller_camera_lighting_uses_smooth_actor_interpolation(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )

    controller.apply_scene(
        plotter,
        [mesh],
        show_bounds=False,
        show_axes=False,
        lighting=True,
        lighting_profile="camera",
        smooth_shading=True,
        specular=0.2,
    )

    assert plotter.actors[0].property.calls == ["LightingOn", "SetInterpolationToPhong"]


def test_preview_scene_controller_resets_camera_from_combined_bounds() -> None:
    plotter = FakePlotter()
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    polyline = Polyline(np.array([[0, 0, -1], [0, 0, 3]]))

    diagnostic = controller.reset_camera(plotter, [mesh, polyline])

    assert diagnostic.bounds == (0.0, 2.0, 0.0, 4.0, -1.0, 3.0)
    assert plotter.camera_position == diagnostic.camera_position
    assert controller.home_camera == diagnostic.camera_position
    assert plotter.camera_position[1] == (1.0, 2.0, 1.0)


def test_pyvista_previewer_delegates_scene_behavior_to_shared_controller() -> None:
    class FakeController:
        home_camera = ["home"]

        def __init__(self) -> None:
            self.configure_calls = []
            self.apply_calls = []
            self.reset_calls = []
            self.polyline_calls = []

        def configure_plotter(self, plotter, **kwargs):
            self.configure_calls.append((plotter, kwargs))
            return type("Diagnostic", (), {"eye_dome_lighting": False})()

        def apply_scene(self, plotter, datasets, **kwargs):
            self.apply_calls.append((plotter, list(datasets), kwargs))

        def add_feature_edges(self, plotter, mesh, index):
            return None

        def reset_camera(self, plotter, datasets):
            self.reset_calls.append((plotter, list(datasets)))
            return type("CameraDiagnostic", (), {"camera_position": ["camera"]})()

        def polyline_to_pyvista(self, polyline):
            self.polyline_calls.append(polyline)
            return "polydata"

    previewer = PyVistaPreviewer(console=None)
    fake_controller = FakeController()
    previewer._scene_controller = fake_controller
    plotter = FakePlotter()
    polyline = Polyline(np.array([[0, 0, 0], [1, 0, 0]]))

    previewer._configure_plotter(plotter, show_bounds=False, show_axes=True)
    previewer._apply_scene(
        plotter,
        [polyline],
        show_edges=True,
        face_edges=False,
        show_bounds=False,
        show_axes=True,
        align_camera=True,
    )
    previewer._reset_camera(plotter, [polyline])
    polydata = previewer._polyline_to_pyvista(polyline)

    assert fake_controller.configure_calls == [(plotter, {"show_bounds": False, "show_axes": True})]
    assert fake_controller.apply_calls == [
        (
            plotter,
            [polyline],
            {
                "show_edges": True,
                "face_edges": False,
                "show_bounds": False,
                "show_axes": True,
                "align_camera": True,
            },
        )
    ]
    assert fake_controller.reset_calls == [(plotter, [polyline])]
    assert fake_controller.polyline_calls == [polyline]
    assert previewer._home_camera == ["camera"]
    assert polydata == "polydata"


def test_preview_module_does_not_import_reference_review_ui() -> None:
    preview_source = Path("src/impression/preview.py").read_text()

    assert "impression.devtools.reference_review" not in preview_source
    assert "PySide6" not in preview_source


def test_qt_preview_surface_config_has_workbench_defaults() -> None:
    config = QtPreviewSurfaceConfig.workbench_default()

    assert config.controller_options.style.background == "#07111f"
    assert config.controller_options.interaction.show_bounds is False
    assert config.controller_options.interaction.show_axes is False
    assert config.controller_options.interaction.enable_eye_dome_lighting is False
    assert config.apply_options.show_edges is False
    assert config.apply_options.face_edges is False
    assert config.apply_options.show_bounds is False
    assert config.apply_options.show_axes is False
    assert config.apply_options.align_camera is True
    assert config.auto_update is False
    assert config.qvtk_base == "QOpenGLWidget"


def test_qt_preview_configures_qopengl_backend_before_pyvistaqt_import() -> None:
    import sys
    import vtkmodules.qt

    original_base = getattr(vtkmodules.qt, "QVTKRWIBase", None)
    sys.modules.pop("pyvistaqt.rwi", None)
    try:
        vtkmodules.qt.QVTKRWIBase = "QWidget"
        configure_qvtk_backend("QOpenGLWidget")

        assert vtkmodules.qt.QVTKRWIBase == "QOpenGLWidget"
    finally:
        if original_base is not None:
            vtkmodules.qt.QVTKRWIBase = original_base


def test_qt_preview_configures_opengl_compatibility_surface_format() -> None:
    from PySide6.QtGui import QSurfaceFormat

    configure_qt_preview_surface_format()

    fmt = QSurfaceFormat.defaultFormat()
    assert fmt.renderableType() == QSurfaceFormat.RenderableType.OpenGL
    assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
    assert fmt.majorVersion() == 2
    assert fmt.minorVersion() == 1
    assert fmt.depthBufferSize() == 24
    assert fmt.stencilBufferSize() == 8


def test_qt_preview_supported_environment_rejects_offscreen_by_default(monkeypatch) -> None:
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert qt_preview_supported_environment()

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert not qt_preview_supported_environment()
    assert qt_preview_supported_environment(allow_offscreen=True)


def test_qt_preview_scene_handoff_delegates_to_shared_controller() -> None:
    calls = []

    class FakeController:
        def apply_scene(self, plotter, datasets, **kwargs):
            calls.append((plotter, tuple(datasets), kwargs))

    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    options = PreviewSceneApplyOptions(
        show_edges=True,
        face_edges=True,
        show_bounds=False,
        show_axes=False,
        align_camera=True,
    )

    apply_qt_preview_scene(FakeController(), "plotter", [mesh], options)

    assert calls == [
        (
            "plotter",
            (mesh,),
            {
                "show_edges": True,
                "face_edges": True,
                "show_bounds": False,
                "show_axes": False,
                "align_camera": True,
                "show_object_fill": True,
                "show_polylines": True,
                "smooth_shading": True,
                "lighting": True,
                "lighting_profile": "camera",
                "specular": 0.2,
                "background": None,
                "background_top": None,
            },
        )
    ]


def test_qt_preview_preserves_display_options_when_resolving_camera_alignment() -> None:
    options = PreviewSceneApplyOptions(
        show_edges=True,
        face_edges=False,
        show_bounds=False,
        show_axes=True,
        align_camera=False,
        show_object_fill=False,
        show_polylines=False,
        smooth_shading=False,
        lighting=False,
        lighting_profile="flat",
        specular=0.0,
        background="#07111f",
        background_top="#10223a",
    )

    resolved = preview_scene_options_for_camera_state(
        options,
        align_camera=True,
        camera_aligned=False,
    )

    assert resolved == PreviewSceneApplyOptions(
        show_edges=True,
        face_edges=False,
        show_bounds=False,
        show_axes=True,
        align_camera=True,
        show_object_fill=False,
        show_polylines=False,
        smooth_shading=False,
        lighting=False,
        lighting_profile="flat",
        specular=0.0,
        background="#07111f",
        background_top="#10223a",
    )


def test_reference_review_shell_does_not_apply_preview_scenes() -> None:
    shell_source = Path("src/impression/devtools/reference_review/ui/shell.py").read_text()

    assert "mesh_to_pyvista" not in shell_source
    assert "extract_feature_edges" not in shell_source
    assert "._apply_scene(" not in shell_source
    assert "import pyvista" not in shell_source
