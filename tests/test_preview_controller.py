from __future__ import annotations

import io
import numpy as np
import threading
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from watchfiles import Change

from impression._config import UnitSettings
from impression.mesh import Mesh, Polyline
from impression.preview import (
    PreviewControllerOptions,
    PreviewInteractionPolicy,
    PreviewReloadCoordinator,
    ReloadReason,
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
        self.light_calls: list[FakeLight] = []
        self.active_lights: list[FakeLight] = []
        self.camera_position = None
        self.renderer = FakeRenderer(self)

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

    def add_light(self, light: "FakeLight") -> None:
        self.light_calls.append(light)
        if light not in self.active_lights:
            self.active_lights.append(light)

    def clear(self) -> None:
        self.clear_calls += 1
        self.active_lights.clear()

    def add_mesh(self, mesh: object, **kwargs: object) -> "FakeActor":
        call = dict(kwargs)
        call["mesh"] = mesh
        self.mesh_calls.append(call)
        actor = FakeActor()
        self.actors.append(actor)
        return actor


class FakeInteractivePreviewPlotter(FakePlotter):
    def __init__(self, show_driver: Callable[["FakeInteractivePreviewPlotter"], None]) -> None:
        super().__init__()
        self.camera_position = ["original-camera"]
        self.key_events: dict[str, Callable[[], None]] = {}
        self.render_calls = 0
        self.closed = False
        self._show_driver = show_driver

    def add_key_event(self, key: str, callback: Callable[[], None]) -> None:
        self.key_events[key] = callback

    def reset_camera_clipping_range(self) -> None:
        return None

    def render(self) -> None:
        self.render_calls += 1

    def show(self, **_kwargs: object) -> None:
        self._show_driver(self)

    def close(self) -> None:
        self.closed = True


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


class FakeRenderer:
    def __init__(self, plotter: FakePlotter) -> None:
        self._plotter = plotter

    @property
    def lights(self) -> tuple["FakeLight", ...]:
        return tuple(self._plotter.active_lights)


class FakeLight:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        self.switch_calls: list[str] = []

    def switch_on(self) -> None:
        self.switch_calls.append("on")

    def switch_off(self) -> None:
        self.switch_calls.append("off")


class FakeLightingPyVista:
    created: list[FakeLight] = []

    @classmethod
    def Light(cls, **kwargs: object) -> FakeLight:
        light = FakeLight(**kwargs)
        cls.created.append(light)
        return light


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
        "split_sharp_edges": True,
        "feature_angle": 60.0,
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
    assert plotter.mesh_calls[0]["split_sharp_edges"] is False
    assert plotter.mesh_calls[0]["feature_angle"] == 60.0


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
    assert plotter.mesh_calls[0]["split_sharp_edges"] is True
    assert plotter.mesh_calls[0]["feature_angle"] == 60.0


def test_preview_scene_controller_splits_sharp_edges_for_face_colored_mesh(monkeypatch) -> None:
    import impression.preview as preview_module

    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(unit_settings=UnitSettings("millimeters", "mm", 1.0))
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        face_colors=np.array([[0.2, 0.4, 0.8, 1.0]]),
    )

    controller.apply_scene(plotter, [mesh], show_bounds=False, show_axes=False)

    assert plotter.mesh_calls[0]["split_sharp_edges"] is True
    assert plotter.mesh_calls[0]["feature_angle"] == 60.0


def test_preview_scene_controller_reuses_predefined_light_presets(monkeypatch) -> None:
    import impression.preview as preview_module

    FakeLightingPyVista.created = []
    plotter = FakePlotter()
    pv_mesh = FakePvMesh()
    monkeypatch.setattr(preview_module, "mesh_to_pyvista", lambda mesh: pv_mesh)
    controller = PreviewSceneController(
        unit_settings=UnitSettings("millimeters", "mm", 1.0),
        pyvista_provider=lambda: FakeLightingPyVista,
    )
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
    )
    controller.apply_scene(
        plotter,
        [mesh],
        show_bounds=False,
        show_axes=False,
        lighting=True,
        lighting_profile="camera",
        smooth_shading=True,
    )
    controller.apply_scene(
        plotter,
        [mesh],
        show_bounds=False,
        show_axes=False,
        lighting=False,
        lighting_profile="flat",
        smooth_shading=False,
    )

    assert [light.kwargs for light in FakeLightingPyVista.created] == [
        {"light_type": "headlight", "intensity": 0.9},
        {"light_type": "camera light", "intensity": 0.35},
    ]
    head, fill = FakeLightingPyVista.created
    assert head.switch_calls == ["on", "on", "off"]
    assert fill.switch_calls == ["off", "on", "off"]
    assert len(FakeLightingPyVista.created) == 2
    assert plotter.light_calls == [head, fill, head, fill, head, fill]
    assert plotter.active_lights == [head, fill]


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


def test_reload_coordinator_keeps_one_active_and_latest_forced_replacement(
    tmp_path: Path,
) -> None:
    coordinator = PreviewReloadCoordinator()
    model = tmp_path / "model.py"
    helper = tmp_path / "helper.py"

    first = coordinator.submit_reload(
        ReloadReason.FILE_CHANGE,
        changed_paths=(model,),
    )
    assert first is not None
    assert coordinator.begin_next_build() == first

    automatic = coordinator.submit_reload(
        ReloadReason.FILE_CHANGE,
        changed_paths=(helper,),
    )
    forced = coordinator.submit_reload(
        ReloadReason.MANUAL_REFRESH,
        force=True,
        changed_paths=(model,),
    )

    assert automatic is not None
    assert forced is not None
    assert not coordinator.is_current(first.generation)
    assert not coordinator.complete_build(forced.generation)
    assert coordinator.complete_build(first.generation)

    replacement = coordinator.begin_next_build()
    assert replacement is not None
    assert replacement.generation == forced.generation
    assert replacement.reason == ReloadReason.MANUAL_REFRESH
    assert replacement.force
    assert replacement.changed_paths == (helper, model)
    assert coordinator.is_current(replacement.generation)
    assert coordinator.complete_build(replacement.generation)
    assert coordinator.begin_next_build() is None


def test_reload_coordinator_shutdown_rejects_new_and_pending_work() -> None:
    coordinator = PreviewReloadCoordinator()
    pending = coordinator.submit_reload(ReloadReason.ANIMATION)
    assert pending is not None

    coordinator.shutdown()

    assert coordinator.begin_next_build() is None
    assert coordinator.submit_reload(ReloadReason.MANUAL_REFRESH, force=True) is None
    assert not coordinator.is_current(pending.generation)


def test_model_watcher_uses_low_latency_coalescing(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def build():\n    return None\n")
    submitted: list[tuple[ReloadReason, tuple[Path, ...]]] = []
    watch_call: dict[str, object] = {}

    def fake_watch(*paths, **kwargs):
        watch_call["paths"] = paths
        watch_call["kwargs"] = kwargs
        yield {(Change.modified, str(model_path))}

    monkeypatch.setattr("impression.preview.watch", fake_watch)
    previewer = PyVistaPreviewer(console=None)
    previewer._watch_model_file(
        {"path": model_path},
        lambda reason, paths: submitted.append((reason, paths)),
        threading.Event(),
        [tmp_path],
        None,
        watch_paths_getter=lambda: (model_path,),
    )

    assert watch_call["paths"] == (str(tmp_path),)
    assert watch_call["kwargs"]["debounce"] == 50
    assert watch_call["kwargs"]["step"] == 10
    assert submitted == [(ReloadReason.FILE_CHANGE, (model_path.resolve(),))]


def test_model_watcher_delivers_real_filesystem_change_within_budget(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def build():\n    return None\n")
    submitted = threading.Event()
    observed: list[tuple[ReloadReason, tuple[Path, ...], float]] = []
    stop_event = threading.Event()
    previewer = PyVistaPreviewer(console=Console(file=io.StringIO()))

    def capture(reason: ReloadReason, paths: tuple[Path, ...]) -> None:
        observed.append((reason, paths, time.monotonic()))
        submitted.set()

    watcher = threading.Thread(
        target=previewer._watch_model_file,
        args=(
            {"path": model_path},
            capture,
            stop_event,
            [tmp_path],
            None,
        ),
        kwargs={"watch_paths_getter": lambda: (model_path,)},
        daemon=True,
    )
    watcher.start()
    time.sleep(0.05)
    changed_at = time.monotonic()
    model_path.write_text("def build():\n    return 1\n")

    try:
        assert submitted.wait(timeout=0.25)
        assert observed[0][0] == ReloadReason.FILE_CHANGE
        assert observed[0][1] == (model_path.resolve(),)
        assert observed[0][2] - changed_at <= 0.25
    finally:
        stop_event.set()
        watcher.join(timeout=1.0)


def test_preview_filesystem_change_reaches_build_submission_within_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def build():\n    return None\n")
    initial = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    rebuilt = Mesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    output = io.StringIO()
    timer_callback: dict[str, Callable[[], None]] = {}
    build_started: list[float] = []
    applied: list[list[Mesh | Polyline]] = []
    changed_at = 0.0

    def drive_preview(plotter: FakeInteractivePreviewPlotter) -> None:
        nonlocal changed_at
        time.sleep(0.05)
        changed_at = time.monotonic()
        model_path.write_text("def build():\n    return 1\n")
        callback = timer_callback["callback"]
        for _ in range(250):
            callback()
            if len(applied) == 2:
                return
            time.sleep(0.001)
        raise AssertionError("filesystem change did not rebuild the preview")

    plotter = FakeInteractivePreviewPlotter(drive_preview)

    class FakeBackend:
        @staticmethod
        def Plotter(**_kwargs: object) -> FakeInteractivePreviewPlotter:
            return plotter

    previewer = PyVistaPreviewer(console=Console(file=output, force_terminal=False))
    monkeypatch.setattr(previewer, "_ensure_backend", lambda: FakeBackend())
    monkeypatch.setattr(previewer, "_configure_plotter", lambda *_args, **_kwargs: None)

    def apply_scene(_plotter, datasets, **_kwargs: object) -> None:
        applied.append(list(datasets))

    monkeypatch.setattr(previewer, "_apply_scene", apply_scene)

    def install_timer(_plotter, callback: Callable[[], None], _interval: float):
        timer_callback["callback"] = callback
        return lambda: None

    monkeypatch.setattr(previewer, "_install_timer_callback", install_timer)

    def scene_factory() -> Mesh:
        build_started.append(time.monotonic())
        return rebuilt

    previewer.show(
        scene_factory=scene_factory,
        initial_scene=initial,
        model_path=model_path,
        watch_files=True,
        target_fps=60,
        watch_paths_getter=lambda: (model_path,),
    )

    assert build_started
    assert build_started[0] - changed_at <= 0.25
    assert applied[0][0] is initial
    assert applied[1][0] is rebuilt


def test_preview_r_route_retains_last_good_scene_camera_and_recovers(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def build():\n    return None\n")
    initial = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    recovered = Mesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    timer_callback: dict[str, object] = {}
    applied: list[list[Mesh | Polyline]] = []
    force_calls: list[str] = []
    build_results: list[object] = [RuntimeError("broken model"), recovered]

    def drive_preview(plotter: FakeInteractivePreviewPlotter) -> None:
        callback = timer_callback["callback"]
        plotter.key_events["r"]()
        for _ in range(200):
            callback()
            if "Preview rebuild failed" in output.getvalue():
                break
            time.sleep(0.001)
        else:
            raise AssertionError("forced failure did not reach the preview route")

        assert len(applied) == 1
        assert plotter.camera_position == ["original-camera"]

        plotter.key_events["r"]()
        for _ in range(200):
            callback()
            if "Preview recovered after rebuild failure" in output.getvalue():
                break
            time.sleep(0.001)
        else:
            raise AssertionError("recovery did not reach the preview route")

    plotter = FakeInteractivePreviewPlotter(drive_preview)

    class FakeBackend:
        @staticmethod
        def Plotter(**_kwargs: object) -> FakeInteractivePreviewPlotter:
            return plotter

    previewer = PyVistaPreviewer(console=console)
    monkeypatch.setattr(previewer, "_ensure_backend", lambda: FakeBackend())
    monkeypatch.setattr(previewer, "_configure_plotter", lambda *_args, **_kwargs: None)

    def apply_scene(_plotter, datasets, **_kwargs: object) -> None:
        applied.append(list(datasets))
        _plotter.camera_position = (
            ["original-camera"]
            if len(applied) == 1
            else ["renderer-mutated-camera"]
        )

    monkeypatch.setattr(previewer, "_apply_scene", apply_scene)

    def install_timer(_plotter, callback, _interval: float):
        timer_callback["callback"] = callback
        return lambda: None

    monkeypatch.setattr(previewer, "_install_timer_callback", install_timer)

    def idle_watcher(
        _model_state,
        _submit_reload,
        watcher_stop_event: threading.Event,
        _watch_roots,
        _control_path,
        **_kwargs: object,
    ) -> None:
        watcher_stop_event.wait(timeout=1.0)

    monkeypatch.setattr(previewer, "_watch_model_file", idle_watcher)

    def scene_factory() -> object:
        result = build_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    previewer.show(
        scene_factory=scene_factory,
        initial_scene=initial,
        model_path=model_path,
        watch_files=True,
        target_fps=60,
        force_scene_reload=lambda: force_calls.append("forced"),
    )

    assert len(applied) == 2
    assert applied[0] == [initial]
    assert applied[1] == [recovered]
    assert force_calls == ["forced", "forced"]
    assert plotter.camera_position == ["original-camera"]
    assert plotter.render_calls == 1
    assert plotter.closed


def test_preview_discards_stale_build_before_renderer_mutation(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def build():\n    return None\n")
    initial = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    stale = Mesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    current = Mesh(
        vertices=np.array([[0, 0, 0], [3, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    timer_callback: dict[str, Callable[[], None]] = {}
    applied: list[list[Mesh | Polyline]] = []
    first_started = threading.Event()
    release_first = threading.Event()
    build_count = 0

    def drive_preview(plotter: FakeInteractivePreviewPlotter) -> None:
        callback = timer_callback["callback"]
        plotter.key_events["r"]()
        callback()
        assert first_started.wait(timeout=0.25)
        plotter.key_events["r"]()
        callback()
        release_first.set()
        for _ in range(250):
            callback()
            if len(applied) == 2:
                return
            time.sleep(0.001)
        raise AssertionError("latest build did not reach renderer mutation")

    plotter = FakeInteractivePreviewPlotter(drive_preview)

    class FakeBackend:
        @staticmethod
        def Plotter(**_kwargs: object) -> FakeInteractivePreviewPlotter:
            return plotter

    previewer = PyVistaPreviewer(console=Console(file=io.StringIO(), force_terminal=False))
    monkeypatch.setattr(previewer, "_ensure_backend", lambda: FakeBackend())
    monkeypatch.setattr(previewer, "_configure_plotter", lambda *_args, **_kwargs: None)

    def apply_scene(_plotter, datasets, **_kwargs: object) -> None:
        applied.append(list(datasets))

    monkeypatch.setattr(previewer, "_apply_scene", apply_scene)

    def install_timer(_plotter, callback: Callable[[], None], _interval: float):
        timer_callback["callback"] = callback
        return lambda: None

    monkeypatch.setattr(previewer, "_install_timer_callback", install_timer)

    def idle_watcher(
        _model_state,
        _submit_reload,
        watcher_stop_event: threading.Event,
        _watch_roots,
        _control_path,
        **_kwargs: object,
    ) -> None:
        watcher_stop_event.wait(timeout=1.0)

    monkeypatch.setattr(previewer, "_watch_model_file", idle_watcher)

    def scene_factory() -> Mesh:
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            first_started.set()
            assert release_first.wait(timeout=0.25)
            return stale
        return current

    previewer.show(
        scene_factory=scene_factory,
        initial_scene=initial,
        model_path=model_path,
        watch_files=True,
        target_fps=60,
    )

    assert build_count == 2
    assert applied[0][0] is initial
    assert applied[1][0] is current
    assert all(batch[0] is not stale for batch in applied)


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
    assert config.qvtk_base == "QWidget"


def test_qt_preview_configures_selected_backend_before_pyvistaqt_import() -> None:
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


def test_qt_preview_keeps_default_surface_format_for_vtk_qwidget() -> None:
    from PySide6.QtGui import QSurfaceFormat

    original = QSurfaceFormat.defaultFormat()
    configure_qt_preview_surface_format()

    fmt = QSurfaceFormat.defaultFormat()
    assert fmt.renderableType() == original.renderableType()
    assert fmt.profile() == original.profile()
    assert fmt.majorVersion() == original.majorVersion()
    assert fmt.minorVersion() == original.minorVersion()
    assert fmt.depthBufferSize() == original.depthBufferSize()
    assert fmt.stencilBufferSize() == original.stencilBufferSize()


def test_qt_preview_does_not_force_widgets_rhi_compositor_off(monkeypatch) -> None:
    import os

    monkeypatch.delenv("QT_WIDGETS_RHI", raising=False)
    configured_qt_opengl = os.environ.get("QT_OPENGL")
    import impression.preview_qt  # noqa: F401

    assert os.environ["QT_OPENGL"] == (configured_qt_opengl or "desktop")
    assert "QT_WIDGETS_RHI" not in os.environ


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
