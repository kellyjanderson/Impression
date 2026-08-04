from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from typer.testing import CliRunner

import impression.preview as preview_module
from impression.cli import app
from impression.mesh import Mesh, Polyline
from impression.modeling import (
    SurfaceConsumerCollection,
    SurfaceConsumerRecord,
    SurfaceSceneGroup,
    SurfaceSceneNode,
    export_tessellation_request,
    make_box,
    translate,
)
from impression.modeling.group import MeshGroup
from impression.preview import PreviewBackendError, PyVistaPreviewer, _collect_datasets_from_scene


def test_preview_collects_direct_surface_body_once_with_preview_policy(monkeypatch) -> None:
    body = make_box(size=(2.0, 3.0, 4.0))
    real_tessellate = preview_module.tessellate_surface_body
    requests = []

    def recording_tessellation(surface_body, request):
        requests.append(request)
        return real_tessellate(surface_body, request)

    monkeypatch.setattr(preview_module, "tessellate_surface_body", recording_tessellation)
    previewer = PyVistaPreviewer(console=Console())

    datasets = previewer.collect_datasets(body)

    assert len(datasets) == 1
    assert isinstance(datasets[0], Mesh)
    assert datasets[0].n_faces > 0
    assert [request.intent for request in requests] == ["preview"]


def test_export_command_collects_direct_surface_body_once_with_export_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "surface_model.py"
    model.write_text(
        "from impression.modeling import make_box\n\n"
        "def build():\n"
        "    return make_box(size=(2.0, 3.0, 4.0))\n"
    )
    output = tmp_path / "surface_model.stl"
    real_tessellate = preview_module.tessellate_surface_body
    requests = []

    def recording_tessellation(surface_body, request):
        requests.append(request)
        return real_tessellate(surface_body, request)

    monkeypatch.setattr(preview_module, "tessellate_surface_body", recording_tessellation)

    result = CliRunner().invoke(
        app,
        ("export", str(model), "--output", str(output)),
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert output.stat().st_size > 84
    assert [request.intent for request in requests] == ["export"]


def test_preview_command_reaches_viewport_boundary_with_direct_surface_body(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "surface_preview_model.py"
    model.write_text(
        "from impression.modeling import make_box\n\n"
        "def build():\n"
        "    return make_box(size=(2.0, 3.0, 4.0))\n"
    )
    viewport_datasets = []

    def capture_viewport(self, *, initial_scene, **_kwargs) -> None:
        viewport_datasets.extend(self.collect_datasets(initial_scene))

    monkeypatch.setattr(PyVistaPreviewer, "show", capture_viewport)

    result = CliRunner().invoke(app, ("preview", str(model), "--no-watch"))

    assert result.exit_code == 0, result.output
    assert len(viewport_datasets) == 1
    assert isinstance(viewport_datasets[0], Mesh)
    assert viewport_datasets[0].n_faces > 0


def test_mixed_nested_scene_preserves_order_and_applies_transforms_once() -> None:
    translated_surface = translate(make_box(size=(2.0, 2.0, 2.0)), (10.0, 0.0, 0.0))
    compatibility_mesh = Mesh(
        vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        faces=np.asarray(((0, 1, 2),)),
    )
    mesh_group = MeshGroup([compatibility_mesh]).translate((0.0, 5.0, 0.0))
    polyline = Polyline(np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))))
    collection_body = translate(make_box(size=(1.0, 1.0, 1.0)), (0.0, 0.0, 7.0))
    collection = SurfaceConsumerCollection(
        (
            SurfaceConsumerRecord(
                body=collection_body,
                source_id="collection-body",
                order=0,
            ),
        )
    )

    datasets = _collect_datasets_from_scene(
        [translated_surface, [mesh_group, polyline], collection],
        tessellation_request=export_tessellation_request(),
    )

    assert [type(dataset) for dataset in datasets] == [Mesh, Mesh, Polyline, Mesh]
    assert np.allclose(datasets[0].bounds[0:2], (9.0, 11.0))
    assert np.allclose(datasets[1].bounds[2:4], (5.0, 6.0))
    assert np.allclose(datasets[3].bounds[4:6], (6.5, 7.5))


def test_surface_scene_group_preserves_node_order_visibility_and_transform() -> None:
    first_transform = np.eye(4)
    first_transform[0, 3] = 3.0
    hidden_transform = np.eye(4)
    hidden_transform[0, 3] = 99.0
    second_transform = np.eye(4)
    second_transform[0, 3] = 8.0
    root = SurfaceSceneGroup(
        "root",
        (
            SurfaceSceneNode("first", make_box(size=(1.0, 1.0, 1.0)), first_transform),
            SurfaceSceneNode(
                "hidden",
                make_box(size=(1.0, 1.0, 1.0)),
                hidden_transform,
                visible=False,
            ),
            SurfaceSceneGroup(
                "nested",
                (
                    SurfaceSceneNode(
                        "second",
                        make_box(size=(2.0, 2.0, 2.0)),
                        second_transform,
                    ),
                ),
            ),
        ),
    )

    datasets = _collect_datasets_from_scene(root)

    assert len(datasets) == 2
    assert np.allclose(datasets[0].bounds[0:2], (2.5, 3.5))
    assert np.allclose(datasets[1].bounds[0:2], (7.0, 9.0))


def test_unsupported_scene_value_has_named_consumer_diagnostic() -> None:
    try:
        _collect_datasets_from_scene({"unsupported": object()})
    except PreviewBackendError as exc:
        diagnostic = str(exc)
    else:  # pragma: no cover - required refusal
        raise AssertionError("unsupported payload was accepted")

    assert "Unsupported scene value dict" in diagnostic
    assert "SurfaceBody, Mesh, Polyline, or a supported group" in diagnostic
