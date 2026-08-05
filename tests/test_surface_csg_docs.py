from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
from typing import Iterable, get_type_hints

import impression.modeling as modeling
from impression.modeling import (
    SurfaceBody,
    SurfaceBooleanResult,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    export_tessellation_request,
    inventory_legacy_primitive_mesh_assumptions,
    preview_tessellation_request,
    tessellate_surface_body,
)
from impression.modeling import mesh_tools


SURFACE_CSG_EXAMPLES = (
    Path("docs/examples/csg/union_example.py"),
    Path("docs/examples/csg/difference_example.py"),
    Path("docs/examples/csg/intersection_example.py"),
)

EXPLICIT_MESH_CSG_EXAMPLES = (
    Path("docs/examples/csg/union_meshes_example.py"),
    Path("docs/examples/csg/teeth_union_example.py"),
    Path("docs/examples/csg/tooth_union_example.py"),
)


def test_csg_reference_documents_the_surface_only_runtime_contract(project_root: Path) -> None:
    doc = (project_root / "docs/modeling/csg.md").read_text(encoding="utf-8")

    assert "boolean_union(bodies: Iterable[SurfaceBody]" in doc
    assert "boolean_difference(base: SurfaceBody, cutters: Iterable[SurfaceBody]" in doc
    assert "boolean_intersection(bodies: Iterable[SurfaceBody]" in doc
    assert doc.count("-> SurfaceBooleanResult") >= 3
    assert "Passing `Mesh`" in doc
    assert "`MeshGroup`, or a mixed collection raises `TypeError`" in doc
    assert "before CSG family selection or kernel dispatch" in doc
    assert "result.body" in doc
    assert '`no-cut`' in doc
    assert '`invalid`' in doc
    assert '`unsupported`' in doc
    assert "does not fall back to mesh" in doc
    assert "impression.modeling.mesh_tools" in doc
    assert "intentionally absent from the\ntop-level `impression.modeling` export table" in doc

    forbidden = (
        "boolean_union(meshes",
        "boolean_intersection(meshes",
        "public boolean execution helpers remain mesh-primary",
        "Surface-body booleans are still in migration",
    )
    assert not any(token in doc for token in forbidden)


def test_csg_reference_keeps_surface_reference_evidence_explicit(project_root: Path) -> None:
    doc = (project_root / "docs/modeling/csg.md").read_text(encoding="utf-8")

    assert "surfacebody/csg_union_box_post" in doc
    assert "surfacebody/csg_difference_slot" in doc
    assert "surfacebody/csg_intersection_box_sphere" in doc
    assert "dirty and clean reference images" in doc
    assert "dirty and clean\nreference STL files" in doc
    assert "triptych-style operand/result presentation" in doc
    assert "edge-protrusion cue" in doc
    assert "expected\nsection bitmap" in doc
    assert "same shape but rotated" in doc


def test_tutorials_use_surface_results_and_explicit_mesh_tool_guidance(project_root: Path) -> None:
    getting_started = (project_root / "docs/tutorials/getting-started.md").read_text(encoding="utf-8")
    serious_modeling = (project_root / "docs/tutorials/serious-modeling.md").read_text(encoding="utf-8")

    assert "result = boolean_union([base, post])" in getting_started
    assert "return result.body" in getting_started
    assert "returns a `SurfaceBody`, not a mesh" in getting_started
    assert "Public CSG accepts only `SurfaceBody` operands" in getting_started
    assert "executable mesh boolean lane" not in serious_modeling
    assert "surface-only public booleans" in serious_modeling
    assert "impression.modeling.mesh_tools" in serious_modeling


def test_surface_csg_examples_execute_through_public_modeling_api(project_root: Path) -> None:
    for relative_path in SURFACE_CSG_EXAMPLES:
        source = (project_root / relative_path).read_text(encoding="utf-8")
        namespace = runpy.run_path(str(project_root / relative_path))
        body = namespace["build"]()

        assert isinstance(body, SurfaceBody), relative_path
        assert body.shell_count == 1, relative_path
        assert "make_box_mesh" not in source, relative_path
        assert "make_cylinder_mesh" not in source, relative_path
        assert "make_sphere_mesh" not in source, relative_path
        assert "result.body" in source, relative_path
        for request in (preview_tessellation_request(), export_tessellation_request()):
            mesh = tessellate_surface_body(body, request).mesh
            assert mesh.n_faces > 0, (relative_path, request.consumer)


def test_mesh_csg_examples_import_union_only_from_explicit_tool_boundary(project_root: Path) -> None:
    for relative_path in EXPLICIT_MESH_CSG_EXAMPLES:
        source = (project_root / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(relative_path))
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

        assert any(
            node.module == "impression.modeling.mesh_tools"
            and any(alias.name == "union_meshes" for alias in node.names)
            for node in imports
        ), relative_path
        assert not any(
            node.module == "impression.modeling"
            and any(alias.name == "union_meshes" for alias in node.names)
            for node in imports
        ), relative_path


def test_source_docs_and_exports_share_one_surface_boolean_inventory(project_root: Path) -> None:
    expected = {
        boolean_union: (("bodies", "tolerance"), Iterable[SurfaceBody]),
        boolean_difference: (("base", "cutters", "tolerance"), SurfaceBody),
        boolean_intersection: (("bodies", "tolerance"), Iterable[SurfaceBody]),
    }
    for function, (parameter_names, first_type) in expected.items():
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        assert tuple(signature.parameters) == parameter_names
        assert hints[parameter_names[0]] == first_type
        assert hints["return"] is SurfaceBooleanResult

    assert get_type_hints(boolean_difference)["cutters"] == Iterable[SurfaceBody]
    assert "union_meshes" not in modeling.__all__
    assert not hasattr(modeling, "union_meshes")
    assert "union_meshes" in mesh_tools.__all__
    assert callable(mesh_tools.union_meshes)

    inventory_paths = (
        Path("README.md"),
        Path("docs/modeling/csg.md"),
        Path("docs/tutorials/getting-started.md"),
        *SURFACE_CSG_EXAMPLES,
        *EXPLICIT_MESH_CSG_EXAMPLES,
        Path("docs/examples/csg/tooth_example.py"),
        Path("docs/examples/csg/tooth_parts_example.py"),
    )
    report = inventory_legacy_primitive_mesh_assumptions(
        {
            str(path): (project_root / path).read_text(encoding="utf-8")
            for path in inventory_paths
        }
    )
    assert report.stale_findings == ()


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout


def test_clean_wheel_exposes_the_same_surface_boolean_contract(project_root: Path, tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    install_root = tmp_path / "installed"
    wheelhouse.mkdir()

    _run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(project_root),
        ],
        cwd=tmp_path,
    )
    wheels = tuple(wheelhouse.glob("impression-*.whl"))
    assert len(wheels) == 1
    _run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        cwd=tmp_path,
    )

    smoke = textwrap.dedent(
        """
        import inspect
        import os
        from pathlib import Path
        from typing import Iterable, get_type_hints

        import impression.modeling as modeling
        from impression.modeling.mesh_tools import union_meshes

        install_root = Path(os.environ["IMPRESSION_CLEAN_INSTALL_ROOT"]).resolve()
        module_path = Path(modeling.__file__).resolve()
        assert install_root in module_path.parents, (install_root, module_path)

        expected = {
            modeling.boolean_union: (("bodies", "tolerance"), Iterable[modeling.SurfaceBody]),
            modeling.boolean_difference: (("base", "cutters", "tolerance"), modeling.SurfaceBody),
            modeling.boolean_intersection: (("bodies", "tolerance"), Iterable[modeling.SurfaceBody]),
        }
        for function, (parameter_names, first_type) in expected.items():
            signature = inspect.signature(function)
            hints = get_type_hints(function)
            assert tuple(signature.parameters) == parameter_names
            assert hints[parameter_names[0]] == first_type
            assert hints["return"] is modeling.SurfaceBooleanResult

        assert get_type_hints(modeling.boolean_difference)["cutters"] == Iterable[modeling.SurfaceBody]
        assert "union_meshes" not in modeling.__all__
        assert not hasattr(modeling, "union_meshes")
        assert callable(union_meshes)

        outer = modeling.make_box(size=(2, 2, 2))
        inner = modeling.make_box(size=(1, 1, 1))
        result = modeling.boolean_union([outer, inner])
        assert isinstance(result, modeling.SurfaceBooleanResult)
        assert result.status == "succeeded"
        assert result.body is not None

        try:
            modeling.boolean_union([modeling.make_box_mesh(size=(1, 1, 1))])
        except TypeError as exc:
            message = str(exc)
            assert "accepts only SurfaceBody operands" in message
            assert "impression.modeling.mesh_tools" in message
        else:
            raise AssertionError("clean wheel accepted a mesh modeling operand")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_root)
    env["IMPRESSION_CLEAN_INSTALL_ROOT"] = str(install_root)
    env.pop("PYTHONHOME", None)
    _run_checked([sys.executable, "-c", smoke], cwd=tmp_path, env=env)
