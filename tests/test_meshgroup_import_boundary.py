from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from impression.modeling import (
    MESH_GROUP_COMPATIBILITY_BOUNDARY,
    SurfaceBody,
    group,
    make_box,
    make_box_mesh,
    translate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_ROOT = PROJECT_ROOT / "src" / "impression" / "modeling"

MESH_COMPATIBILITY_MODULES = {
    "__init__.py",
    "_ops_mesh.py",
    "csg.py",
    "group.py",
    "ops.py",
    "transform.py",
}


@dataclass(frozen=True)
class MeshGroupImportBoundaryRule:
    """Static rule preventing authored surface modules from importing MeshGroup."""

    boundary: str = "meshgroup-import-boundary"
    allowed_modules: tuple[str, ...] = tuple(sorted(MESH_COMPATIBILITY_MODULES))
    banned_imports: tuple[str, ...] = ("MeshGroup", "group")


@dataclass(frozen=True)
class MeshGroupImportBoundaryViolation:
    """Diagnostic emitted by the static MeshGroup import boundary check."""

    path: str
    imported_name: str
    line: int
    boundary: str = "meshgroup-import-boundary"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "imported_name": self.imported_name,
            "line": self.line,
            "path": self.path,
        }


def _meshgroup_import_violations() -> list[MeshGroupImportBoundaryViolation]:
    rule = MeshGroupImportBoundaryRule()
    violations: list[MeshGroupImportBoundaryViolation] = []
    for path in sorted(MODELING_ROOT.glob("*.py")):
        if path.name in rule.allowed_modules:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if not (module == "impression.modeling.group" or (node.level == 1 and module == "group")):
                continue
            for alias in node.names:
                if alias.name in rule.banned_imports:
                    violations.append(
                        MeshGroupImportBoundaryViolation(
                            path=str(path.relative_to(PROJECT_ROOT)),
                            imported_name=alias.name,
                            line=node.lineno,
                        )
                    )
    return violations


def test_surface_authored_modules_do_not_import_meshgroup_composition_truth() -> None:
    violations = _meshgroup_import_violations()

    assert [violation.canonical_payload() for violation in violations] == []


def test_transform_route_keeps_surface_and_meshgroup_boundaries_separate() -> None:
    surface_body = make_box(size=(1.0, 1.0, 1.0))
    mesh_body = make_box_mesh(size=(1.0, 1.0, 1.0))
    mesh_group = group([mesh_body])

    moved_surface = translate(surface_body, (1.0, 0.0, 0.0))
    moved_group = translate(mesh_group, (1.0, 0.0, 0.0))

    assert isinstance(moved_surface, SurfaceBody)
    assert np.allclose(moved_surface.transform_matrix[:3, 3], [1.0, 0.0, 0.0])
    assert "transform_mesh_compatibility" not in moved_surface.metadata
    assert moved_group is mesh_group
    assert moved_group.metadata["mesh_group_compatibility"]["boundary"] == MESH_GROUP_COMPATIBILITY_BOUNDARY
    assert moved_group.metadata["transform_mesh_compatibility"][0]["boundary"] == "explicit-mesh-compatibility"
