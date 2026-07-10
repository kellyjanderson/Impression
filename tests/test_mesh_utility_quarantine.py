from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from impression.modeling import (
    MESH_TOOL_BOUNDARY,
    MESH_TOOL_CLASSIFICATION,
    MeshUtilityClassification,
    classify_mesh_utility,
    make_box,
)
from impression.modeling import mesh_tools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_ROOT = PROJECT_ROOT / "src" / "impression" / "modeling"

MESH_TOOL_ALLOWED_MODULES = {
    "_ops_mesh.py",
    "mesh_tools/__init__.py",
}


@dataclass(frozen=True)
class MeshUtilityImportBoundaryViolation:
    """Diagnostic for modules importing retained mesh tools outside quarantine."""

    path: str
    imported_module: str
    line: int
    boundary: str = "mesh-utility-quarantine"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "imported_module": self.imported_module,
            "line": self.line,
            "path": self.path,
        }


def _mesh_utility_import_violations() -> list[MeshUtilityImportBoundaryViolation]:
    violations: list[MeshUtilityImportBoundaryViolation] = []
    for path in sorted(MODELING_ROOT.rglob("*.py")):
        relative = path.relative_to(MODELING_ROOT).as_posix()
        if relative in MESH_TOOL_ALLOWED_MODULES:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            imports_legacy_ops = module == "impression.modeling._ops_mesh" or (
                node.level == 1 and module == "_ops_mesh"
            )
            if imports_legacy_ops:
                violations.append(
                    MeshUtilityImportBoundaryViolation(
                        path=f"src/impression/modeling/{relative}",
                        imported_module=module,
                        line=node.lineno,
                    )
                )
    return violations


def test_mesh_utility_namespace_classifies_retained_tools() -> None:
    classification = classify_mesh_utility("hull_mesh")

    assert isinstance(classification, MeshUtilityClassification)
    assert classification.boundary == MESH_TOOL_BOUNDARY
    assert classification.classification == MESH_TOOL_CLASSIFICATION
    assert classification.canonical_payload()["symbol"] == "hull_mesh"


def test_mesh_tools_namespace_re_exports_retained_mesh_analysis_tooling() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    analysis = mesh_tools.analyze_mesh(mesh)

    assert analysis.n_vertices == mesh.n_vertices
    assert analysis.n_faces == mesh.n_faces


def test_modeling_modules_route_mesh_utilities_through_quarantine_namespace() -> None:
    violations = _mesh_utility_import_violations()

    assert [violation.canonical_payload() for violation in violations] == []


def test_public_hull_route_imports_mesh_tools_namespace() -> None:
    ops_source = (MODELING_ROOT / "ops.py").read_text()

    assert "from .mesh_tools import hull_mesh" in ops_source
    assert "from ._ops_mesh import hull_mesh" not in ops_source
