from __future__ import annotations

from dataclasses import dataclass

from impression.mesh import Mesh, MeshAnalysis, analyze_mesh, repair_mesh, section_mesh_with_plane
from impression.modeling._ops_mesh import hull_mesh, manifold_from_mesh_group, mesh_from_manifold

MESH_TOOL_BOUNDARY = "explicit-mesh-tool"
MESH_TOOL_CLASSIFICATION = "mesh-utility-quarantine"


@dataclass(frozen=True)
class MeshUtilityClassification:
    """Classification record for retained mesh utilities."""

    symbol: str
    module: str
    purpose: str = "analysis-repair-debugging"
    boundary: str = MESH_TOOL_BOUNDARY
    classification: str = MESH_TOOL_CLASSIFICATION

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "classification": self.classification,
            "module": self.module,
            "purpose": self.purpose,
            "symbol": self.symbol,
        }


def classify_mesh_utility(symbol: str, *, module: str = "impression.modeling.mesh_tools") -> MeshUtilityClassification:
    return MeshUtilityClassification(symbol=str(symbol), module=str(module))


__all__ = [
    "MESH_TOOL_BOUNDARY",
    "MESH_TOOL_CLASSIFICATION",
    "Mesh",
    "MeshAnalysis",
    "MeshUtilityClassification",
    "analyze_mesh",
    "classify_mesh_utility",
    "hull_mesh",
    "manifold_from_mesh_group",
    "mesh_from_manifold",
    "repair_mesh",
    "section_mesh_with_plane",
]
