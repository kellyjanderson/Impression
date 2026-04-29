from __future__ import annotations

import warnings

import pytest

from impression.mesh import combine_meshes
from impression.modeling import (
    HingeSurfaceAssembly,
    SurfaceBody,
    SurfaceConsumerCollection,
    handoff_hinge_surface,
    make_bistable_hinge,
    make_living_hinge,
    make_traditional_hinge_leaf,
    make_traditional_hinge_pair,
    tessellate_surface_body,
)


def _combined_collection_mesh(collection: SurfaceConsumerCollection):
    meshes = [tessellate_surface_body(record.body).mesh for record in collection.items]
    return combine_meshes(meshes)


def test_surface_traditional_hinge_leaf_returns_surface_body_without_deprecation() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        leaf = make_traditional_hinge_leaf(width=24.0, knuckle_count=5, backend="surface")

    assert isinstance(leaf, SurfaceBody)
    mesh = tessellate_surface_body(leaf).mesh
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_surface_hinge_paths_preserve_consumer_color_metadata() -> None:
    leaf = make_traditional_hinge_leaf(width=24.0, knuckle_count=5, color="#7f8fa6", backend="surface")
    pair = make_traditional_hinge_pair(
        width=24.0,
        knuckle_count=5,
        include_pin=True,
        leaf_a_color="#7f8fa6",
        leaf_b_color="#8f7f6a",
        pin_color="#b0b0b0",
        backend="surface",
    )
    living = make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8, color="#7d8f7a", backend="surface")
    bistable = make_bistable_hinge(width=40.0, preload_offset=2.0, color="#8e7a9c", backend="surface")

    assert leaf.consumer_metadata() == {"color": "#7f8fa6"}
    pair_collection = handoff_hinge_surface(pair)
    assert pair_collection.items[0].metadata["color"] == "#7f8fa6"
    assert pair_collection.items[1].metadata["color"] == "#8f7f6a"
    assert pair_collection.items[2].metadata["color"] == "#b0b0b0"
    assert handoff_hinge_surface(living).items[0].metadata["color"] == "#7d8f7a"
    assert handoff_hinge_surface(bistable).items[0].metadata["color"] == "#8e7a9c"


def test_surface_traditional_hinge_pair_handoff_is_deterministic_and_tessellates() -> None:
    assembly = make_traditional_hinge_pair(
        width=24.0,
        knuckle_count=5,
        include_pin=True,
        opened_angle_deg=32.0,
        backend="surface",
    )

    assert isinstance(assembly, HingeSurfaceAssembly)
    assert assembly.assembly_type == "traditional_hinge_pair"
    assert assembly.state == "opened"
    assert len(assembly.components) == 3
    assert assembly.stable_identity == make_traditional_hinge_pair(
        width=24.0,
        knuckle_count=5,
        include_pin=True,
        opened_angle_deg=32.0,
        backend="surface",
    ).stable_identity

    collection = handoff_hinge_surface(assembly, metadata={"fixture": "traditional"})
    assert isinstance(collection, SurfaceConsumerCollection)
    assert len(collection.items) == 3
    assert collection.metadata["assembly_type"] == "traditional_hinge_pair"
    mesh = _combined_collection_mesh(collection)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_surface_living_hinge_handoff_returns_structured_surface_collection() -> None:
    assembly = make_living_hinge(
        width=48.0,
        height=20.0,
        hinge_band_width=12.0,
        slit_pitch=1.8,
        backend="surface",
    )

    assert isinstance(assembly, HingeSurfaceAssembly)
    assert assembly.assembly_type == "living_hinge"
    assert assembly.state == "flat"
    assert assembly.components[0].payload["slot_count"] > 0

    collection = handoff_hinge_surface(assembly)
    assert len(collection.items) == 1
    mesh = _combined_collection_mesh(collection)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_surface_bistable_hinge_handoff_returns_structured_surface_collection() -> None:
    assembly = make_bistable_hinge(
        width=40.0,
        preload_offset=2.0,
        backend="surface",
    )

    assert isinstance(assembly, HingeSurfaceAssembly)
    assert assembly.assembly_type == "bistable_hinge"
    assert assembly.state == "preloaded"
    assert len(assembly.components) == 1

    collection = handoff_hinge_surface(assembly)
    assert len(collection.items) == 1
    mesh = _combined_collection_mesh(collection)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_handoff_hinge_surface_rejects_unknown_input() -> None:
    with pytest.raises(TypeError):
        handoff_hinge_surface(object())  # type: ignore[arg-type]
