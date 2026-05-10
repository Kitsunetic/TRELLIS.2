from __future__ import annotations

from typing import Dict

import cumesh


def mesh_topology_stats(mesh) -> Dict[str, int]:
    vertices = mesh.vertices.contiguous().cuda()
    faces = mesh.faces.contiguous().cuda()

    cu_mesh = cumesh.CuMesh()
    cu_mesh.init(vertices, faces)
    cu_mesh.get_edges()
    cu_mesh.get_boundary_info()
    cu_mesh.get_connected_components()
    cu_mesh.get_boundary_connected_components()
    cu_mesh.get_boundary_loops()

    return {
        "num_vertices": int(cu_mesh.num_vertices),
        "num_faces": int(cu_mesh.num_faces),
        "num_edges": int(cu_mesh.num_edges),
        "num_boundaries": int(cu_mesh.num_boundaries),
        "num_connected_components": int(cu_mesh.num_conneted_components),
        "num_boundary_connected_components": int(cu_mesh.num_boundary_conneted_components),
        "num_boundary_loops": int(cu_mesh.num_boundary_loops),
    }


def aggressive_repair_mesh(
    mesh,
    *,
    max_hole_perimeter: float = 10.0,
    min_component_area: float = 1e-4,
) -> None:
    """
    Aggressively bias a mesh toward one solid component.

    This is intentionally stronger than the default light cleanup because the
    shoe-last workflow tolerates a filled interior if that yields a watertight
    exterior.
    """
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(min_component_area)
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)

    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(min_component_area)
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    mesh.unify_face_orientations()
