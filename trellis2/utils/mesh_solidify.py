from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from scipy import ndimage
from skimage import measure

from trellis2.representations import Mesh


def _voxelize_o3d_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    resolution: int,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> np.ndarray:
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / resolution,
        min_bound=min_bound,
        max_bound=max_bound,
    )
    volume = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int32)
    if coords.size == 0:
        return volume
    coords = np.clip(coords, 0, resolution - 1)
    volume[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return volume


def _mesh_to_o3d(mesh: Mesh) -> o3d.geometry.TriangleMesh:
    tri_mesh = o3d.geometry.TriangleMesh()
    tri_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.detach().cpu().numpy())
    tri_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.detach().cpu().numpy().astype(np.int32))
    return tri_mesh


def _load_control_mesh(path: str) -> o3d.geometry.TriangleMesh:
    tri_mesh = o3d.io.read_triangle_mesh(str(Path(path)))
    vertices = np.asarray(tri_mesh.vertices)
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    tri_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return tri_mesh


def smooth_mesh_taubin(mesh: Mesh, *, iterations: int = 10) -> Mesh:
    if iterations <= 0:
        return mesh
    tri_mesh = _mesh_to_o3d(mesh)
    tri_mesh = tri_mesh.filter_smooth_taubin(number_of_iterations=iterations)
    tri_mesh.compute_vertex_normals()
    return Mesh(
        torch.from_numpy(np.asarray(tri_mesh.vertices).astype(np.float32)),
        torch.from_numpy(np.asarray(tri_mesh.triangles).astype(np.int32)),
    ).to(mesh.device)


def _largest_component(volume: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, num_labels = ndimage.label(volume, structure=structure)
    if num_labels <= 1:
        return volume
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def solidify_mesh_with_control(
    mesh: Mesh,
    *,
    control_mesh_path: Optional[str] = None,
    resolution: int = 128,
    generated_surface_dilation: int = 2,
    control_surface_dilation: int = 2,
    closing_iters: int = 2,
    final_closing_iters: int = 1,
) -> Mesh:
    """
    Rebuild a generated shoe mesh as a single solid volume.

    This path is intentionally topology-biased. The user has explicitly allowed
    the last to be fully embedded inside the shoe, so the goal here is one
    watertight component rather than preserving an empty interior cavity.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    min_bound = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    max_bound = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    structure = np.ones((3, 3, 3), dtype=bool)

    generated = _voxelize_o3d_mesh(
        _mesh_to_o3d(mesh),
        resolution=resolution,
        min_bound=min_bound,
        max_bound=max_bound,
    )
    if generated_surface_dilation > 0:
        generated = ndimage.binary_dilation(generated, structure=structure, iterations=generated_surface_dilation)

    volume = generated.copy()

    if control_mesh_path is not None:
        control = _voxelize_o3d_mesh(
            _load_control_mesh(control_mesh_path),
            resolution=resolution,
            min_bound=min_bound,
            max_bound=max_bound,
        )
        control = ndimage.binary_fill_holes(control)
        if control_surface_dilation > 0:
            control = ndimage.binary_dilation(control, structure=structure, iterations=control_surface_dilation)
        volume |= control

    if closing_iters > 0:
        volume = ndimage.binary_closing(volume, structure=structure, iterations=closing_iters)
    volume = ndimage.binary_fill_holes(volume)
    volume = _largest_component(volume)
    if final_closing_iters > 0:
        volume = ndimage.binary_closing(volume, structure=structure, iterations=final_closing_iters)
    volume = ndimage.binary_fill_holes(volume)

    verts, faces, _normals, _values = measure.marching_cubes(
        volume.astype(np.float32),
        level=0.5,
        spacing=(1.0 / resolution, 1.0 / resolution, 1.0 / resolution),
    )
    verts = verts - 0.5

    return Mesh(
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int32)),
    ).to(mesh.device)
