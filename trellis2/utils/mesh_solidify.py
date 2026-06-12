from __future__ import annotations

from pathlib import Path
from typing import Optional

import cumesh
import numpy as np
import open3d as o3d
import torch
import trimesh
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


def _keep_largest_mesh_component(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = tri_mesh.split(only_watertight=False)
    if len(components) <= 1:
        return vertices, faces
    largest = max(components, key=lambda component: component.area)
    return np.asarray(largest.vertices), np.asarray(largest.faces)


def _sample_bvh_distance_grid(
    bvh,
    *,
    resolution: int,
    min_bound: float,
    scale: float,
    signed: bool,
    sdf_mode: str,
    chunk_size: int,
) -> np.ndarray:
    total = resolution**3
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    field = np.empty(total, dtype=np.float32)
    denom = float(resolution - 1)
    device = torch.device("cuda")
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        flat = torch.arange(start, end, device=device)
        z = flat % resolution
        y = (flat // resolution) % resolution
        x = flat // (resolution * resolution)
        points = torch.stack((x, y, z), dim=-1).float()
        points = points / denom * scale + min_bound
        if signed:
            distances = bvh.signed_distance(points, mode=sdf_mode)[0]
        else:
            distances = bvh.unsigned_distance(points)[0]
        field[start:end] = distances.detach().cpu().numpy()
    return field.reshape((resolution, resolution, resolution))


def solidify_mesh_with_sdf(
    mesh: Mesh,
    *,
    control_mesh_path: Optional[str] = None,
    resolution: int = 192,
    generated_surface_thickness: float = 1.5,
    control_offset: float = 1.0,
    smoothing_sigma: float = 0.75,
    sdf_mode: str = "raystab",
    chunk_size: int = 1048576,
    domain_margin: float = 0.03,
    keep_largest_component: bool = True,
) -> Mesh:
    """
    Rebuild a mesh from a dense distance field instead of a binary voxel volume.

    The generated mesh is often open, so it is used as a smooth unsigned-distance
    band. The control last is unioned as a signed-distance solid, matching the
    embedded-last requirement while reducing blocky voxel stair-stepping.
    """
    if resolution <= 1:
        raise ValueError(f"resolution must be greater than 1, got {resolution}")
    if sdf_mode not in {"watertight", "raystab"}:
        raise ValueError(f"sdf_mode must be 'watertight' or 'raystab', got {sdf_mode!r}")
    if domain_margin < 0:
        raise ValueError(f"domain_margin must be non-negative, got {domain_margin}")

    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    generated_bvh = cumesh.cuBVH(vertices, faces)
    min_bound = -0.5 - domain_margin
    scale = 1.0 + 2.0 * domain_margin
    voxel_size = scale / float(resolution - 1)
    field = _sample_bvh_distance_grid(
        generated_bvh,
        resolution=resolution,
        min_bound=min_bound,
        scale=scale,
        signed=False,
        sdf_mode=sdf_mode,
        chunk_size=chunk_size,
    )
    field -= generated_surface_thickness * voxel_size

    if control_mesh_path is not None:
        control_mesh = _load_control_mesh(control_mesh_path)
        control_vertices = np.asarray(control_mesh.vertices).astype(np.float32)
        control_faces = np.asarray(control_mesh.triangles).astype(np.int32)
        control_bvh = cumesh.cuBVH(control_vertices, control_faces)
        control_field = _sample_bvh_distance_grid(
            control_bvh,
            resolution=resolution,
            min_bound=min_bound,
            scale=scale,
            signed=True,
            sdf_mode=sdf_mode,
            chunk_size=chunk_size,
        )
        control_field -= control_offset * voxel_size
        field = np.minimum(field, control_field)

    if smoothing_sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=smoothing_sigma)

    verts, faces, _normals, _values = measure.marching_cubes(
        field,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
    )
    verts = verts + min_bound
    if keep_largest_component:
        verts, faces = _keep_largest_mesh_component(verts, faces)

    return Mesh(
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int32)),
    ).to(mesh.device)


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
