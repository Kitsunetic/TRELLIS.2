from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from trellis2.modules.sparse import SparseTensor


def load_trimesh_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not meshes:
            raise ValueError(f"No triangle mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported control mesh type: {type(mesh)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Control mesh is empty: {path}")
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)


def build_signed_distance_grid(
    mesh: trimesh.Trimesh,
    resolution: int,
    bounds: Tuple[float, float] = (-0.5, 0.5),
    chunk_size: int = 16384,
) -> torch.Tensor:
    """
    Build an SDF grid in TRELLIS coordinates with inside < 0 and outside > 0.
    """
    lo, hi = bounds
    axis = np.linspace(lo, hi, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    signed_chunks = []
    try:
        import igl

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        igl_chunk_size = max(chunk_size, 262144)
        for start in range(0, points.shape[0], igl_chunk_size):
            pts = points[start : start + igl_chunk_size].astype(np.float64, copy=False)
            signed, _, _, _ = igl.signed_distance(
                pts,
                vertices,
                faces,
                igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
            )
            signed_chunks.append(signed.astype(np.float32, copy=False))
    except Exception:
        query = trimesh.proximity.ProximityQuery(mesh)
        for start in range(0, points.shape[0], chunk_size):
            pts = points[start : start + chunk_size]
            # trimesh uses positive-inside, negative-outside for watertight meshes.
            signed_chunks.append((-query.signed_distance(pts)).astype(np.float32))

    sdf_xyz = np.concatenate(signed_chunks, axis=0).reshape(resolution, resolution, resolution)
    sdf_zyx = torch.from_numpy(np.ascontiguousarray(sdf_xyz.transpose(2, 1, 0)))
    return sdf_zyx[None, None]


def sample_mesh_surface_np(mesh: trimesh.Trimesh, num_samples: int, offset: float) -> Tuple[torch.Tensor, torch.Tensor]:
    points, face_ids = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_ids]
    points_t = torch.from_numpy(points.astype(np.float32))
    normals_t = torch.from_numpy(normals.astype(np.float32))
    targets_t = points_t + offset * normals_t
    return points_t, targets_t


def query_sdf_grid(points: torch.Tensor, sdf_grid: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_zeros(points.shape[:-1])
    grid = (points * 2.0).view(1, -1, 1, 1, 3).clamp(-1.0, 1.0)
    sdf = F.grid_sample(
        sdf_grid.to(device=points.device, dtype=points.dtype),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sdf.view(-1)


def sample_torch_mesh_surface(vertices: torch.Tensor, faces: torch.Tensor, num_samples: int) -> torch.Tensor:
    if vertices.numel() == 0 or faces.numel() == 0 or num_samples <= 0:
        return vertices.new_zeros((0, 3))

    tri = vertices[faces.long()]
    areas = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1) * 0.5
    valid = areas > 1e-12
    if not valid.any():
        return vertices.new_zeros((0, 3))

    valid_idx = valid.nonzero(as_tuple=False).flatten()
    probs = areas[valid].detach()
    face_ids = valid_idx[torch.multinomial(probs, num_samples, replacement=True)]
    selected = tri[face_ids]

    uv = torch.rand(num_samples, 2, device=vertices.device, dtype=vertices.dtype)
    sqrt_u = torch.sqrt(uv[:, :1].clamp_min(1e-12))
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - uv[:, 1:2])
    w2 = sqrt_u * uv[:, 1:2]
    return w0 * selected[:, 0] + w1 * selected[:, 1] + w2 * selected[:, 2]


class LastSurfaceShapeGuidance:
    """
    Shoe-last shape-SLat guidance based on last SDF penetration and one-sided
    target coverage from offset last samples to generated surface samples.
    """

    def __init__(
        self,
        pipeline,
        spatial_control_path: str,
        last_offset: float = 0.025,
        last_sdf_resolution: int = 128,
        generated_samples: int = 2048,
        last_samples: int = 2048,
        lambda_pen: float = 10.0,
        lambda_cov: float = 1.0,
        sdf_chunk_size: int = 16384,
    ):
        self.pipeline = pipeline
        self.spatial_control_path = spatial_control_path
        self.decoder = pipeline.models["shape_slat_decoder"]
        self.shape_slat_normalization = pipeline.shape_slat_normalization
        self.last_offset = last_offset
        self.generated_samples = generated_samples
        self.last_samples = last_samples
        self.lambda_pen = lambda_pen
        self.lambda_cov = lambda_cov
        self._resolution: Optional[int] = None

        mesh = load_trimesh_mesh(spatial_control_path)
        self.last_mesh = mesh
        self.sdf_grid = build_signed_distance_grid(mesh, last_sdf_resolution, chunk_size=sdf_chunk_size)
        self.last_surface_samples, self.last_targets = sample_mesh_surface_np(mesh, last_samples, last_offset)

    def set_resolution(self, resolution: int) -> None:
        self._resolution = resolution

    def _unnormalize_shape_slat(self, pred_x0: SparseTensor) -> SparseTensor:
        std = torch.tensor(self.shape_slat_normalization["std"], device=pred_x0.device, dtype=pred_x0.dtype)[None]
        mean = torch.tensor(self.shape_slat_normalization["mean"], device=pred_x0.device, dtype=pred_x0.dtype)[None]
        return pred_x0 * std + mean

    def _decode_meshes(self, pred_x0: SparseTensor, resolution: Optional[int]) -> List:
        if resolution is not None:
            self.decoder.set_resolution(resolution)
        elif self._resolution is not None:
            self.decoder.set_resolution(self._resolution)
        slat = self._unnormalize_shape_slat(pred_x0)
        return self.decoder(slat)

    def points_loss(self, generated_points: torch.Tensor, t: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        if generated_points.numel() == 0:
            zero = generated_points.sum() * 0.0
            return zero, {
                "guidance_loss": 0.0,
                "penetration_loss": 0.0,
                "coverage_loss": 0.0,
                "t": float(t),
            }

        sdf_values = query_sdf_grid(generated_points, self.sdf_grid)
        penetration_loss = F.relu(-sdf_values).square().mean()

        targets = self.last_targets.to(device=generated_points.device, dtype=generated_points.dtype)
        distances = torch.cdist(targets, generated_points)
        coverage_loss = distances.square().min(dim=1).values.mean()
        loss = self.lambda_pen * penetration_loss + self.lambda_cov * coverage_loss
        return loss, {
            "guidance_loss": float(loss.detach().item()),
            "penetration_loss": float(penetration_loss.detach().item()),
            "coverage_loss": float(coverage_loss.detach().item()),
            "t": float(t),
        }

    def compute_loss(
        self,
        pred_x0: SparseTensor,
        t: float,
        resolution: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        meshes = self._decode_meshes(pred_x0, resolution)
        losses = []
        metrics: List[Dict[str, float]] = []
        for mesh in meshes:
            points = sample_torch_mesh_surface(mesh.vertices, mesh.faces, self.generated_samples)
            if points.numel() == 0:
                loss = pred_x0.feats.sum() * 0.0
                metric = {
                    "guidance_loss": 0.0,
                    "penetration_loss": 0.0,
                    "coverage_loss": 0.0,
                    "t": float(t),
                }
            else:
                loss, metric = self.points_loss(points, t)
            losses.append(loss)
            metrics.append(metric)

        if not losses:
            zero = pred_x0.feats.sum() * 0.0
            return zero, {
                "guidance_loss": 0.0,
                "penetration_loss": 0.0,
                "coverage_loss": 0.0,
                "t": float(t),
            }

        loss = torch.stack(losses).mean()
        return loss, {
            "guidance_loss": float(loss.detach().item()),
            "penetration_loss": float(np.mean([m["penetration_loss"] for m in metrics])),
            "coverage_loss": float(np.mean([m["coverage_loss"] for m in metrics])),
            "t": float(t),
        }


def build_shape_slat_guidance(guidance_type: str, pipeline, spatial_control_path: str, **kwargs):
    guidance_type = guidance_type.lower()
    if guidance_type == "last_surface":
        return LastSurfaceShapeGuidance(
            pipeline,
            spatial_control_path,
            last_offset=kwargs.get("last_offset", 0.025),
            last_sdf_resolution=kwargs.get("last_sdf_resolution", 128),
            generated_samples=kwargs.get("generated_samples", 2048),
            last_samples=kwargs.get("last_samples", 2048),
            lambda_pen=kwargs.get("lambda_pen", 10.0),
            lambda_cov=kwargs.get("lambda_cov", 1.0),
            sdf_chunk_size=kwargs.get("sdf_chunk_size", 262144),
        )
    raise ValueError(f"Unsupported shape SLat guidance type: {guidance_type}")
