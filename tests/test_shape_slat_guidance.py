import torch
import trimesh

from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines.guidance.shape_slat_guidance import (
    LastSurfaceShapeGuidance,
    build_signed_distance_grid,
    query_sdf_grid,
)


def test_sdf_grid_uses_negative_inside_positive_outside():
    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    sdf_grid = build_signed_distance_grid(mesh, resolution=32)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.45, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sdf = query_sdf_grid(points, sdf_grid)

    assert sdf[0] < 0
    assert sdf[1] > 0
    assert sdf[2].abs() < 0.02


def test_last_surface_loss_penetration_and_one_sided_coverage():
    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    guidance = object.__new__(LastSurfaceShapeGuidance)
    guidance.sdf_grid = build_signed_distance_grid(mesh, resolution=32)
    guidance.last_targets = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)
    guidance.lambda_pen = 10.0
    guidance.lambda_cov = 1.0

    inside = torch.tensor([[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]], dtype=torch.float32)
    outside = torch.tensor([[0.40, 0.0, 0.0], [0.45, 0.0, 0.0]], dtype=torch.float32)

    inside_loss, inside_metrics = guidance.points_loss(inside)
    outside_loss, outside_metrics = guidance.points_loss(outside)

    assert inside_metrics["penetration_loss"] > 0
    assert outside_metrics["penetration_loss"] == 0
    assert outside_metrics["coverage_loss"] == 0
    assert inside_loss > outside_loss

    extra_generated_points = torch.cat([outside, torch.tensor([[0.49, 0.0, 0.0]], dtype=torch.float32)], dim=0)
    _, extra_metrics = guidance.points_loss(extra_generated_points)
    assert extra_metrics["coverage_loss"] == 0


def test_shape_guidance_loss_backpropagates_to_sparse_features():
    class DummyMesh:
        def __init__(self, vertices):
            self.vertices = vertices
            self.faces = torch.tensor([[0, 1, 2]], device=vertices.device, dtype=torch.int64)

    class DummyDecoder:
        def set_resolution(self, resolution):
            self.resolution = resolution

        def __call__(self, slat):
            return [DummyMesh(slat.feats[:, :3])]

    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    guidance = object.__new__(LastSurfaceShapeGuidance)
    guidance.decoder = DummyDecoder()
    guidance.shape_slat_normalization = {"std": [1.0, 1.0, 1.0], "mean": [0.0, 0.0, 0.0]}
    guidance.sdf_grid = build_signed_distance_grid(mesh, resolution=24)
    guidance.last_targets = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)
    guidance.generated_samples = 16
    guidance.lambda_pen = 10.0
    guidance.lambda_cov = 1.0
    guidance._resolution = None

    feats = torch.tensor(
        [
            [0.40, 0.00, 0.00],
            [0.45, 0.05, 0.00],
            [0.45, 0.00, 0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.int32)
    slat = SparseTensor(feats=feats, coords=coords)

    loss, metrics = guidance.compute_loss(slat, t=0.5, resolution=512)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["guidance_loss"] >= 0
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
