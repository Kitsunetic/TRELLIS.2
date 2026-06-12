import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch
import trimesh

from trellis2.representations import Mesh
from trellis2.utils import mesh_topology_stats, render_utils, smooth_mesh_taubin, solidify_mesh_with_sdf


def load_mesh(path: str) -> Mesh:
    loaded = trimesh.load(path, force="scene")
    if hasattr(loaded, "geometry"):
        meshes = [geom for geom in loaded.geometry.values() if hasattr(geom, "vertices") and hasattr(geom, "faces")]
        if len(meshes) == 0:
            raise ValueError(f"No mesh geometry found in {path}")
        tri_mesh = trimesh.util.concatenate(meshes)
    else:
        tri_mesh = loaded
    return Mesh(
        torch.from_numpy(tri_mesh.vertices.astype("float32")),
        torch.from_numpy(tri_mesh.faces.astype("int32")),
    )


def export_mesh(mesh: Mesh, path: str) -> None:
    tri_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    tri_mesh.export(path, file_type=Path(path).suffix.lstrip("."))


def make_vis_frames(result):
    if "normal" in result:
        frames = []
        masks = result.get("mask")
        for i, normal in enumerate(result["normal"]):
            frame = normal.copy()
            if masks is not None:
                frame = np.where(masks[i] > 0, frame, 0)
            frames.append(frame)
        return frames
    if "color" in result:
        return result["color"]
    raise ValueError(f"Unsupported render output keys: {sorted(result.keys())}")


def main(args):
    mesh = load_mesh(args.input)
    before = mesh_topology_stats(mesh)
    mesh = solidify_mesh_with_sdf(
        mesh,
        control_mesh_path=args.control,
        resolution=args.resolution,
        generated_surface_thickness=args.generated_surface_thickness,
        control_offset=args.control_offset,
        smoothing_sigma=args.smoothing_sigma,
        sdf_mode=args.sdf_mode,
        chunk_size=args.chunk_size,
        domain_margin=args.domain_margin,
    )
    solidified = mesh_topology_stats(mesh)
    if args.smoothing_iters > 0:
        mesh = smooth_mesh_taubin(mesh, iterations=args.smoothing_iters)
    after = mesh_topology_stats(mesh)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_mesh(mesh, args.output)
    if args.topology_output is not None:
        Path(args.topology_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.topology_output, "w") as f:
            json.dump({"before": before, "solidified": solidified, "after": after}, f, indent=2)
    if args.video_output is not None or args.frame_output is not None:
        video = make_vis_frames(render_utils.render_video(mesh.cuda()))
        if args.video_output is not None:
            Path(args.video_output).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.video_output, video, fps=15)
        if args.frame_output is not None:
            Path(args.frame_output).parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.frame_output, video[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply SDF solidify to an existing mesh/GLB")
    parser.add_argument("--input", required=True)
    parser.add_argument("--control", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topology_output", default=None)
    parser.add_argument("--video_output", default=None)
    parser.add_argument("--frame_output", default=None)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--generated_surface_thickness", type=float, default=1.5)
    parser.add_argument("--control_offset", type=float, default=1.0)
    parser.add_argument("--smoothing_sigma", type=float, default=0.75)
    parser.add_argument("--sdf_mode", type=str, default="raystab", choices=["watertight", "raystab"])
    parser.add_argument("--chunk_size", type=int, default=1048576)
    parser.add_argument("--domain_margin", type=float, default=0.03)
    parser.add_argument("--smoothing_iters", type=int, default=10)
    main(parser.parse_args())
