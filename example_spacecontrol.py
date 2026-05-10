import os
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image

from trellis2.pipelines import SpaceControlPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import aggressive_repair_mesh, mesh_topology_stats, render_utils, smooth_mesh_taubin, solidify_mesh_with_control, to_glb_z_up


def ensure_model_and_patch(repo_id="microsoft/TRELLIS.2-4B", local_dir="results/TRELLIS.2-4B"):
    """모델이 없으면 다운로드하고, SpaceControl에 필요한 인코더 설정을 자동 패치합니다."""
    print(f"Checking model directory: {local_dir}")

    # 1. 모델 다운로드 (이미 있으면 캐시 확인 후 빠르게 넘어감)
    if not os.path.exists(local_dir) or not os.path.exists(os.path.join(local_dir, "pipeline.json")):
        print("Downloading TRELLIS.2-4B model...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

    # 2. pipeline.json 패치
    json_path = os.path.join(local_dir, "pipeline.json")
    with open(json_path, "r") as f:
        config = json.load(f)

    models_dict = config.get("args", {}).get("models", {})
    if "sparse_structure_encoder" not in models_dict:
        print("Patching pipeline.json: Injecting missing sparse_structure_encoder...")
        models_dict["sparse_structure_encoder"] = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"

        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)
    else:
        print("Model configuration is already patched for SpaceControl.")

    return local_dir


def resolve_padding(args) -> np.ndarray:
    padding = np.array([args.normalize_padding, args.normalize_padding, args.normalize_padding], dtype=np.float64)
    if args.normalize_padding_x is not None:
        padding[0] = args.normalize_padding_x
    if args.normalize_padding_y is not None:
        padding[1] = args.normalize_padding_y
    if args.normalize_padding_z is not None:
        padding[2] = args.normalize_padding_z
    fill_ratio = 1.0 - 2.0 * padding
    if np.any(fill_ratio <= 0.0) or np.any(fill_ratio > 1.0):
        raise ValueError(f"normalize padding must keep per-axis fill ratios in (0, 1], got padding={padding.tolist()}")
    return padding


def normalize_control_mesh(mesh_path: str, padding: np.ndarray, out_dir: str) -> str:
    source_path = Path(mesh_path)
    normalized_path = Path(out_dir) / f"{source_path.stem}_spacecontrol_normalized{source_path.suffix}"

    mesh = o3d.io.read_triangle_mesh(str(source_path))
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    center = (min_bound + max_bound) / 2
    max_extent = (max_bound - min_bound).max()
    if max_extent <= 0:
        raise ValueError(f"Control mesh has invalid extent: {mesh_path}")

    scales = (1.0 - 2.0 * padding) / max_extent
    vertices = np.asarray(mesh.vertices)
    vertices = (vertices - center) * scales[None, :]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(str(normalized_path), mesh)
    return str(normalized_path)


def make_vis_frames(result):
    if "shaded" in result:
        return render_utils.make_pbr_vis_frames(result)
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
    # 0. 자동 다운로드 및 패치 실행
    local_model_path = ensure_model_and_patch()
    os.makedirs(args.out_dir, exist_ok=True)

    spatial_control_path = args.control
    if args.normalize_control:
        padding = resolve_padding(args)
        spatial_control_path = normalize_control_mesh(args.control, padding, args.out_dir)

    # 1. Setup Environment Map
    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        )
    )

    # 2. Load Pipeline
    print("Loading Trellis2 SpaceControl Pipeline...")
    pipeline = SpaceControlPipeline.from_pretrained(local_model_path)
    pipeline.cuda()

    # 3. Load Image & Spatial Control Shape
    print(f"Loading image from: {args.image}")
    image = Image.open(args.image)

    print(f"Using spatial control mesh from: {spatial_control_path}")

    # 4. Run Pipeline with SpaceControl
    print(f"Generating 3D model with control_mode={args.control_mode}, tau={args.tau}...")
    sparse_structure_sampler_params = {
        "spatial_control_mesh_path": spatial_control_path,
        "control_mode": args.control_mode,
        "guidance_type": args.guidance_type,
        "space_control_tau": args.tau,
    }
    if args.steps is not None:
        sparse_structure_sampler_params["steps"] = args.steps
    if args.guidance_strength is not None:
        sparse_structure_sampler_params["guidance_strength"] = args.guidance_strength
    if args.guidance_interval is not None:
        sparse_structure_sampler_params["guidance_interval"] = tuple(args.guidance_interval)
    if args.guidance_rescale is not None:
        sparse_structure_sampler_params["guidance_rescale"] = args.guidance_rescale
    if args.rescale_t is not None:
        sparse_structure_sampler_params["rescale_t"] = args.rescale_t
    if args.geometry_guidance_strength is not None:
        sparse_structure_sampler_params["geometry_guidance_strength"] = args.geometry_guidance_strength
    if args.geometry_guidance_interval is not None:
        sparse_structure_sampler_params["geometry_guidance_interval"] = tuple(args.geometry_guidance_interval)
    if args.geometry_guidance_schedule is not None:
        sparse_structure_sampler_params["geometry_guidance_schedule"] = args.geometry_guidance_schedule
    if args.geometry_guidance_rescale is not None:
        sparse_structure_sampler_params["geometry_guidance_rescale"] = args.geometry_guidance_rescale
    if args.geometry_grad_clip is not None:
        sparse_structure_sampler_params["geometry_grad_clip"] = args.geometry_grad_clip
    if args.geometry_guidance_cfg_mode is not None:
        sparse_structure_sampler_params["geometry_guidance_cfg_mode"] = args.geometry_guidance_cfg_mode

    for key in (
        "bce_weight",
        "dice_weight",
        "envelope_radius",
        "interior_weight",
        "contain_weight",
        "outside_weight",
        "shell_weight",
        "bottom_weight",
        "bottom_band_ratio",
        "bottom_outer_margin",
    ):
        value = getattr(args, key)
        if value is not None:
            sparse_structure_sampler_params[key] = value

    mesh = pipeline.run(
        image=image,
        sparse_structure_sampler_params=sparse_structure_sampler_params,
    )[0]

    mesh.simplify(16777216)  # nvdiffrast limit
    topology_before = mesh_topology_stats(mesh)
    topology_solidified = None
    if args.solidify_volume:
        mesh = solidify_mesh_with_control(
            mesh,
            control_mesh_path=spatial_control_path if args.solidify_include_control else None,
            resolution=args.solidify_resolution,
            generated_surface_dilation=args.solidify_generated_surface_dilation,
            control_surface_dilation=args.solidify_control_surface_dilation,
            closing_iters=args.solidify_closing_iters,
            final_closing_iters=args.solidify_final_closing_iters,
        )
        topology_solidified = mesh_topology_stats(mesh)
    if args.make_watertight:
        aggressive_repair_mesh(
            mesh,
            max_hole_perimeter=args.watertight_max_hole_perimeter,
            min_component_area=args.watertight_min_component_area,
        )
    elif args.fill_holes_max_hole_perimeter > 0:
        mesh.fill_holes(max_hole_perimeter=args.fill_holes_max_hole_perimeter)
    if args.solidify_smoothing_iters > 0:
        mesh = smooth_mesh_taubin(mesh, iterations=args.solidify_smoothing_iters)
    topology_after = mesh_topology_stats(mesh)

    # ---------------------------------------------------------
    # [추가됨] 출력 디렉토리 생성 및 파일명 자동 구성
    # ---------------------------------------------------------
    # 입력 이미지 경로에서 파일명만 추출 (예: 'assets/shoe2.jpg' -> 'shoe2')
    image_basename = os.path.splitext(os.path.basename(args.image))[0]

    video_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}.mp4")
    mesh_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}.glb")
    metrics_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}-guidance.json")
    topology_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}-topology.json")
    # ---------------------------------------------------------

    # 5. Render Video
    print(f"Rendering video to {video_out}...")
    if hasattr(mesh, "attrs") and getattr(mesh, "attrs", None) is not None:
        render_ret = render_utils.render_video(mesh, envmap=envmap)
    else:
        render_ret = render_utils.render_video(mesh)
    video = make_vis_frames(render_ret)
    imageio.mimsave(video_out, video, fps=15)

    # 6. Export to GLB
    print(f"Exporting model to {mesh_out}...")
    if hasattr(mesh, "attrs") and getattr(mesh, "attrs", None) is not None:
        glb = to_glb_z_up(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(mesh_out, extension_webp=True)
    else:
        trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(mesh_out, file_type="glb")
    if pipeline.last_guidance_metrics is not None:
        with open(metrics_out, "w") as f:
            json.dump(pipeline.last_guidance_metrics, f, indent=2)
        print(f"Saved guidance metrics to {metrics_out}")
    with open(topology_out, "w") as f:
        payload = {"before": topology_before}
        if topology_solidified is not None:
            payload["solidified"] = topology_solidified
        payload["after"] = topology_after
        json.dump(payload, f, indent=2)
    print(f"Saved topology stats to {topology_out}")
    print("✨ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaceControl Trellis2 Pipeline")

    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--control", type=str, required=True, help="Path to the spatial control mesh")
    parser.add_argument("--tau", type=int, default=6, help="Strength of spatial control (typically 1~10, default 6)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save the generated files")
    parser.add_argument("--control_mode", type=str, default="spacecontrol", choices=["none", "spacecontrol", "guidance", "both"])
    parser.add_argument("--guidance_type", type=str, default="containment", choices=["latent", "occupancy", "containment", "shell"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rescale_t", type=float, default=None)
    parser.add_argument("--guidance_strength", type=float, default=None)
    parser.add_argument("--guidance_interval", type=float, nargs=2, default=None)
    parser.add_argument("--guidance_rescale", type=float, default=None)
    parser.add_argument("--geometry_guidance_strength", type=float, default=1.0)
    parser.add_argument("--geometry_guidance_interval", type=float, nargs=2, default=[0.5, 0.95])
    parser.add_argument("--geometry_guidance_schedule", type=str, default="constant", choices=["constant", "linear_decay", "linear_rise"])
    parser.add_argument("--geometry_guidance_rescale", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--geometry_grad_clip", type=float, default=5.0)
    parser.add_argument("--geometry_guidance_cfg_mode", type=str, default="cond_only", choices=["cond_only", "with_cfg"])
    parser.add_argument("--bce_weight", type=float, default=None)
    parser.add_argument("--dice_weight", type=float, default=None)
    parser.add_argument("--envelope_radius", type=int, default=2)
    parser.add_argument("--interior_weight", type=float, default=0.1)
    parser.add_argument("--contain_weight", type=float, default=2.0)
    parser.add_argument("--outside_weight", type=float, default=2.5)
    parser.add_argument("--shell_weight", type=float, default=0.0)
    parser.add_argument("--bottom_weight", type=float, default=1.0)
    parser.add_argument("--bottom_band_ratio", type=float, default=0.18)
    parser.add_argument("--bottom_outer_margin", type=int, default=1)
    parser.add_argument("--normalize_control", action="store_true", help="Renormalize the control mesh into the unit cube")
    parser.add_argument("--normalize_padding", type=float, default=0.0)
    parser.add_argument("--normalize_padding_x", type=float, default=0.02)
    parser.add_argument("--normalize_padding_y", type=float, default=0.0)
    parser.add_argument("--normalize_padding_z", type=float, default=0.0)
    parser.add_argument("--fill_holes_max_hole_perimeter", type=float, default=0.0)
    parser.add_argument("--solidify_volume", action="store_true")
    parser.add_argument("--solidify_include_control", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--solidify_resolution", type=int, default=128)
    parser.add_argument("--solidify_generated_surface_dilation", type=int, default=2)
    parser.add_argument("--solidify_control_surface_dilation", type=int, default=2)
    parser.add_argument("--solidify_closing_iters", type=int, default=2)
    parser.add_argument("--solidify_final_closing_iters", type=int, default=1)
    parser.add_argument("--solidify_smoothing_iters", type=int, default=0)
    parser.add_argument("--make_watertight", action="store_true")
    parser.add_argument("--watertight_max_hole_perimeter", type=float, default=10.0)
    parser.add_argument("--watertight_min_component_area", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
