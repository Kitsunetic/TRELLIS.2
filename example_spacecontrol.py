import argparse
import json
import os

import cv2
import imageio
import numpy as np
import o_voxel
import torch
import utils3d
from huggingface_hub import snapshot_download
from PIL import Image

from trellis2.pipelines import SpaceControlPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory


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


def render_under_view(mesh, envmap, resolution=1024, radius=2.0, fov=40.0):
    renderer = render_utils.get_renderer(mesh, resolution=resolution, near=1, far=100, ssaa=2)
    eye = torch.tensor([0.0, 0.0, -float(radius)], dtype=torch.float32, device="cuda")
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")
    extrinsics = utils3d.torch.extrinsics_look_at(eye, target, up)
    fov_rad = torch.deg2rad(torch.tensor(float(fov), dtype=torch.float32, device="cuda"))
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)
    render = renderer.render(mesh, extrinsics, intrinsics, envmap=envmap)
    image = render["shaded"].detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def render_named_view(mesh, envmap, eye_xyz, up_xyz, resolution=1024, fov=40.0):
    renderer = render_utils.get_renderer(mesh, resolution=resolution, near=1, far=100, ssaa=2)
    eye = torch.tensor(eye_xyz, dtype=torch.float32, device="cuda")
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    up = torch.tensor(up_xyz, dtype=torch.float32, device="cuda")
    extrinsics = utils3d.torch.extrinsics_look_at(eye, target, up)
    fov_rad = torch.deg2rad(torch.tensor(float(fov), dtype=torch.float32, device="cuda"))
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)
    render = renderer.render(mesh, extrinsics, intrinsics, envmap=envmap)
    image = render["shaded"].detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def render_canonical_views(mesh, envmap, resolution=1024, radius=2.0, fov=40.0):
    views = {
        "top": {"eye": [0.0, 0.0, float(radius)], "up": [0.0, 1.0, 0.0]},
        "bottom": {"eye": [0.0, 0.0, -float(radius)], "up": [0.0, 1.0, 0.0]},
        "left": {"eye": [-float(radius), 0.0, 0.0], "up": [0.0, 0.0, 1.0]},
        "right": {"eye": [float(radius), 0.0, 0.0], "up": [0.0, 0.0, 1.0]},
    }
    rendered = {}
    for name, camera in views.items():
        rendered[name] = render_named_view(
            mesh,
            envmap=envmap,
            eye_xyz=camera["eye"],
            up_xyz=camera["up"],
            resolution=resolution,
            fov=fov,
        )
    return rendered


def main(args):
    # 0. 자동 다운로드 및 패치 실행
    local_model_path = ensure_model_and_patch()

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

    print(f"Using spatial control mesh from: {args.control}")
    spatial_control_path = args.control

    # 4. Run Pipeline with SpaceControl
    print(f"Generating 3D model with tau={args.tau}...")
    mesh = pipeline.run(
        image=image,
        seed=args.seed,
        sparse_structure_sampler_params={
            "spatial_control_mesh_path": spatial_control_path,
            "space_control_tau": args.tau,
        },
    )[0]

    mesh.simplify(16777216)  # nvdiffrast limit

    # ---------------------------------------------------------
    # [추가됨] 출력 디렉토리 생성 및 파일명 자동 구성
    # ---------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # 입력 이미지 경로에서 파일명만 추출 (예: 'assets/shoe2.jpg' -> 'shoe2')
    image_basename = os.path.splitext(os.path.basename(args.image))[0]
    output_basename = args.name or f"{image_basename}-tau{args.tau}-seed{args.seed}"

    video_out = os.path.join(args.out_dir, f"{output_basename}.mp4")
    mesh_out = os.path.join(args.out_dir, f"{output_basename}.glb")
    # ---------------------------------------------------------

    if args.under_view_out:
        print(f"Rendering under-view image to {args.under_view_out}...")
        os.makedirs(os.path.dirname(args.under_view_out), exist_ok=True)
        under_view = render_under_view(
            mesh,
            envmap=envmap,
            resolution=args.under_view_resolution,
            radius=args.under_view_radius,
            fov=args.under_view_fov,
        )
        Image.fromarray(under_view).save(args.under_view_out)

    if args.view_dir_out:
        print(f"Rendering canonical views to {args.view_dir_out}...")
        os.makedirs(args.view_dir_out, exist_ok=True)
        view_images = render_canonical_views(
            mesh,
            envmap=envmap,
            resolution=args.view_resolution,
            radius=args.view_radius,
            fov=args.view_fov,
        )
        for view_name, view_image in view_images.items():
            view_out = os.path.join(args.view_dir_out, f"{output_basename}-{view_name}.png")
            Image.fromarray(view_image).save(view_out)

    if not args.skip_video:
        print(f"Rendering video to {video_out}...")
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(video_out, video, fps=15)

    if not args.skip_glb:
        print(f"Exporting model to {mesh_out}...")
        glb = o_voxel.postprocess.to_glb(
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
    print("✨ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaceControl Trellis2 Pipeline")

    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--control", type=str, required=True, help="Path to the spatial control mesh")
    parser.add_argument("--tau", type=int, default=6, help="Strength of spatial control (typically 1~10, default 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save the generated files")
    parser.add_argument("--name", type=str, default=None, help="Optional basename for outputs")
    parser.add_argument("--skip_video", action="store_true", help="Skip video rendering")
    parser.add_argument("--skip_glb", action="store_true", help="Skip GLB export")
    parser.add_argument("--under_view_out", type=str, default=None, help="Optional output path for an underside render")
    parser.add_argument("--under_view_resolution", type=int, default=1024, help="Resolution for underside render")
    parser.add_argument("--under_view_radius", type=float, default=2.0, help="Camera radius for underside render")
    parser.add_argument("--under_view_fov", type=float, default=40.0, help="Field of view for underside render")
    parser.add_argument("--view_dir_out", type=str, default=None, help="Optional directory for top/bottom/left/right renders")
    parser.add_argument("--view_resolution", type=int, default=1024, help="Resolution for canonical view renders")
    parser.add_argument("--view_radius", type=float, default=2.0, help="Camera radius for canonical view renders")
    parser.add_argument("--view_fov", type=float, default=40.0, help="Field of view for canonical view renders")

    args = parser.parse_args()
    main(args)
