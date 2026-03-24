import os
import json
import argparse
from huggingface_hub import snapshot_download

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
import o_voxel
import torch
from PIL import Image

from trellis2.pipelines import SpaceControlPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils


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

    video_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}.mp4")
    mesh_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}.glb")
    # ---------------------------------------------------------

    # 5. Render Video
    print(f"Rendering video to {video_out}...")
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave(video_out, video, fps=15)

    # 6. Export to GLB
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
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save the generated files")

    args = parser.parse_args()
    main(args)
