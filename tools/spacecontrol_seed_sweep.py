import argparse
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Run example_spacecontrol.py across multiple seeds and taus and build a grid.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--control", required=True, help="Spatial control mesh path")
    parser.add_argument("--tau", type=int, default=6, help="SpaceControl tau")
    parser.add_argument("--taus", type=int, nargs="+", default=None, help="Optional list of taus to run")
    parser.add_argument("--out_dir", default="outputs/spacecontrol_seed_sweep", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)), help="Seeds to run")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(6)), help="GPU indices to use")
    parser.add_argument("--under_view_resolution", type=int, default=768, help="Per-seed underside image resolution")
    parser.add_argument("--under_view_radius", type=float, default=2.0, help="Camera radius for underside render")
    parser.add_argument("--under_view_fov", type=float, default=40.0, help="Field of view for underside render")
    parser.add_argument("--columns", type=int, default=None, help="Grid columns")
    return parser.parse_args()


def make_command(args, tau, seed, run_dir, under_view_path):
    return [
        sys.executable,
        "example_spacecontrol.py",
        "--image",
        args.image,
        "--control",
        args.control,
        "--tau",
        str(tau),
        "--seed",
        str(seed),
        "--name",
        f"tau{tau:02d}-seed{seed:02d}",
        "--out_dir",
        str(run_dir),
        "--under_view_out",
        str(under_view_path),
        "--under_view_resolution",
        str(args.under_view_resolution),
        "--under_view_radius",
        str(args.under_view_radius),
        "--under_view_fov",
        str(args.under_view_fov),
        "--skip_video",
        "--skip_glb",
    ]


def run_worker(gpu_id, job_queue, failures, args, run_dir, under_dir, log_dir):
    while True:
        try:
            tau, seed = job_queue.get_nowait()
        except queue.Empty:
            return

        under_view_path = under_dir / f"tau{tau:02d}-seed{seed:02d}.png"
        log_path = log_dir / f"tau{tau:02d}-seed{seed:02d}.log"
        cmd = make_command(args, tau, seed, run_dir, under_view_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_path, "w") as log_file:
            log_file.write(f"GPU {gpu_id}\n")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, env=env, stdout=log_file, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            failures.append((tau, seed, gpu_id, log_path))

        job_queue.task_done()


def build_grid(image_paths, out_path, columns, label_fn=None):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    tile_w, tile_h = images[0].size
    label_h = 40
    tiles = []
    for path, image in zip(image_paths, images):
        tile = Image.new("RGB", (tile_w, tile_h + label_h), "black")
        tile.paste(image, (0, 0))
        draw = ImageDraw.Draw(tile)
        label = label_fn(path) if label_fn is not None else path.stem
        draw.text((16, tile_h + 10), label, fill="white")
        tiles.append(tile)

    rows = math.ceil(len(tiles) / columns)
    grid = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "black")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = (idx // columns) * (tile_h + label_h)
        grid.paste(tile, (x, y))
    grid.save(out_path)


def main():
    args = parse_args()
    taus = args.taus or [args.tau]
    columns = args.columns or len(args.seeds)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / "runs"
    under_dir = out_dir / "under_views"
    log_dir = out_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    under_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    job_queue = queue.Queue()
    for tau in taus:
        for seed in args.seeds:
            job_queue.put((tau, seed))

    failures = []
    threads = []
    for gpu_id in args.gpus:
        thread = threading.Thread(
            target=run_worker, args=(gpu_id, job_queue, failures, args, run_dir, under_dir, log_dir), daemon=True
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if failures:
        for tau, seed, gpu_id, log_path in failures:
            print(f"Tau {tau} seed {seed} failed on GPU {gpu_id}. See {log_path}")
        raise SystemExit(1)

    image_paths = [under_dir / f"tau{tau:02d}-seed{seed:02d}.png" for tau in taus for seed in args.seeds]
    grid_path = out_dir / "under_view_grid.png"
    build_grid(image_paths, grid_path, columns, label_fn=lambda path: path.stem.replace("-", " "))
    print(grid_path)


if __name__ == "__main__":
    main()
