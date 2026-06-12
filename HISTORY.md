# HISTORY

## Scope

This file summarizes the current TRELLIS.2 shoe-last guidance work so another AI can continue without re-discovering the same issues.

Repo root: `/home/rvi/dev/TRELLIS.2`

Current user goal:
- Generate a shoe mesh conditioned on a `last`
- Final output should become a single watertight mesh
- It is acceptable if the `last` ends up fully embedded inside the shoe mesh
- Hollow interior is not required


## High-level Conclusions

1. The original TRELLIS.2 raw mesh path is still bad for this use case.
   - Raw outputs are highly fragmented and open.
   - This is not just a small postprocess issue.

2. The sparse guidance path now works and is wired into the repo.
   - `control_mode=none|spacecontrol|guidance|both`
   - Geometry guidance is available at sparse-structure stage.

3. The old interpretation of guidance as an outer hollow shell was too strict for the user’s real requirement.
   - We switched `containment` guidance to an **embedded-last** objective.
   - The goal is now: keep the generated mass around/over the last and reduce outside leakage.
   - The last may be entirely inside the final mesh.

4. The exporter orientation mismatch was real.
   - Upstream `o_voxel.postprocess.to_glb()` applies an extra X-axis `-90 deg` transform.
   - Repo export now uses a local wrapper that keeps exported GLBs in the same internal Z-up frame as rendering.

5. The strongest result so far is not “raw generator got good”.
   - The best current path is:
     - generate with `both + containment`
     - then run a new **volume solidify** postprocess
   - This produces a single watertight component.


## Important Files Changed

### Export / orientation

- `trellis2/utils/glb_utils.py`
  - Added `to_glb_z_up()`
  - Wraps upstream exporter and cancels the built-in extra quarter-turn

- `example.py`
- `app.py`
- `example_spacecontrol.py`
  - Switched GLB export call sites to `to_glb_z_up()`

### Sparse guidance

- `trellis2/pipelines/guidance/__init__.py`
- `trellis2/pipelines/guidance/geometry_guidance.py`
  - Added sparse guidance package
  - Added shell-style and embedded-last guidance variants
  - `ContainmentGeometryGuidance` is now aliased to `EmbeddedLastGeometryGuidance`
  - Bottom support logic uses Z-up / low-Z band

- `trellis2/pipelines/samplers/flow_euler_geometry_guidance.py`
  - New geometry-guided sparse sampler

- `trellis2/pipelines/samplers/__init__.py`
  - Exports new sampler

- `trellis2/pipelines/spacecontrol.py`
  - Wires `control_mode=none|spacecontrol|guidance|both`
  - Keeps sparse decoder available during guidance
  - Stores guidance metrics in `pipeline.last_guidance_metrics`

- `trellis2/pipelines/samplers/flow_euler.py`
- `trellis2/pipelines/samplers/flow_euler_geometry_guidance.py`
  - Fixed `space_control_tau` leakage by always `pop()`-ing it inside samplers

### Mesh repair / solidify

- `trellis2/representations/mesh/base.py`
  - Added wrappers for:
    - `remove_duplicate_faces()`
    - `repair_non_manifold_edges()`
    - `remove_small_connected_components(min_area)`
    - `unify_face_orientations()`

- `trellis2/utils/mesh_repair.py`
  - Added:
    - `mesh_topology_stats(mesh)`
    - `aggressive_repair_mesh(mesh, ...)`

- `trellis2/utils/mesh_solidify.py`
  - Added:
    - `solidify_mesh_with_control(...)`
    - `smooth_mesh_taubin(...)`
  - This is currently the key path for getting one-piece watertight output

- `trellis2/utils/__init__.py`
  - Exports `to_glb_z_up`, `aggressive_repair_mesh`, `mesh_topology_stats`, `solidify_mesh_with_control`, `smooth_mesh_taubin`

### Example entrypoints

- `example_spacecontrol.py`
  - Extended to pass sparse CFG + geometry guidance params
  - Added topology JSON output
  - Added `--solidify_volume` path
  - Added plain-mesh render/export branch for solidified meshes
  - Added `--solidify_smoothing_iters`

- `example_guidance.py`
  - Thin wrapper over `example_spacecontrol.py`
  - Includes guidance and solidify CLI args

### Other fix

- `trellis2/modules/image_feature_extractor.py`
  - Patched for DINOv3 compatibility by using `.last_hidden_state`
  - Do not revert unless a better upstream-compatible fix is verified


## Guidance Semantics

### Z-up correction

The repo should now be treated as **Z-up** internally.

Important fix in `geometry_guidance.py`:
- bottom/sole support uses low-Z, not Y

### Current guidance modes

Implemented in `build_geometry_guidance(...)`:
- `latent`
- `occupancy`
- `containment`
- `shell`

### Meaning of `containment` now

`containment` no longer means “preserve a hollow shell around the last”.

It now means:
- encourage occupancy through the last volume
- penalize occupancy outside a dilated envelope
- encourage support near low-Z sole band

This matches the clarified user requirement better.


## Why Raw Meshes Still Fail

The decoder path is fundamentally permissive of open surfaces.

Relevant fact:
- `trellis2/models/sc_vaes/fdg_vae.py` uses flexible dual grid / O-Voxel extraction
- That representation can naturally yield open boundaries and fragmented surfaces

Consequence:
- light hole filling alone does not reliably produce watertight output
- even aggressive repair only improves topology partway if applied directly to the raw decoded surface


## Key Experimental Results

### Earlier sparse-guidance comparison

Generated outputs:
- `results/guidance_zup/`
- `results/both_zup/`

Observed:
- `guidance-only` was poor and fragmented
- `both` was meaningfully better

Metrics previously observed:

`guidance-only`
- `outside_fraction` mean `0.4568`
- `shell_fill` mean `0.0724`
- `bottom_fill` mean `0.1050`
- `envelope_iou` mean `0.0528`

`both`
- `outside_fraction` mean `0.0305`
- `shell_fill` mean `0.3600`
- `bottom_fill` mean `0.5183`
- `envelope_iou` mean `0.4286`

### Aggressive repair without solidify

Run:
- `results/both_containment_watertight/`

Important topology numbers from:
- `results/both_containment_watertight/shoe4_rembg-tau6-topology.json`

Before aggressive repair:
- `num_boundaries = 922873`
- `num_connected_components = 78910`
- `num_boundary_loops = 1960`

After aggressive repair:
- `num_boundaries = 294090`
- `num_connected_components = 793`
- `num_boundary_loops = 5`

Interpretation:
- huge improvement
- still not watertight
- still not one-piece

### Containment-guided run metrics

Run:
- `results/both_containment_watertight/shoe4_rembg-tau6-guidance.json`

Representative values:
- `containment_recall` mean about `0.7688`, last about `0.9852`
- `outside_fraction` mean about `0.0261`
- `bottom_fill` mean about `0.5034`
- `envelope_iou` mean about `0.4162`

Interpretation:
- guidance is doing something useful
- raw surface quality remains the limiting factor


## Volume Solidify Work

### Core idea

Instead of trying to rescue the raw FDG surface directly:
- voxelize generated surface into a dense grid
- voxelize the control `last`
- union them
- run morphology
- keep only largest connected component
- fill holes in volume
- re-extract a mesh with marching cubes

This is acceptable because the user explicitly said:
- the `last` may be completely inside the final mesh
- interior cavity is not important

### Independent proof that this works

A direct offline test on an already bad GLB showed:
- `num_boundaries = 0`
- `num_connected_components = 1`
- `watertight = True`

That validated the approach before wiring it into the example path.

### End-to-end solidified run

Command pattern used:
- `both + containment + solidify_volume`

Output directory:
- `results/both_containment_solidified/`

Important file:
- `results/both_containment_solidified/shoe4_rembg-tau6-topology.json`

Topology:
- before:
  - `num_boundaries = 896042`
  - `num_connected_components = 63012`
  - `num_boundary_loops = 1871`
- solidified:
  - `num_boundaries = 0`
  - `num_connected_components = 1`
  - `num_boundary_loops = 0`
- after:
  - same as solidified

Also verified separately from exported GLB:
- `watertight = True`
- `components = 1`

### Better solidify parameter sweep

Several parameter sets were tested on the bad GLB.

Best current tradeoff:
- `resolution = 192`
- `generated_surface_dilation = 1`
- `control_surface_dilation = 1`
- `closing_iters = 1`
- `final_closing_iters = 1`

This still produced:
- `num_boundaries = 0`
- `num_connected_components = 1`
- `watertight = True`

Artifacts:
- `results/both_containment_solidified_r192/`
  - `shoe4_rembg-solid-r192.glb`
  - `shoe4_rembg-solid-r192.mp4`
  - `shoe4_rembg-solid-r192-topology.json`

### Smoothing after solidify

Added Taubin smoothing as an option because marching-cubes solidify produces visible voxel stair-stepping.

Test artifact:
- `results/both_containment_solidified_r192_smooth/`
  - `shoe4_rembg-solid-r192-smooth.glb`
  - `shoe4_rembg-solid-r192-smooth.mp4`
  - `topology.json`

This remained:
- `watertight = True`
- `components = 1`

But visually it is still obviously voxel-derived.

### SDF solidify follow-up

The blocky look was confirmed to come primarily from binary voxel-volume solidify, not from sparse shape guidance.

Added an SDF-based solidify path:
- `solidify_mesh_with_sdf(...)`
- `--solidify_method sdf`
- `solidify_existing_mesh.py` for reprocessing an existing GLB without rerunning TRELLIS generation

Best current SDF artifact:
- `results/both_containment_sdf_r192_margin/`
  - `shoe4_rembg-sdf-r192-margin.glb`
  - `shoe4_rembg-sdf-r192-margin.mp4`
  - `frame0.png`
  - `topology.json`

Topology:
- before:
  - `num_boundaries = 736705`
  - `num_connected_components = 65901`
  - `num_boundary_loops = 68023`
- SDF solidified:
  - `num_boundaries = 0`
  - `num_connected_components = 1`
  - `num_boundary_loops = 0`

Important detail:
- SDF sampling needs a small domain margin because the generated shoe reaches the unit-cube boundary.
- Without `--domain_margin 0.03`, the SDF result had one component but still had boundary edges.
- With `--domain_margin 0.03`, the result became watertight and one-piece.


## Current Best Practical Recipe

For now, the best working path is:

1. Generate with:
   - `control_mode=both`
   - `guidance_type=containment`

2. Then solidify with:
   - `--solidify_volume`
   - `--solidify_resolution 192`
   - `--solidify_generated_surface_dilation 1`
   - `--solidify_control_surface_dilation 1`
   - `--solidify_closing_iters 1`
   - `--solidify_final_closing_iters 1`
   - optionally `--solidify_smoothing_iters 10`

This is the current best answer to:
- one watertight mesh
- last embedded inside is okay

For a less blocky result from an existing raw/aggressively repaired GLB, the current better-looking path is:

```bash
CUDA_VISIBLE_DEVICES=2 python solidify_existing_mesh.py \
  --input results/both_containment_watertight/shoe4_rembg-tau6.glb \
  --control assets/last_normalized.ply \
  --output results/both_containment_sdf_r192_margin/shoe4_rembg-sdf-r192-margin.glb \
  --topology_output results/both_containment_sdf_r192_margin/topology.json \
  --resolution 192 \
  --generated_surface_thickness 1.5 \
  --control_offset 1.0 \
  --smoothing_sigma 0.75 \
  --domain_margin 0.03 \
  --smoothing_iters 10
```

### No-guidance baseline

Created a baseline with generation guidance minimized by using `control_mode=none`.

Important execution detail:
- On the current Blackwell GPU environment, the default `flex_gemm` sparse convolution backend failed during Triton compilation for `cuda:120`.
- The successful raw generation used `SPARSE_CONV_BACKEND=spconv`.
- The input control mesh was still passed to the CLI for consistency with existing command shape, but `control_mode=none` means it was not used for SpaceControl or geometry guidance.
- The SDF postprocess intentionally omitted `--control`, so the last was not unioned back into the no-guidance result.

Raw no-guidance artifacts:
- `results/no_guidance_raw/shoe4_rembg-tau6.glb`
- `results/no_guidance_raw/shoe4_rembg-tau6.mp4`
- `results/no_guidance_raw/shoe4_rembg-tau6-topology.json`

Raw topology:
- before/after:
  - `num_boundaries = 57`
  - `num_connected_components = 1474`
  - `num_boundary_loops = 1`

SDF no-guidance artifacts:
- `results/no_guidance_sdf_r192_margin/shoe4_rembg-sdf-r192-margin.glb`
- `results/no_guidance_sdf_r192_margin/shoe4_rembg-sdf-r192-margin.mp4`
- `results/no_guidance_sdf_r192_margin/frame0.png`
- `results/no_guidance_sdf_r192_margin/topology.json`

SDF topology:
- before:
  - `num_boundaries = 103130`
  - `num_connected_components = 13410`
  - `num_boundary_loops = 13447`
- SDF solidified / after smoothing:
  - `num_boundaries = 0`
  - `num_connected_components = 1`
  - `num_boundary_loops = 0`

Reproduction commands:

```bash
SPARSE_CONV_BACKEND=spconv CUDA_VISIBLE_DEVICES=1 python example_guidance.py \
  --image assets/shoe4_rembg.png \
  --control assets/last_normalized.ply \
  --control_mode none \
  --out_dir results/no_guidance_raw
```

```bash
CUDA_VISIBLE_DEVICES=1 python solidify_existing_mesh.py \
  --input results/no_guidance_raw/shoe4_rembg-tau6.glb \
  --output results/no_guidance_sdf_r192_margin/shoe4_rembg-sdf-r192-margin.glb \
  --topology_output results/no_guidance_sdf_r192_margin/topology.json \
  --video_output results/no_guidance_sdf_r192_margin/shoe4_rembg-sdf-r192-margin.mp4 \
  --frame_output results/no_guidance_sdf_r192_margin/frame0.png \
  --resolution 192 \
  --generated_surface_thickness 1.5 \
  --smoothing_sigma 0.75 \
  --domain_margin 0.03 \
  --smoothing_iters 10
```


## Known Limitations

1. The raw generator is still poor.
   - The good topology currently comes from postprocess solidification, not from the decoder itself.

2. Solidify introduces voxelized shape artifacts.
   - Higher resolution helps but costs more.
   - Smoothing helps but does not remove the core discretization look.
   - The SDF solidify path reduces this blockiness, but can still look over-smoothed/lumpy because it rebuilds from a distance field.

3. The final solidified mesh is plain geometry.
   - In the solidify branch, export uses plain `trimesh.Trimesh(...).export(...)`
   - This means the solidified result is focused on geometry/topology, not PBR fidelity.

4. `voxelize_sq_francis(...)` still writes `merged_mesh_voxelized.ply` in cwd.
   - This is a side effect and should be cleaned up later.


## Good Next Steps

The next AI should not waste time pushing `fill_holes()` harder on the raw decoded mesh. That path already plateaued.

Most promising next direction:

1. Continue tuning **SDF solidify**
   - Current implementation uses `cumesh.bvh.cuBVH` distance queries.
   - Generated raw mesh is open, so it is treated as an unsigned-distance band.
   - Control last is unioned as signed distance.
   - Tune `resolution`, `generated_surface_thickness`, `control_offset`, `smoothing_sigma`, and `domain_margin`.

2. If keeping current solidify path:
   - tune `resolution`, morphology radii, and smoothing together
   - compare `128`, `192`, `256` if memory/time allow

3. If revisiting guidance:
   - keep `containment` semantics aligned with “embedded last is okay”
   - do not go back to enforcing a hollow shell unless the user explicitly changes direction


## Commands / Artifacts Worth Reusing

### End-to-end run that produced watertight one-piece output

```bash
python example_guidance.py \
  --image assets/shoe4_rembg.png \
  --control assets/last_normalized.ply \
  --control_mode both \
  --guidance_type containment \
  --steps 12 \
  --rescale_t 5.0 \
  --guidance_strength 7.5 \
  --guidance_interval 0.6 1.0 \
  --guidance_rescale 0.7 \
  --geometry_guidance_strength 1.0 \
  --geometry_guidance_interval 0.5 0.95 \
  --geometry_guidance_schedule constant \
  --geometry_guidance_rescale true \
  --geometry_grad_clip 5.0 \
  --geometry_guidance_cfg_mode cond_only \
  --envelope_radius 2 \
  --contain_weight 2.0 \
  --outside_weight 2.5 \
  --bottom_weight 1.0 \
  --bottom_band_ratio 0.18 \
  --bottom_outer_margin 1 \
  --solidify_volume \
  --solidify_resolution 128 \
  --solidify_generated_surface_dilation 2 \
  --solidify_control_surface_dilation 2 \
  --solidify_closing_iters 2 \
  --solidify_final_closing_iters 1 \
  --out_dir results/both_containment_solidified
```

### Current recommended solidify settings

Use these on top of the same command:

```bash
--solidify_resolution 192 \
--solidify_generated_surface_dilation 1 \
--solidify_control_surface_dilation 1 \
--solidify_closing_iters 1 \
--solidify_final_closing_iters 1 \
--solidify_smoothing_iters 10
```


## Workspace Notes

- Current checkout has a local `results` symlink to `/home/rvi/ns2/jaehyeok/results/TRELLIS.2`.
- The `results` symlink is local workspace state and should not be committed.
- The implementation files below are tracked in git:
  - `example_guidance.py`
  - `example_spacecontrol.py`
  - `trellis2/utils/glb_utils.py`
  - `trellis2/utils/mesh_repair.py`
  - `trellis2/utils/mesh_solidify.py`
  - `trellis2/pipelines/guidance/`
  - `trellis2/pipelines/samplers/flow_euler_geometry_guidance.py`
  - `trellis2/pipelines/spacecontrol.py`
  - `trellis2/pipelines/samplers/flow_euler.py`
  - `trellis2/representations/mesh/base.py`
  - `trellis2/modules/image_feature_extractor.py`
- The experimental `results/...` artifact directories referenced above are not tracked by git, but are currently accessible through the local symlink. Regenerate them with the recorded commands if the symlink target is unavailable.
