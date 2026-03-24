# Trellis2 + SpaceControl Integration Guide

이 문서는 기존 `Trellis2` (Image-to-3D) 파이프라인에 사용자의 3D 기하학적 형태를 가이드로 사용하는 `SpaceControl` 기능을 통합하기 위해 수정된 사항들과 실행 방법을 안내합니다.


## Installation

본 프로젝트는 고성능 3D 렌더링 및 복셀화를 위해 다양한 라이브러리 의존성을 가집니다. 아래 명령어를 순차적으로 실행하여 환경을 구축하세요.

```sh
# 필수 라이브러리 및 커스텀 커널 빌드
# --nvdiffrec: 차분 가능한 렌더링 엔진
# --cumesh: 고속 메쉬 처리 유틸리티
# --flexgemm: 가속 행렬 연산
./setup.sh --basic --nvdiffrec --cumesh --o-voxel --flexgemm
```


## 🛠️ 주요 수정 사항 (Modifications)

`trellis2_image_to_3d.py` 파이프라인 코드에 다음 3가지 핵심 기능이 추가/수정되었습니다.

### 1. Voxel 인코더 로드 (`model_names_to_load`)
기존 Trellis2 파이프라인은 추론 시 인코더를 사용하지 않지만, SpaceControl은 사용자의 3D 뼈대(Mesh/Superquadrics)를 Latent 공간으로 변환해야 하므로 `sparse_structure_encoder`를 로드하도록 `model_names_to_load` 리스트에 추가했습니다.

### 2. 공간 제어 데이터 인코딩 (`encode_spatial_control`)
입력된 3D 메쉬를 $64 \times 64 \times 64$ 해상도의 이진 복셀(Voxel) 텐서로 변환하고, 이를 다시 Trellis의 Latent 형태로 인코딩하는 `encode_spatial_control` 메서드를 추가했습니다.
* **VRAM 최적화:** `low_vram` 모드를 고려하여, 인코딩을 수행할 때만 인코더를 GPU(`device`)로 올리고 작업이 끝나면 다시 CPU로 내리도록 구현했습니다.
* **유틸리티 사용:** 내부적으로 `gui.utils.voxelize_sq_francis` 함수를 사용하여 3D 메쉬의 복셀화를 수행합니다.

### 3. 구조 생성 단계 제어 주입 (`run` 메서드 수정)
파이프라인의 실행 흐름을 담당하는 `run` 메서드를 수정하여 뼈대 생성 단계에만 제어 신호를 주입했습니다.
* `sample_sparse_structure` (뼈대 생성) 함수가 실행되기 직전, 조건 딕셔너리(`cond_512`, `cond_1024`)에 `"control"` 키로 추출한 `spatial_control_latent`를 주입합니다.
* 뼈대 생성이 완료된 후에는 이후 단계(`shape_slat`, `tex_slat`)에서 충돌 및 에러가 발생하지 않도록 `"control"` 키를 딕셔너리에서 안전하게 삭제(pop)합니다.

---

## 🚀 실행 방법 (Usage)

수정된 파이프라인을 테스트하기 위해 작성된 `example_spacecontrol.py` 스크립트를 사용합니다. 이 스크립트는 `argparse`를 지원하여 매번 코드를 수정할 필요 없이 터미널에서 입력 파일 경로를 동적으로 지정할 수 있습니다.

### 기본 실행 명령어
터미널에서 아래 명령어를 실행하세요. `--image`와 `--control` 파라미터는 필수입니다.

```bash
python example_spacecontrol.py \
  --image assets/shoe2.jpg \
  --control assets/last_normalize.ply
```

### 선택 파라미터 활용
결과물(동영상 및 3D 모델)의 저장 이름이나 경로를 변경하고 싶다면 아래와 같이 선택 파라미터를 추가할 수 있습니다.

```bash
python example_spacecontrol.py \
  --image assets/shoe3-no_lace-rembg.png \
  --control assets/last_normalized.ply \
  --video_out output_shoe_video.mp4 \
  --mesh_out output_shoe_model.glb \
  --tau 6
```


```sh
runit -n 2 --image \
  shoe2-rembg.png shoe2-rembg.png shoe2-rembg.png shoe2-rembg.png \
  shoe2-no_lace-rembg.png shoe2-no_lace-rembg.png shoe2-no_lace-rembg.png shoe2-no_lace-rembg.png \
  shoe3-no_lace-rembg.png shoe3-no_lace-rembg.png shoe3-no_lace-rembg.png shoe3-no_lace-rembg.png \
  shoe4_rembg.png shoe4_rembg.png shoe4_rembg.png shoe4_rembg.png \
  --tau 0 2 4 6 0 2 4 6 0 2 4 6 0 2 4 6 \
-- python example_spacecontrol.py --image assets/{image} --control assets/last_normalized.ply --tau {tau} --out_dir results/spacecontrol
```


### 파라미터 목록 (Arguments)
* `--image` (필수): 질감과 외형의 기준이 될 원본 이미지 파일의 경로 (예: `.jpg`, `.png`)
* `--control` (필수): 뼈대 가이드로 사용할 3D 기하학적 형태 파일의 경로 (예: `.ply`, `.obj`)
* `--video_out` (선택): 렌더링된 360도 회전 결과물 영상의 저장 경로 (기본값: `sample_spacecontrol.mp4`)
* `--mesh_out` (선택): 최종 생성된 3D 모델(GLB)의 저장 경로 (기본값: `sample_spacecontrol.glb`)

---

## 💡 참고 사항 (Notes)
* **메모리(VRAM):** 해상도가 높은 이미지를 처리하거나 복잡한 메쉬를 병합할 때 VRAM을 많이 소모할 수 있습니다. `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` 설정이 적용되어 있으나, OOM(Out of Memory) 발생 시 다른 GPU 작업을 종료 후 실행하는 것을 권장합니다.
* **유틸리티 의존성:** 실행을 위해서는 SpaceControl 프로젝트의 `gui/utils.py` (복셀화 관련 코드)가 동일한 환경 내에 올바르게 위치해야 합니다.

