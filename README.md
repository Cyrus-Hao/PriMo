# PriMo: Prior-Guided Mobile Reconstruction Beyond Texture

PriMo is a prior-guided mobile reconstruction project built on top of [MP-SfM](https://github.com/cvg/mpsfm). It targets a practical failure mode of mobile SfM: **camera misregistration in low-texture, weak-texture, and repetitive-texture scenes**.

Compared with the original MP-SfM codebase, PriMo mainly adds **mobile prior pose integration**, **prior-pose-driven pair selection**, **prior pose refinement during registration**, and **optional PromptDA depth prompting** for ARKit-like captures.

## ✨ Overview

PriMo focuses on making MP-SfM work better for mobile captures where image evidence alone is not reliable enough. In the current codebase, this is done by injecting prior information into the original pipeline rather than replacing it.

The repository currently provides:

- ingestion of **ARKit / mobile prior poses and intrinsics** into the reconstruction workflow,
- a more flexible **pair selection strategy based on prior pose geometry** instead of relying only on NetVLAD retrieval,
- a **prior pose refine framework** for both initialization and incremental registration,
- optional integration of **Prompt Depth Anything** using low-resolution sensor depth as prompt.

In our current experiments, these changes effectively reduce the misregistration issue of MP-SfM in **low-texture / weak-texture** scenes.

## 🚀 Highlights

- **ARKit prior pose and intrinsics integration.** PriMo reads mobile pose and camera metadata and injects them into the MP-SfM registration pipeline.
- **Prior-pose-based pair selection.** PriMo adds a retrieval alternative that selects image pairs directly from prior pose geometry, replacing the original NetVLAD-only strategy when prior poses are available.
- **Prior pose refine framework.** PriMo refines prior-guided poses during initialization and next-view registration instead of using the raw priors as fixed inputs.
- **Effective mitigation of weak-texture misregistration.** The main practical gain of PriMo is improved stability and robustness in low-texture and weak-texture scenes where the original MP-SfM registration is more brittle.
- **Optional PromptDA depth prompting.** PriMo integrates PromptDA so low-resolution ARKit depth can be used as a metric depth prompt.

## Setup

PriMo follows the same core environment stack as [MP-SfM](https://github.com/cvg/mpsfm): build `pyceres` and `pycolmap` from source, then install the Python package and its dependencies. The provided `Dockerfile` automates these lower-level system and compilation steps for containerized use.

### 1. Clone the repository and submodules

```bash
git clone --recursive <your-repo-url>
cd PriMo
```

If you already cloned the repository without submodules, run:

```bash
git submodule update --init --recursive
```

For public release, all third-party projects, including `third_party/PromptDA`, are expected to be fetched through git submodules.

### 2. Install dependencies

After building `pyceres` and `pycolmap` from source, install the Python dependencies:

```bash
pip install -r requirements.txt
python -m pip install -e .
```

### Optional

- For faster inference with transformer-based models, install [xformers](https://github.com/facebookresearch/xformers).
- For faster MASt3R matching, compile the CUDA kernels for RoPE:

```bash
DIR=$PWD
cd third_party/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd $DIR
```

- For `DepthPro`, the current repository follows the MP-SfM setup and keeps `ml-depth-pro` as a separate third-party dependency:

```bash
DIR=$PWD
cd third_party/ml-depth-pro/
pip install -e . --no-deps
cd $DIR
```

- `PromptDA` is integrated slightly differently from the other depth models: PriMo imports it directly from `third_party/PromptDA` via `sys.path`, so **running PriMo does not require an extra `pip install -e third_party/PromptDA` step**. As long as the submodule exists and the checkpoint path is configured, PromptDA works inside PriMo.
- If you want to run the official PromptDA standalone scripts outside PriMo, you can still install it manually:

```bash
DIR=$PWD
cd third_party/PromptDA/
pip install -e . --no-deps
cd $DIR
```

### Docker

This repository already provides a `Dockerfile`, but it is closer to an MP-SfM-style base environment than a fully pre-baked PriMo runtime image.

Build the image locally:

```bash
docker build -t primo:latest .
```

Run it with the repository mounted:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd):/workspace/PriMo \
  -w /workspace/PriMo primo:latest
```

Inside the container, finish by installing the package:

```bash
pip install -e .
```

Optional steps inside Docker are the same as above:

```bash
# optional MASt3R speed up
DIR=$PWD
cd third_party/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd $DIR

# optional DepthPro install
cd third_party/ml-depth-pro/
pip install -e . --no-deps
cd $DIR

# optional PromptDA standalone install
cd third_party/PromptDA/
pip install -e . --no-deps
cd $DIR
```

### 3. Prepare mobile pose and intrinsics files

PriMo includes `camera_processor.py` to convert mobile capture metadata into PriMo-compatible YAML files.

```bash
python camera_processor.py \
    --camera-matrix local/example/sofa/camera_matrix.csv \
    --odometry local/example/sofa/odometry.csv \
    --images-dir local/example/images \
    --output-dir local/example \
    --sample-interval 1
```

This writes:

- `local/example/camera_poses.yaml`
- `local/example/intrinsics.yaml`

### 4. Run reconstruction

```bash
python reconstruct.py \
    --conf sp-lg_m3dv2 \
    --data_dir local/example \
    --intrinsics_pth local/example/intrinsics.yaml \
    --images_dir local/example/images \
    --cache_dir local/example/cache_dir \
    --pose_config_path local/example/camera_poses.yaml \
    --extract \
    --verbose 0
```

Outputs are written under `local/example/sfm_outputs`, while extracted priors and matches are cached under `local/example/cache_dir`.

## What PriMo Changes over MP-SfM

PriMo is not a full rewrite of MP-SfM. Instead, it introduces a small number of focused changes around registration and mobile priors:

1. **Inject prior poses and intrinsics from mobile capture metadata.**
2. **Select matching pairs from prior poses** through `pairs_from_prior_pose(...)`, instead of relying only on NetVLAD retrieval.
3. **Refine prior-guided initialization** using normalized correspondences, an essential matrix induced by the prior relative pose, and Levenberg-Marquardt optimization over a 6-DoF update.
4. **Refine incremental prior poses** with pose-only optimization from 2D-3D correspondences collected from already registered views.
5. **Optionally improve monocular depth priors** with PromptDA using ARKit-like low-resolution depth prompts.

## Prompted Depth with PromptDA

PriMo also supports Prompt Depth Anything for mobile captures with rough sensor depth, such as ARKit or Stray Scanner depth maps.

Configure `configs/defaults/promptda.yaml` with:

```yaml
extractors:
  promptda:
    pretrained_path: /absolute/path/to/prompt_depth_anything_vitl.ckpt
    prompt_depth_dir: /absolute/path/to/arkit_depth_png
```

Then run:

```bash
python reconstruct.py \
    --conf defaults/promptda \
    --data_dir local/example \
    --intrinsics_pth local/example/intrinsics.yaml \
    --images_dir local/example/images \
    --cache_dir local/example/cache_dir \
    --pose_config_path local/example/camera_poses.yaml \
    --extract \
    --verbose 0
```

By default, PromptDA depth prompts are expected as `.png` files and interpreted as millimeter depth, which is converted internally to meters.

## Repository Structure

- `reconstruct.py`: main entry point for reconstruction experiments.
- `camera_processor.py`: converts mobile metadata into `camera_poses.yaml` and `intrinsics.yaml`.
- `configs/`: high-level reconstruction presets and per-estimator defaults.
- `PriMo/sfm/mapper/registration.py`: prior-pose-guided registration and prior pose refinement logic.
- `PriMo/extraction/pairs/prior_pose.py`: prior-pose-based pair selection that can replace the original NetVLAD retrieval path.
- `PriMo/extraction/imagewise/geometry/models/depth/promptda.py`: PromptDA wrapper used inside PriMo.
- `third_party/`: external projects vendored as submodules.

In particular, the PromptDA integration is source-tree based: PriMo loads `third_party/PromptDA` directly, similar in spirit to the other vendored third-party geometry models, but without requiring a mandatory package installation step for the PriMo pipeline itself.

## Current Focus

PriMo is especially aimed at improving reconstruction robustness in cases where camera registration is unreliable because of:

- weak visual texture,
- repeated patterns or symmetric structure,
- low-overlap mobile trajectories,
- noisy mobile depth and pose priors.

The current codebase should be considered an actively evolving research release built around a specific goal: making MP-SfM more reliable for difficult mobile captures, especially those that suffer from low- or weak-texture misregistration.

## 🙏 Acknowledgements

PriMo is built on top of the excellent [MP-SfM](https://github.com/cvg/mpsfm) codebase. We especially thank the MP-SfM authors for making their implementation available and for providing the foundation that this project extends.

We also gratefully acknowledge the open-source projects used in this repository or its integrated priors and matchers:

- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R](https://github.com/naver/mast3r)
- [Metric3D](https://github.com/yvanyin/metric3d)
- [Depth Pro](https://github.com/apple/ml-depth-pro)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Prompt Depth Anything](https://github.com/DepthAnything/PromptDA)
- [DSINE](https://github.com/baegwangbin/DSINE)
- [LightGlue](https://github.com/cvg/LightGlue)
- [SuperPoint](https://github.com/rpautrat/SuperPoint)
- [NetVLAD](https://github.com/Relja/netvlad)

We especially thank [MP-SfM](https://github.com/cvg/mpsfm), since PriMo directly extends its reconstruction framework rather than starting from scratch.
