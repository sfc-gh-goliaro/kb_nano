#!/bin/bash
# ============================================================================
# setup_repro_envs.sh
#
# Prepares EVERY environment needed to reproduce table_results.tex on a machine
# configured like the reproduction host (8x H200, CUDA 12.9 + 13.0 toolkits,
# conda env `dev` with torch 2.10.0+cu128 + vLLM 0.18.0 + fastkernels -e, as
# produced by /code/users/goliaro/init.sh).
#
# This script is ADDITIVE on top of init.sh. It creates:
#   * main-env extras (DP3 reference, Pi0 checkpoint download, CosyVoice3 asset)
#   * isolated uv venvs for rows whose reference libs are ABI-incompatible with
#     torch 2.10 (fastkernels reinstalled `-e --no-deps` so its L1 ops JIT-rebuild
#     against the venv's torch; fastkernels pins only torch>=2.0)
#   * external reference repos
#   * an instant-ngp source build (produces pyngp)
#   * model / checkpoint / asset downloads
#
# Most rows run in the base `dev` env. The isolated venvs are used ONLY for the
# reference side of a handful of rows (via bench flags like --vllm-python /
# --reference-python, or by running the whole bench in the venv). See
# docs/neurips-repro/h200/REPRODUCTION.md for the exact per-row run commands.
#
# Idempotent-ish: safe to re-run; existing venvs/clones are reused.
# ============================================================================
set -uo pipefail

KB=/home/yak/kb_nano
REF=/home/yak/reference_code
VENVS=/home/yak/repro_venvs
export HF_HOME=/home/yak/data-fast/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
# CUDA 12.9 matches torch 2.9's cu128 ABI for source/JIT builds (gsplat, instant-ngp,
# fastkernels ops). System default /usr/local/cuda is 13.0 — do NOT use it for these.
export CUDA_129=/usr/local/cuda-12.9
mkdir -p "$REF" "$VENVS" "$KB/third_party"

log(){ echo -e "\n\033[1;36m==== $* ====\033[0m"; }

# ---------------------------------------------------------------------------
log "1. Main-env (dev) extras"
# ---------------------------------------------------------------------------
# Whisper: HF datasets' Audio auto-decode uses torchcodec, whose prebuilt libs are
# ABI-incompatible with torch 2.10 (undefined symbol: torch_from_blob). We decode
# LibriSpeech with PyAV instead (see the _decode_audio_array helper in the Whisper
# worker templates in tests/bench_vllm.py).
uv pip install av
# DP3 reference (diffusion_policy_3d) runs in the MAIN env — the bench stubs pytorch3d
# and the pointnet encoder needs no spconv:
uv pip install zarr dill hydra-core einops
# Pi0 checkpoint is fetched from the public GCS bucket with anonymous gcsfs:
uv pip install gcsfs
# TTT-E2E JAX reference (jax + equinox usually already present in dev):
uv pip install "jax" equinox optax || true
# CosyVoice3 (vllm-omni) needs a Whisper mel-filter asset the bench's auto-download
# helper does not reliably fetch:
VO=$(python -c "import os,vllm_omni;print(os.path.dirname(vllm_omni.__file__))")
mkdir -p "$VO/model_executor/models/cosyvoice3/assets"
curl -sL https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
  -o "$VO/model_executor/models/cosyvoice3/assets/mel_filters.npz"

# ---------------------------------------------------------------------------
log "2. External reference repositories"
# ---------------------------------------------------------------------------
clone(){ [ -d "$2/.git" ] || git clone ${3:-} "$1" "$2"; }
clone https://github.com/test-time-training/e2e         "$REF/ttt-e2e"                     # TTT-E2E JAX ref
clone https://github.com/NVlabs/Fast-dLLM               "$REF/Fast-dLLM"                   # LLaDA (Fast-dLLM)
clone https://github.com/YanjieZe/3D-Diffusion-Policy   "$REF/3D-Diffusion-Policy"         # DP3
clone https://github.com/Physical-Intelligence/openpi   "$REF/openpi"                      # Pi0
clone https://github.com/Pointcept/PointTransformerV3   "$KB/third_party/PointTransformerV3"  # PTv3

# ---------------------------------------------------------------------------
log "3. Isolated venv: OpenPI (Pi0 reference + JAX->PyTorch checkpoint convert)"
#    openpi is hard-pinned to jax 0.5.3 / jaxtyping 0.2.36 (incompatible with the
#    dev env's jax 0.11). Build its own env; it pulls torch 2.7.1+cu126.
# ---------------------------------------------------------------------------
uv venv "$VENVS/openpi" --python 3.11
uv pip install --python "$VENVS/openpi/bin/python" -e "$REF/openpi"
# transformers_replace overlay provides PI0ForConditionalGeneration:
uv pip install --python "$VENVS/openpi/bin/python" transformers==4.53.2
OPT_TF=$("$VENVS/openpi/bin/python" -c "import os,transformers;print(os.path.dirname(transformers.__file__))")
cp -r "$REF/openpi/src/openpi/models_pytorch/transformers_replace/"* "$OPT_TF/"
# Download + convert the Pi0 checkpoint (public GCS, anon; recursive get nests one level):
PI0_DL=/home/yak/data-fast/pi0_ckpt
mkdir -p "$PI0_DL"
python - <<PY
import gcsfs,os
fs=gcsfs.GCSFileSystem(token='anon')
fs.get('openpi-assets/checkpoints/pi0_aloha_pen_uncap', "$PI0_DL/pi0_aloha_pen_uncap", recursive=True)
PY
JAX_PLATFORMS=cpu "$VENVS/openpi/bin/python" "$REF/openpi/examples/convert_jax_model_to_pytorch.py" \
  --config-name pi0_aloha_pen_uncap \
  --checkpoint-dir "$PI0_DL/pi0_aloha_pen_uncap/pi0_aloha_pen_uncap" \
  --output-path "$PI0_DL/pi0_aloha_pen_uncap_pytorch"
mkdir -p "$PI0_DL/pi0_aloha_pen_uncap_pytorch/assets"
cp -r "$PI0_DL/pi0_aloha_pen_uncap/pi0_aloha_pen_uncap/assets/"* "$PI0_DL/pi0_aloha_pen_uncap_pytorch/assets/" 2>/dev/null || true

# ---------------------------------------------------------------------------
log "4. Isolated venv: DLRMv2 (TorchRec + fbgemm-gpu, torch 2.9)"
#    fbgemm-gpu/torchrec MUST match torch: 1.4.0 <-> torch 2.9 (1.3.0 is torch 2.8
#    and aborts std::length_error; latest is torch 2.10 and undefined-symbols).
# ---------------------------------------------------------------------------
uv venv "$VENVS/dlrm" --python 3.12
uv pip install --python "$VENVS/dlrm/bin/python" \
  "torch==2.9.1" "torchrec==1.4.0" "fbgemm-gpu==1.4.0" \
  datasets scikit-learn safetensors numpy pandas triton
uv pip install --python "$VENVS/dlrm/bin/python" -e "$KB" --no-deps

# ---------------------------------------------------------------------------
log "5. Isolated venv: PointTransformerV3 (spconv, torch 2.9)"
#    spconv-cu126 wheel; datasets/pyarrow pinned to the dev-env combo (a skew
#    crashes on pyarrow.PyExtensionType).
# ---------------------------------------------------------------------------
uv venv "$VENVS/ptv3" --python 3.12
uv pip install --python "$VENVS/ptv3/bin/python" \
  "torch==2.9.1" spconv-cu126 "datasets==4.0.0" "pyarrow==18.1.0" \
  einops numpy safetensors triton addict timm h5py scipy huggingface_hub hf_transfer
uv pip install --python "$VENVS/ptv3/bin/python" -e "$KB" --no-deps

# ---------------------------------------------------------------------------
log "6. Isolated venv: 3DGS (gsplat, torch 2.9)"
#    gsplat JIT-compiles its CUDA extension on first use -> needs CUDA 12.9 nvcc.
# ---------------------------------------------------------------------------
uv venv "$VENVS/gs" --python 3.12
CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" \
  uv pip install --python "$VENVS/gs/bin/python" \
  "torch==2.9.1" gsplat numpy safetensors triton ninja jaxtyping huggingface_hub hf_transfer pillow
uv pip install --python "$VENVS/gs/bin/python" -e "$KB" --no-deps

# ---------------------------------------------------------------------------
log "7. instant-ngp source build (produces pyngp for the InstantNGP reference)"
#    No wheel exists for pyngp OR tinycudann; tiny-cuda-nn is compiled as an
#    instant-ngp submodule. Build against CUDA 12.9, main-env python 3.12.
# ---------------------------------------------------------------------------
[ -d "$KB/third_party/instant-ngp/.git" ] || \
  git clone --recursive https://github.com/NVlabs/instant-ngp "$KB/third_party/instant-ngp"
( cd "$KB/third_party/instant-ngp"
  CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" LD_LIBRARY_PATH="$CUDA_129/lib64:${LD_LIBRARY_PATH:-}" \
    cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=OFF -DPython_EXECUTABLE="$(which python)"
  CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" LD_LIBRARY_PATH="$CUDA_129/lib64:${LD_LIBRARY_PATH:-}" \
    cmake --build build --config RelWithDebInfo -j 16 )

# ---------------------------------------------------------------------------
log "8. Isolated venv: paper-era vLLM 0.20.1 (Gemma-4 + Qwen-Omni baselines)"
#    vLLM 0.18 (dev env) has no gemma4; 0.20.1 (available first week of May 2026,
#    paper submission) has native Gemma4ForConditionalGeneration. transformers>=5.5
#    provides the gemma4 config. Used as the vLLM baseline via bench --vllm-python.
#    Multimodal deps included so it also serves the Qwen-2.5-Omni baseline.
# ---------------------------------------------------------------------------
uv venv "$VENVS/vllm020" --python 3.12
uv pip install --python "$VENVS/vllm020/bin/python" "vllm==0.20.1"
uv pip install --python "$VENVS/vllm020/bin/python" \
  "transformers==5.8.0" fastsafetensors datasets pillow hf_transfer librosa soundfile av qwen-omni-utils
uv pip install --python "$VENVS/vllm020/bin/python" -e "$KB" --no-deps   # for rank-align real_prompts import

# ---------------------------------------------------------------------------
log "9. (Optional) Isolated venv: vLLM 0.16.0 (Qwen-Omni version-consistency check)"
# ---------------------------------------------------------------------------
uv venv "$VENVS/vllm016" --python 3.12
uv pip install --python "$VENVS/vllm016/bin/python" "vllm==0.16.0"
uv pip install --python "$VENVS/vllm016/bin/python" \
  datasets pillow av librosa soundfile hf_transfer qwen-omni-utils fastsafetensors

# ---------------------------------------------------------------------------
log "10. Model / checkpoint downloads (weights cache to \$HF_HOME)"
# ---------------------------------------------------------------------------
for M in \
  openai/whisper-large-v3 \
  google/gemma-4-26B-A4B-it \
  GSAI-ML/LLaDA-8B-Instruct \
  black-forest-labs/FLUX.1-dev \
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  Qwen/Qwen2.5-Omni-7B \
  OpenFold/OpenFold3 ; do
  echo "  downloading $M ..."
  huggingface-cli download "$M" || echo "  (WARN: $M download failed — retry manually)"
done
# OpenProteinSet MSAs (OpenFold3) auto-download from the public S3 bucket at run time
# (needs the aws CLI, present in dev). ScanObjectNN (PTv3) + gym-xarm-pointcloud (DP3)
# auto-download from HF at run time.

log "DONE. See docs/neurips-repro/h200/REPRODUCTION.md for per-row run commands."
echo "Isolated venvs created under $VENVS :"
ls -1 "$VENVS"
