#!/bin/bash
# ============================================================================
# B200 reproduction environment setup (adapted from docs/neurips-repro/h200).
# Clones reference repos, builds third-party deps, creates isolated venvs.
# CPU-only work — safe to run while GPU benchmarks are in flight.
# ============================================================================
set -uo pipefail
KB=/home/yak/kb_nano
REF=/home/yak/reference_code
VENVS=/home/yak/repro_venvs
export HF_HOME=/home/yak/data-fast/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
# torch 2.10.0+cu128 / 2.9.x+cu128 -> build JIT/source extensions against CUDA 12.9,
# not the system-default 13.0 (ABI mismatch).
export CUDA_129=/usr/local/cuda-12.9
mkdir -p "$REF" "$VENVS" "$KB/third_party"
log(){ echo -e "\n\033[1;36m==== $* ====\033[0m"; }
clone(){ [ -d "$2/.git" ] || git clone ${3:-} "$1" "$2"; }

log "1. Reference repositories"
clone https://github.com/test-time-training/e2e         "$REF/ttt-e2e"
clone https://github.com/NVlabs/Fast-dLLM               "$REF/Fast-dLLM"
clone https://github.com/YanjieZe/3D-Diffusion-Policy   "$REF/3D-Diffusion-Policy"
clone https://github.com/Physical-Intelligence/openpi   "$REF/openpi"
clone https://github.com/Pointcept/PointTransformerV3   "$KB/third_party/PointTransformerV3"
clone https://github.com/THU-MIG/yolov10                "$KB/third_party/yolov10"
clone https://github.com/microsoft/BitNet               "$REF/BitNet"

log "2. Main-env extras"
uv pip install -q av zarr dill hydra-core einops gcsfs 2>&1 | tail -2

# --- Blackwell FA4 (CuTe-DSL) fix -------------------------------------------
# On sm100 vLLM 0.18.0 routes flash_attn_varlen_func to its bundled FA4 CuTe
# kernels (FA3 is Hopper-only).  That import chain needs a cutlass-dsl whose
# `cutlass.cute.core` still exports `ThrMma` (vllm_flash_attn/cute/utils.py) AND
# a `quack-kernels` that does not require cutlass-dsl 4.6's `_mlir_helpers`.
# 4.6.1 + quack 0.6.x breaks the first; 4.6.1 + quack 0.5.0 breaks the second.
# The working combination is cutlass-dsl 4.5.3 + quack-kernels 0.5.0.
# Without this, every vLLM model that uses encoder attention (Whisper, BGE-M3,
# ColBERTv2, the Qwen-VL ViTs) dies with
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
uv pip install -q "nvidia-cutlass-dsl==4.5.3"
uv pip install -q --no-deps "quack-kernels==0.5.0"
python - <<'PY'
from vllm.vllm_flash_attn import is_fa_version_supported
print("  FA4 supported:", is_fa_version_supported(4))
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
print("  FA4 CuTe interface imports OK")
PY

log "3. CosyVoice3 mel-filter asset"
VO=$(python -c "import os,vllm_omni;print(os.path.dirname(vllm_omni.__file__))" 2>/dev/null)
if [ -n "$VO" ]; then
  mkdir -p "$VO/model_executor/models/cosyvoice3/assets"
  curl -sL https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz \
    -o "$VO/model_executor/models/cosyvoice3/assets/mel_filters.npz" && echo "  mel_filters.npz OK"
fi

log "4. instant-ngp source build (pyngp)"
[ -d "$KB/third_party/instant-ngp/.git" ] || \
  git clone --recursive https://github.com/NVlabs/instant-ngp "$KB/third_party/instant-ngp"
if [ ! -f "$KB/third_party/instant-ngp/build/pyngp"*.so ] 2>/dev/null; then
  ( cd "$KB/third_party/instant-ngp"
    # B200 is sm_100; instant-ngp's CMake gates on detected arch.
    CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" LD_LIBRARY_PATH="$CUDA_129/lib64:${LD_LIBRARY_PATH:-}" \
      cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=OFF \
        -DPython_EXECUTABLE="$(which python)" > /tmp/ngp_cmake.log 2>&1 || tail -30 /tmp/ngp_cmake.log
    CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" LD_LIBRARY_PATH="$CUDA_129/lib64:${LD_LIBRARY_PATH:-}" \
      cmake --build build --config RelWithDebInfo -j 32 > /tmp/ngp_build.log 2>&1 || tail -40 /tmp/ngp_build.log )
fi

log "5. Isolated venv: DLRMv2 (torchrec + fbgemm-gpu, torch 2.9)"
if [ ! -x "$VENVS/dlrm/bin/python" ]; then
  uv venv "$VENVS/dlrm" --python 3.12
  uv pip install -q --python "$VENVS/dlrm/bin/python" \
    "torch==2.9.1" "torchrec==1.4.0" "fbgemm-gpu==1.4.0" \
    datasets scikit-learn safetensors numpy pandas triton
  uv pip install -q --python "$VENVS/dlrm/bin/python" -e "$KB" --no-deps
fi

log "6. Isolated venv: PointTransformerV3 (spconv, torch 2.9)"
if [ ! -x "$VENVS/ptv3/bin/python" ]; then
  uv venv "$VENVS/ptv3" --python 3.12
  uv pip install -q --python "$VENVS/ptv3/bin/python" \
    "torch==2.9.1" spconv-cu126 "datasets==4.0.0" "pyarrow==18.1.0" \
    einops numpy safetensors triton addict timm h5py scipy huggingface_hub hf_transfer
  uv pip install -q --python "$VENVS/ptv3/bin/python" -e "$KB" --no-deps
fi

log "7. Isolated venv: 3DGS (gsplat, torch 2.9)"
if [ ! -x "$VENVS/gs/bin/python" ]; then
  uv venv "$VENVS/gs" --python 3.12
  CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" \
    uv pip install -q --python "$VENVS/gs/bin/python" \
    "torch==2.9.1" gsplat numpy safetensors triton ninja jaxtyping huggingface_hub hf_transfer pillow
  uv pip install -q --python "$VENVS/gs/bin/python" -e "$KB" --no-deps
fi

log "8. Isolated venv: OpenPI (Pi0 reference)"
if [ ! -x "$VENVS/openpi/bin/python" ]; then
  uv venv "$VENVS/openpi" --python 3.11
  uv pip install -q --python "$VENVS/openpi/bin/python" -e "$REF/openpi"
  uv pip install -q --python "$VENVS/openpi/bin/python" transformers==4.53.2
  OPT_TF=$("$VENVS/openpi/bin/python" -c "import os,transformers;print(os.path.dirname(transformers.__file__))")
  cp -r "$REF/openpi/src/openpi/models_pytorch/transformers_replace/"* "$OPT_TF/"
fi

log "9. Pi0 checkpoint download + convert"
# openpi pins torch 2.7.1+cu126, which has no sm_100 kernels -- its PyTorch
# reference dies with "no kernel image is available for execution on the device"
# on Blackwell. Move that venv to a cu128 build.
uv pip install -q --python "$VENVS/openpi/bin/python" "torch==2.9.1" torchvision
PI0_DL=/home/yak/data-fast/pi0_ckpt
if [ ! -d "$PI0_DL/pi0_aloha_pen_uncap_pytorch" ]; then
  mkdir -p "$PI0_DL"
  python - <<PY
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
fs.get('openpi-assets/checkpoints/pi0_aloha_pen_uncap', "$PI0_DL/pi0_aloha_pen_uncap", recursive=True)
PY
  JAX_PLATFORMS=cpu "$VENVS/openpi/bin/python" "$REF/openpi/examples/convert_jax_model_to_pytorch.py" \
    --config-name pi0_aloha_pen_uncap \
    --checkpoint-dir "$PI0_DL/pi0_aloha_pen_uncap/pi0_aloha_pen_uncap" \
    --output-path "$PI0_DL/pi0_aloha_pen_uncap_pytorch"
  mkdir -p "$PI0_DL/pi0_aloha_pen_uncap_pytorch/assets"
  cp -r "$PI0_DL/pi0_aloha_pen_uncap/pi0_aloha_pen_uncap/assets/"* \
     "$PI0_DL/pi0_aloha_pen_uncap_pytorch/assets/" 2>/dev/null || true
fi

log "10. Isolated venv: vLLM 0.20.1 (Gemma-4 baseline; vLLM 0.18 has no gemma4)"
if [ ! -x "$VENVS/vllm020/bin/python" ]; then
  uv venv "$VENVS/vllm020" --python 3.12
  uv pip install -q --python "$VENVS/vllm020/bin/python" "vllm==0.20.1"
  uv pip install -q --python "$VENVS/vllm020/bin/python" \
    "transformers==5.8.0" fastsafetensors datasets pillow hf_transfer librosa soundfile av qwen-omni-utils
  uv pip install -q --python "$VENVS/vllm020/bin/python" -e "$KB" --no-deps
fi

log "11. Isolated venv: SGLang 0.5.9 (EAGLE-3 reference; base env has no sglang)"
uv venv "$VENVS/sglang" --python 3.12
uv pip install -q --python "$VENVS/sglang/bin/python" "sglang[all]"
uv pip install -q --python "$VENVS/sglang/bin/python" -e "$KB" --no-deps
# Pass --sglang-python to tests/bench_sglang.py. On Blackwell also pass (or let
# it default to) --attention-backend flashinfer: SGLang's EAGLE-3 default is FA3,
# which asserts SM<=90 and otherwise leaves the row with no reference.

log "12. Microsoft BitNet GPU reference (BitNet row)"
# Without these three artifacts bench_microsoft_bitnet.py logs [skip-sota] and
# writes "sota": null, i.e. a paper row with no baseline.
( cd "$REF/BitNet/gpu/bitnet_kernels"
  # Keep upstream's compute_80 PTX (JITs on sm100) and add a native sm_100 binary.
  CUDA_HOME="$CUDA_129" PATH="$CUDA_129/bin:$PATH" nvcc -std=c++17 \
    -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared \
    bitnet_kernels.cu -lcuda \
    -gencode=arch=compute_80,code=compute_80 \
    -gencode=arch=compute_100,code=sm_100 \
    -o libbitnet.so )
# BitNet's model.py imports xformers.ops; 0.0.34 is the build pinned to exactly
# torch 2.10.0, so --no-deps leaves the env's torch alone.
uv pip install -q --no-deps "xformers==0.0.34"
uv pip install -q fire            # BitNet's generate.py entry point
( cd "$REF/BitNet/gpu"
  mkdir -p checkpoints
  SNAP=$(ls -d "$HF_HOME"/hub/models--microsoft--bitnet-b1.58-2B-4T-bf16/snapshots/*/ | head -1)
  python ./convert_safetensors.py --safetensors_file "${SNAP}model.safetensors" \
    --output checkpoints/model_state.pt --model_name 2B
  python ./convert_checkpoint.py --input ./checkpoints/model_state.pt
  rm -f ./checkpoints/model_state.pt )

log "DONE"
ls -1 "$VENVS" 2>/dev/null
