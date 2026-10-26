#!/usr/bin/env bash
set -euo pipefail
SMOLGS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SMOLGS_PYTHON="${SMOLGS_PYTHON:-python}"
export MAX_JOBS="${MAX_JOBS:-4}"
# Pip CUDA wheels keep library headers under site-packages/nvidia, outside CUDA_HOME.
# ATen/cuda/CUDAContext.h includes cuBLAS/cuSPARSE headers even for grid kernels.
SMOLGS_CUDA_INCLUDES="$("$SMOLGS_PYTHON" -c 'import pathlib, sysconfig; root = pathlib.Path(sysconfig.get_path("purelib"))/"nvidia"; print(":".join(str(p) for p in sorted(root.glob("*/include")) if p.is_dir()))')"
if [[ -n "$SMOLGS_CUDA_INCLUDES" ]]; then
    export CPATH="$SMOLGS_CUDA_INCLUDES${CPATH:+:$CPATH}"
fi
# On machines without accessible devices, explicitly set the target architecture,
# e.g. TORCH_CUDA_ARCH_LIST=8.9 for RTX 4090. Do not change the shared CUDA install.
for smolgs_extension in diff-gaussian-rasterization simple-knn gridencoder arithmetic; do
    "$SMOLGS_PYTHON" -m pip install --no-build-isolation --no-deps "$SMOLGS_ROOT/submodules/$smolgs_extension"
done
