#!/usr/bin/env bash
# Build tiny-cuda-nn with the native networks used by current models.
set -euo pipefail
smolgs_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
smolgs_python="${SMOLGS_PYTHON:-python}"
smolgs_revision=3c7931caab7b691c6f30dffe20952ed727d935b9
smolgs_work="$(mktemp -d -t smolgs-tcnn.XXXXXXXX)"
trap 'rm -rf -- "$smolgs_work"' EXIT
if [[ -n "${SMOLGS_TCNN_SOURCE:-}" ]]; then
    smolgs_source="$(cd -- "$SMOLGS_TCNN_SOURCE" && pwd)"
else
    git clone --depth 1 --branch v2.0 --recurse-submodules --shallow-submodules \
        https://github.com/NVlabs/tiny-cuda-nn.git "$smolgs_work/source"
    smolgs_source="$smolgs_work/source"
fi
[[ "$(git -C "$smolgs_source" rev-parse HEAD)" == "$smolgs_revision" ]] || {
    echo "Expected tiny-cuda-nn revision $smolgs_revision" >&2; exit 1;
}
export CUDA_HOME="${CUDA_HOME:-$smolgs_root/.cuda124}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
# Let the build detect the visible GPU unless the user supplies a target.
smolgs_purelib="$("$smolgs_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
smolgs_includes="$("$smolgs_python" - <<'PY'
import pathlib, sysconfig
root = pathlib.Path(sysconfig.get_path('purelib'))/'nvidia'
print(':'.join(str(p) for p in sorted(root.glob('*/include')) if p.is_dir()))
PY
)"
export CPATH="$smolgs_includes${CPATH:+:$CPATH}"
# Torch wheels provide a versioned NVRTC library but not the linker alias.
smolgs_nvrtc="$smolgs_purelib/nvidia/cuda_nvrtc/lib"
if [[ -f "$smolgs_nvrtc/libnvrtc.so.12" ]]; then
    ln -s "$smolgs_nvrtc/libnvrtc.so.12" "$smolgs_work/libnvrtc.so"
    export LIBRARY_PATH="$smolgs_work${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LDFLAGS="-Wl,-rpath,$smolgs_nvrtc${LDFLAGS:+ $LDFLAGS}"
fi
cd -- "$smolgs_source/bindings/torch"
"$smolgs_python" setup.py build_ext --force bdist_wheel --dist-dir "$smolgs_work/wheels"
"$smolgs_python" -m pip install --no-deps --force-reinstall "$smolgs_work"/wheels/*.whl
cd -- "$smolgs_root"
"$smolgs_python" - <<'PY'
import tinycudann
enabled = bool(tinycudann.modules._C.has_networks())
assert enabled, 'Installed tiny-cuda-nn must include neural network support'
print('tiny-cuda-nn installed:', tinycudann.__file__, 'network_support:', enabled)
PY
