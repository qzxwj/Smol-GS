#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/Smol-GS
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1
out="$PWD/outputs/truck_relu_sfm_fallback_20261022"
if [[ -e "$out" ]]; then echo 'Fallback status already exists; refusing overwrite.' >&2; exit 1; fi
mkdir -p "$out"
exec > >(tee -a "$out/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit_code"' EXIT
while [[ ! -f outputs/truck_relu_gain8_20261022/exit_code ]]; do sleep 5; done
if .env/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('outputs/truck_relu_gain8_20261022')
code=int((root/'exit_code').read_text());gates=[str(p) for p in root.glob('gate_*.json') if not json.loads(p.read_text())['passed']]
dead=(root/'tandt/truck/base/octree_failure.json').exists()
r={'parent_exit':code,'failed_gates':gates,'dead_mlp':dead,'activate_sfm_means':code!=0 and (bool(gates) or dead)}
Path('outputs/truck_relu_sfm_fallback_20261022/decision.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True);raise SystemExit(0 if r['activate_sfm_means'] else 3)
PY
then
 bash scripts/run_truck_relu_sfm_means_screen.sh
else
 echo 'No diagnosed positional failure; no additional full run started.'
fi
