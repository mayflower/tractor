#!/usr/bin/env bash
# Cold offline proof for the local Clairic profile with a real inference bundle.
#
# Copies the bundle to a fresh directory, uses empty model caches, starts the
# real server inside a new network namespace that has only loopback (external
# network is actually unreachable, not just disabled by environment), sends a
# real HTTP request inside that namespace and prints a JSON report.
# Needs: util-linux unshare with user namespaces, iproute2, curl, a prepared
# environment (`just setup`) and an exported bundle. Starts no training and
# downloads nothing.
set -euo pipefail

bundle=${1:?usage: clairic-offline-check.sh BUNDLE [cpu|cuda] [fp32|bf16] [REPORT]}
device=${2:-cpu}
precision=${3:-fp32}
report=${4:-reports/clairic/offline-check.json}
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cp -a "$bundle" "$work/bundle"
mkdir -p "$work/cache" "$(dirname "$report")"
export HF_HOME="$work/cache" XDG_CACHE_HOME="$work/cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
cd "$root"
uv run --no-sync tractor verify-bundle --model "$work/bundle" > "$work/verify.json"

unshare --user --map-root-user --net bash -s "$work" "$device" "$precision" "$root" <<'INNER'
set -euo pipefail
work=$1 device=$2 precision=$3 root=$4
ip link set lo up
cd "$root"
if curl -sS --max-time 5 https://huggingface.co > /dev/null 2>&1; then
  echo "external network is reachable; refusing to report an offline proof" >&2
  exit 1
fi
uv run --no-sync tractor serve --api clairic --model "$work/bundle" --device "$device" \
  --precision "$precision" --host 127.0.0.1 --port 8000 > "$work/server.log" 2>&1 &
server=$!
for _ in $(seq 1 900); do
  curl -fsS http://127.0.0.1:8000/readyz > /dev/null 2>&1 && break
  kill -0 "$server" 2> /dev/null || { tail -n 20 "$work/server.log" >&2; exit 1; }
  sleep 1
done
curl -fsS http://127.0.0.1:8000/v1/capabilities > "$work/capabilities.json"
curl -sS --fail-with-body \
  http://127.0.0.1:8000/client/v4/accounts/local/ai/run/@cf/cloudflare/clairic \
  -H 'Content-Type: application/json' --data-binary @examples/clairic/invoice.json > "$work/response.json"
kill "$server"
wait "$server" || true
INNER

uv run --no-sync python - "$work" "$report" <<'PY'
import json, sys
from pathlib import Path
work, report = Path(sys.argv[1]), Path(sys.argv[2])
result = {
    "scope": "Real local inference offline; not model quality or calibration evidence",
    "network": "isolated namespace with loopback only; external connection refused",
    "bundle": json.loads((work / "verify.json").read_text()),
    "capabilities": json.loads((work / "capabilities.json").read_text()),
    "response": json.loads((work / "response.json").read_text()),
}
report.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
