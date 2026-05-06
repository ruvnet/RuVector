#!/usr/bin/env bash
# cluster-smoke-test.sh — ADR-183 Tier 2 iter 13
#
# Integration smoke test for the full ruview vitals + brain stack.
# Covers ADR-183 (vitals/brain), ADR-184 (H10H cluster-3), ADR-185
# (H10H v0 + LLM router), ADR-018 (CSI bridge + H8 embedding workers).
# 38 assertions total. Exits 0 only when all pass.
#
# Usage:
#   bash cluster-smoke-test.sh [--quiet]
#
#   --quiet  suppress pass lines; show only failures + final verdict

set -euo pipefail

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

# Tailscale IPs / hostnames per ADR-183
WORKERS=(
  "root@100.80.54.16:cognitum-cluster-1:50055"
  "root@100.77.220.24:cognitum-cluster-2:50055"
  "root@100.73.75.53:cognitum-cluster-3:50055"
)
V0_HOST="genesis@100.77.59.83"
V0_BRAIN_PORT=9876
V0_GRPC_PORT=50054
V0_SERVICES=(
  "ruview-vitals-worker"
  "ruview-mcp-brain-mini"
  "ruview-pointcloud"
  "ruview-csi-sink"
)
WORKER_SERVICES=(
  "ruview-vitals-worker"
)

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); [[ $QUIET -eq 0 ]] && echo "  [PASS] $*" || true; }
fail() { FAIL=$((FAIL + 1)); echo "  [FAIL] $*"; }

check_service() {
  local host="$1" name="$2"
  local status
  status=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" "systemctl is-active $name 2>&1" 2>&1 || true)
  if [[ "$status" == "active" ]]; then
    pass "$name active on $host"
  else
    fail "$name not active on $host (status=$status)"
  fi
}

check_grpc() {
  local host_ssh="$1" label="$2" port="$3"
  # Use netcat to verify the port is open — full gRPC health RPC would need grpcurl.
  local open
  open=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host_ssh" \
    "timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/$port' 2>&1 && echo open || echo closed" 2>&1 || echo closed)
  if [[ "$open" == "open" ]]; then
    pass "gRPC :$port open on $label"
  else
    fail "gRPC :$port not open on $label"
  fi
}

check_sona_steps() {
  local host="$1" label="$2" min_steps="$3"
  local steps
  steps=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "journalctl -u ruview-vitals-worker --no-pager -n 500 -o cat 2>&1 | grep 'sona: gradient step' | tail -1 | grep -oP 'steps=\K[0-9]+'" 2>&1 || echo 0)
  steps="${steps//[^0-9]/}"
  steps="${steps:-0}"
  if [[ "$steps" -ge "$min_steps" ]]; then
    pass "SONA steps=$steps (≥ $min_steps) on $label"
  else
    fail "SONA steps=$steps (< $min_steps) on $label — adapter not converging"
  fi
}

check_relay() {
  local host="$1" label="$2"
  local has_relay
  # Check env file for RELAY_TARGETS, then check startup journal (may be old),
  # then check runtime log with a wider window.
  has_relay=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "grep -cE 'RUVIEW_VITALS_RELAY_TARGETS=.+' /etc/ruview-vitals-worker.env 2>/dev/null || \
     journalctl -u ruview-vitals-worker --no-pager -n 500 -o cat 2>&1 | grep -c 'UDP relay fan-out up' || echo 0" 2>&1 || echo 0)
  has_relay="${has_relay//[^0-9]/}"
  if [[ "${has_relay:-0}" -gt 0 ]]; then
    pass "relay fan-out active on $label"
  else
    fail "relay fan-out not detected on $label"
  fi
}

check_ruvllm_h10() {
  local host="$1" label="$2" http_port="$3" grpc_port="$4"
  # HTTP health
  local tok_per_sec
  tok_per_sec=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "curl -sf http://127.0.0.1:$http_port/health 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"tok_per_sec\", 0))' 2>/dev/null || echo 0" 2>&1 || echo 0)
  tok_per_sec="${tok_per_sec//[^0-9.]/}"
  local hailo_ok
  hailo_ok=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "curl -sf http://127.0.0.1:$http_port/health 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"hailo_ok\", False))' 2>/dev/null || echo False" 2>&1 || echo False)
  if [[ "${hailo_ok:-False}" == "True" ]]; then
    pass "ruview-ruvllm-h10 hailo_ok=True on $label"
  else
    fail "ruview-ruvllm-h10 hailo_ok not True on $label"
  fi
  # gRPC port open — check from ruvultra via Tailscale (works for root@ or genesis@ prefix)
  local ts_ip="${host##*@}"   # strip everything up to and including @
  local open
  open=$(timeout 3 bash -c "echo > /dev/tcp/${ts_ip}/${grpc_port}" 2>&1 && echo open || echo closed)
  if [[ "$open" == "open" ]]; then
    pass "ruview-ruvllm-h10 gRPC :$grpc_port open on $label"
  else
    fail "ruview-ruvllm-h10 gRPC :$grpc_port not open on $label"
  fi
  # /dev/hailo0
  local dev
  dev=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "test -e /dev/hailo0 && echo ok || echo missing" 2>&1 || echo missing)
  if [[ "${dev:-missing}" == "ok" ]]; then
    pass "/dev/hailo0 present on $label"
  else
    fail "/dev/hailo0 missing on $label"
  fi
}

check_brain_http() {
  local status
  status=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$V0_HOST" \
    "curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:$V0_BRAIN_PORT/health 2>&1 || \
     curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:$V0_BRAIN_PORT/ 2>&1 || echo 000" 2>&1 || echo 000)
  status="${status//[^0-9]/}"
  if [[ "${status:-000}" =~ ^(200|204|404|405)$ ]]; then
    pass "brain HTTP /$V0_BRAIN_PORT reachable on v0 (HTTP $status)"
  else
    fail "brain HTTP /$V0_BRAIN_PORT not reachable on v0 (got $status)"
  fi
}

echo "=== ADR-183 cluster smoke test — $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo ""

echo "-- cognitum-v0 services --"
for svc in "${V0_SERVICES[@]}"; do
  check_service "$V0_HOST" "$svc"
done
check_grpc "$V0_HOST" "cognitum-v0" "$V0_GRPC_PORT"
check_sona_steps "$V0_HOST" "cognitum-v0" 100
check_brain_http

echo ""
echo "-- worker nodes --"
for entry in "${WORKERS[@]}"; do
  host="${entry%%:*}"
  rest="${entry#*:}"
  label="${rest%%:*}"
  port="${rest##*:}"

  for svc in "${WORKER_SERVICES[@]}"; do
    check_service "$host" "$svc"
  done
  check_grpc "$host" "$label" "$port"
  check_sona_steps "$host" "$label" 100
  check_relay "$host" "$label"
done

echo ""
echo "-- ADR-184 Hailo-10H LLM service (cognitum-cluster-3) --"
check_service "root@100.73.75.53" "ruview-ruvllm-h10"
check_ruvllm_h10 "root@100.73.75.53" "cognitum-cluster-3" "8880" "50058"

echo ""
echo "-- ADR-185 Hailo-10H LLM service (cognitum-v0) --"
check_service "$V0_HOST" "ruview-ruvllm-h10"
check_ruvllm_h10 "$V0_HOST" "cognitum-v0" "8880" "50058"

echo ""
echo "-- ADR-185 LLM router (cognitum-v0) --"
check_service "$V0_HOST" "ruview-ruvllm-router"
# Router HTTP /health — expects JSON with backends_healthy > 0
router_health=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$V0_HOST" \
  "curl -sf http://127.0.0.1:8882/health 2>/dev/null" 2>/dev/null || echo '{}')
router_healthy=$(echo "$router_health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('backends_healthy',0))" 2>/dev/null || echo 0)
router_total=$(echo "$router_health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('backends_total',0))" 2>/dev/null || echo 0)
if [[ "${router_healthy:-0}" -ge 1 ]]; then
  pass "router: ${router_healthy}/${router_total} backends healthy on v0"
else
  fail "router: 0 healthy backends on v0 (got $router_health)"
fi
# Router gRPC port reachable from ruvultra
router_ts_ip="100.77.59.83"
router_open=$(timeout 3 bash -c "echo > /dev/tcp/${router_ts_ip}/50060" 2>&1 && echo open || echo closed)
if [[ "$router_open" == "open" ]]; then
  pass "router gRPC :50060 reachable from ruvultra via Tailscale"
else
  fail "router gRPC :50060 not reachable from ruvultra"
fi

echo ""
echo "-- ADR-018 CSI bridge + H8 embedding worker (cluster-1, cluster-2) --"
for csi_entry in "root@100.80.54.16:cognitum-cluster-1" "root@100.77.220.24:cognitum-cluster-2"; do
  csi_host="${csi_entry%%:*}"
  csi_label="${csi_entry##*:}"

  check_service "$csi_host" "ruview-csi-bridge"

  # UDP :5006 must be bound (CSI ingestion — vitals-worker owns :5005)
  udp_bound=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$csi_host" \
    "ss -ulnp 2>/dev/null | grep -cE ':5006\b' || echo 0" 2>&1 || echo 0)
  udp_bound="${udp_bound//[^0-9]/}"
  if [[ "${udp_bound:-0}" -gt 0 ]]; then
    pass "ruview-csi-bridge UDP :5006 bound on $csi_label"
  else
    fail "ruview-csi-bridge UDP :5006 not bound on $csi_label"
  fi

  check_service "$csi_host" "ruvector-hailo-worker"
  check_grpc    "$csi_host" "$csi_label" "50051"
done

echo ""
echo "=== Result: $PASS passed, $FAIL failed ==="

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
