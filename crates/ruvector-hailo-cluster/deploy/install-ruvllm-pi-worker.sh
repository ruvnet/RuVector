#!/usr/bin/env bash
# Install ruvllm-pi-worker on a Pi 5 (ADR-179).
#
# Sibling installer to install.sh (which deploys ruvector-hailo-worker).
# Same pattern: drop binary, create system user, install env file,
# install systemd unit, enable.
#
# Idempotent — re-run after upgrading the binary.
#
# Drops on the Pi:
#   /usr/local/bin/ruvllm-pi-worker          (binary)
#   /var/lib/ruvllm/                         (state dir, rwx by ruvllm-worker)
#   /var/lib/ruvllm/models/                  (rsync target for model artifacts)
#   /etc/ruvllm-pi-worker.env                (config; preserved if exists)
#   /etc/systemd/system/ruvllm-pi-worker.service
#   system user: ruvllm-worker (no home, no shell)
#
# Usage:
#   sudo bash install-ruvllm-pi-worker.sh /path/to/ruvllm-pi-worker

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path/to/ruvllm-pi-worker>" >&2
  exit 1
fi

WORKER_BIN="$1"
if [[ ! -x "$WORKER_BIN" ]]; then
  echo "binary not executable: $WORKER_BIN" >&2; exit 1
fi

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUVLLM_USER="ruvllm-worker"
RUVLLM_GROUP="ruvllm-worker"

echo "==> ensure system user $RUVLLM_USER exists"
if ! getent passwd "$RUVLLM_USER" >/dev/null; then
  useradd \
    --system \
    --no-create-home \
    --home-dir /var/lib/ruvllm \
    --shell /usr/sbin/nologin \
    --comment "ruvllm-pi-worker (ADR-179)" \
    "$RUVLLM_USER"
  echo "    -> created"
else
  echo "    -> already exists"
fi

echo "==> install binary"
install -o root -g root -m 0755 "$WORKER_BIN" /usr/local/bin/ruvllm-pi-worker

echo "==> create state dirs (rwx for $RUVLLM_USER)"
install -d -o "$RUVLLM_USER" -g "$RUVLLM_GROUP" -m 0750 \
  /var/lib/ruvllm \
  /var/lib/ruvllm/models

echo "==> install /etc/ruvllm-pi-worker.env (skipped if exists)"
if [[ ! -f /etc/ruvllm-pi-worker.env ]]; then
  install -o root -g root -m 0644 \
    "$DEPLOY_DIR/ruvllm-pi-worker.env.example" \
    /etc/ruvllm-pi-worker.env
  echo "    -> wrote default; edit RUVLLM_MODEL_PATH before starting"
else
  echo "    -> existing /etc/ruvllm-pi-worker.env preserved"
fi

echo "==> install systemd unit"
install -o root -g root -m 0644 \
  "$DEPLOY_DIR/ruvllm-pi-worker.service" \
  /etc/systemd/system/ruvllm-pi-worker.service

echo "==> daemon-reload + enable"
systemctl daemon-reload
systemctl enable ruvllm-pi-worker.service

echo
echo "Installed (running as $RUVLLM_USER, no root)."
echo "Stage a model under /var/lib/ruvllm/models/<name>/ then:"
echo "    sudo systemctl start ruvllm-pi-worker"
echo "Tail logs:"
echo "    journalctl -u ruvllm-pi-worker -f"
