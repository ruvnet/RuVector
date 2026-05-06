#!/usr/bin/env bash
# Install ruview-vitals-worker on a cluster Pi (ADR-183 Tier 1).
#
# Same idempotent shape as install-ruview-csi-bridge.sh /
# install-ruvllm-pi-worker.sh. Drops:
#
#   /usr/local/bin/ruview-vitals-worker
#   /var/lib/ruview-vitals/                      (state dir)
#   /etc/ruview-vitals-worker.env                (config; preserved)
#   /etc/systemd/system/ruview-vitals-worker.service
#   system user: ruvllm-vitals (no home, no shell)
#
# Usage:
#   sudo bash install-ruview-vitals-worker.sh /path/to/ruview-vitals-worker
#
# Safe to re-run; binary is replaced atomically, env is preserved.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path/to/ruview-vitals-worker>" >&2
  exit 1
fi

WORKER_BIN="$1"
if [[ ! -x "$WORKER_BIN" ]]; then
  echo "binary not executable: $WORKER_BIN" >&2; exit 1
fi

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="ruvllm-vitals"
GROUP_NAME="ruvllm-vitals"

echo "==> ensure system user $USER_NAME exists"
if ! getent passwd "$USER_NAME" >/dev/null; then
  useradd \
    --system \
    --no-create-home \
    --home-dir /nonexistent \
    --shell /usr/sbin/nologin \
    --user-group \
    "$USER_NAME"
fi

echo "==> install binary -> /usr/local/bin/ruview-vitals-worker"
install -o root -g root -m 0755 "$WORKER_BIN" /usr/local/bin/ruview-vitals-worker

echo "==> ensure state dir /var/lib/ruview-vitals"
install -d -o "$USER_NAME" -g "$GROUP_NAME" -m 0750 /var/lib/ruview-vitals

echo "==> ensure config /etc/ruview-vitals-worker.env (preserve if present)"
if [[ ! -f /etc/ruview-vitals-worker.env ]]; then
  install -o root -g root -m 0640 \
    "$DEPLOY_DIR/ruview-vitals-worker.env.example" \
    /etc/ruview-vitals-worker.env
  echo "    (installed default — edit /etc/ruview-vitals-worker.env)"
else
  echo "    (existing /etc/ruview-vitals-worker.env preserved)"
fi

echo "==> ensure LoRA adapter dir /usr/local/share/ruvector (group-writable for SONA saves)"
install -d -o root -g "$GROUP_NAME" -m 0775 /usr/local/share/ruvector
# Fix permissions on any pre-existing adapter JSON files so the worker can overwrite them.
find /usr/local/share/ruvector -name 'node-*.json' -exec chgrp "$GROUP_NAME" {} \; -exec chmod g+w {} \; 2>/dev/null || true

echo "==> install systemd unit"
install -o root -g root -m 0644 \
  "$DEPLOY_DIR/ruview-vitals-worker.service" \
  /etc/systemd/system/ruview-vitals-worker.service

echo "==> reload systemd"
systemctl daemon-reload

echo "==> enable + restart"
systemctl enable ruview-vitals-worker.service
systemctl restart ruview-vitals-worker.service

echo "==> done."
echo "Tail logs:"
echo "  journalctl -u ruview-vitals-worker -f"
echo "Service status:"
echo "  systemctl status ruview-vitals-worker"
