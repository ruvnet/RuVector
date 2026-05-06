#!/usr/bin/env bash
# install-ruview-pointcloud.sh — idempotent installer for ruview-pointcloud
# on cognitum-v0 (the fusion master). ADR-183 Tier 2 iter 7.
#
# Usage: sudo bash install-ruview-pointcloud.sh [/path/to/ruview-pointcloud]
# Run on cognitum-v0 as root (or via sudo).
set -euo pipefail

BINARY="${1:-/usr/local/bin/ruview-pointcloud}"
SERVICE_NAME="ruview-pointcloud"
SERVICE_USER="ruview-pointcloud"
STATE_DIR="/var/lib/ruview-pointcloud"

# --- system user ---
if ! id "${SERVICE_USER}" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin \
        --comment "ruview-pointcloud fusion server" \
        "${SERVICE_USER}"
    # Add to video group for camera access
    usermod -aG video "${SERVICE_USER}" 2>/dev/null || true
    echo "[install] created user ${SERVICE_USER}"
else
    echo "[install] user ${SERVICE_USER} already exists"
    usermod -aG video "${SERVICE_USER}" 2>/dev/null || true
fi

# --- state dir ---
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${STATE_DIR}"
echo "[install] state dir ${STATE_DIR} ready"

# --- binary ---
if [[ "${BINARY}" != "/usr/local/bin/ruview-pointcloud" ]]; then
    install -m 0755 "${BINARY}" /usr/local/bin/ruview-pointcloud
    echo "[install] installed binary from ${BINARY}"
fi
/usr/local/bin/ruview-pointcloud --version 2>/dev/null || \
    /usr/local/bin/ruview-pointcloud --help 2>&1 | head -3

# --- env file (create only if missing) ---
ENV_FILE="/etc/${SERVICE_NAME}.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    cat > "${ENV_FILE}" <<'EOF'
# ruview-pointcloud environment — edit and restart to apply.
#
# Bind to Tailscale IP to serve the viewer across the cluster:
#   RUVIEW_POINTCLOUD_BIND=100.80.54.16:9880
# Default (loopback-only): 127.0.0.1:9880
RUVIEW_POINTCLOUD_BIND=127.0.0.1:9880

# Brain address — must match ruview-mcp-brain-mini.
RUVIEW_BRAIN_URL=http://127.0.0.1:9876
EOF
    chmod 0640 "${ENV_FILE}"
    echo "[install] created ${ENV_FILE}"
else
    echo "[install] ${ENV_FILE} already exists — not overwritten"
fi

# --- systemd unit ---
UNIT_SRC="$(dirname "$0")/${SERVICE_NAME}.service"
UNIT_DST="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ -f "${UNIT_SRC}" ]]; then
    install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"
    echo "[install] installed ${UNIT_DST}"
else
    echo "[install] WARNING: ${UNIT_SRC} not found — systemd unit not updated"
fi

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl status "${SERVICE_NAME}" --no-pager -l | tail -8
echo "[install] done — ${SERVICE_NAME} is running"
