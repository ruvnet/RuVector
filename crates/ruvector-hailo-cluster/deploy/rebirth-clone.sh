#!/usr/bin/env bash
# Rebirth a freshly-cloned cognitum SD card into a new cluster node.
#
# Run on the host that did the dd clone (NOT on the Pi). Operates on
# the cloned SD card before its first boot, scrubbing identity from
# the source so the new Pi joins the tailnet as a separate node.
#
# What it does (in order):
#   1. growpart + resize2fs partition 2 to fill the device
#   2. mount partition 2 as rootfs
#   3. set /etc/hostname + /etc/hosts to the new name
#   4. disable cloud-init's manage_etc_hosts/hostname (else it reverts step 3)
#   5. enable persistent journald (so first-boot failures are debuggable)
#   6. seed RUVECTOR_REBIRTH_PUBKEY into ~genesis/.ssh/authorized_keys
#   7. clear /etc/machine-id (systemd regenerates on first boot)
#   8. delete /etc/ssh/ssh_host_*  (sshd regenerates on first boot)
#   9. clear /var/lib/tailscale/tailscaled.state*  (re-auths as new node)
#  10. clear /root/.bash_history, ~genesis/.bash_history
#  11. clear /var/log/journal/*, /var/log/wtmp, /var/log/btmp
#  12. sync + unmount
#
# Idempotent: re-runnable on the same card.
#
# Usage:
#   sudo bash rebirth-clone.sh <device> <new-hostname>
#
# Optional env vars:
#   RUVECTOR_REBIRTH_PUBKEY="ssh-ed25519 AAAA... operator@host"
#       Seed an SSH pubkey into ~genesis/.ssh/authorized_keys so you can
#       SSH the node from a known operator host the moment it joins WiFi.
#
# Example:
#   RUVECTOR_REBIRTH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" \
#     sudo -E bash rebirth-clone.sh /dev/sdd cognitum-v1

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <device> <new-hostname>" >&2
  echo "example: $0 /dev/sdd cognitum-v1" >&2
  exit 1
fi

DEV="$1"
NEW_HOSTNAME="$2"

# ---- sanity checks ----------------------------------------------------------

if [[ ! -b "$DEV" ]]; then
  echo "not a block device: $DEV" >&2; exit 1
fi

# refuse to scribble on the host's own root or boot disk
HOST_ROOT_DEV=$(findmnt -no SOURCE / | sed 's/[0-9]*$//')
HOST_BOOT_DEV=$(findmnt -no SOURCE /boot 2>/dev/null | sed 's/[0-9]*$//' || true)
if [[ "$DEV" == "$HOST_ROOT_DEV" || "$DEV" == "$HOST_BOOT_DEV" ]]; then
  echo "refusing to operate on host's own disk ($DEV)" >&2; exit 1
fi

# ensure it looks like a freshly-dd'd Pi card: p1 vfat boot, p2 ext4 root
P1="${DEV}1"
P2="${DEV}2"
if [[ ! -b "$P1" || ! -b "$P2" ]]; then
  echo "expected ${P1} and ${P2} to exist (Pi layout: vfat boot + ext4 root)" >&2
  echo "did you run partprobe $DEV after dd?" >&2
  exit 1
fi

# unmount anything auto-mounted from this device (GNOME)
for m in $(mount | awk -v d="$DEV" '$1 ~ "^"d {print $1}'); do
  echo "unmounting $m"
  umount "$m" || true
done

# validate hostname
if [[ ! "$NEW_HOSTNAME" =~ ^[a-z][a-z0-9-]{0,62}$ ]]; then
  echo "invalid hostname: $NEW_HOSTNAME" >&2
  echo "must match [a-z][a-z0-9-]{0,62} (RFC 1123 subset)" >&2
  exit 1
fi

# ---- step 1: grow rootfs partition + filesystem -----------------------------

echo "==> growpart $DEV 2"
# growpart returns 1 if no growth needed; that's fine on re-runs
growpart "$DEV" 2 || true
partprobe "$DEV"
sleep 1

echo "==> e2fsck -f $P2"
# e2fsck exit codes: 0=clean, 1=errors corrected (still success),
# 2=corrected but reboot required (also success for our offline use),
# >=4 are real failures.
set +e
e2fsck -fy "$P2"
fsck_rc=$?
set -e
if (( fsck_rc > 2 )); then
  echo "e2fsck failed with rc=$fsck_rc" >&2
  exit "$fsck_rc"
fi

echo "==> resize2fs $P2"
resize2fs "$P2"

# ---- step 2: mount rootfs ---------------------------------------------------

MNT=$(mktemp -d -t cognitum-rebirth.XXXXXX)
trap 'umount "$MNT/boot/firmware" 2>/dev/null || true; umount "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true' EXIT

echo "==> mount $P2 -> $MNT"
mount "$P2" "$MNT"

# also mount bootfs in case we want to write to /boot/firmware later
if [[ -d "$MNT/boot/firmware" ]]; then
  echo "==> mount $P1 -> $MNT/boot/firmware"
  mount "$P1" "$MNT/boot/firmware"
fi

# ---- step 3: hostname -------------------------------------------------------

OLD_HOSTNAME=$(cat "$MNT/etc/hostname" 2>/dev/null | tr -d '\n' || echo "")
echo "==> hostname: $OLD_HOSTNAME -> $NEW_HOSTNAME"
echo "$NEW_HOSTNAME" > "$MNT/etc/hostname"

# replace OLD_HOSTNAME wherever it appears in /etc/hosts
if [[ -n "$OLD_HOSTNAME" && -f "$MNT/etc/hosts" ]]; then
  sed -i "s/\b${OLD_HOSTNAME}\b/${NEW_HOSTNAME}/g" "$MNT/etc/hosts"
fi
# guarantee a 127.0.1.1 line for the new hostname
if ! grep -qE "^127\.0\.1\.1\s+${NEW_HOSTNAME}\b" "$MNT/etc/hosts" 2>/dev/null; then
  echo "127.0.1.1   ${NEW_HOSTNAME}" >> "$MNT/etc/hosts"
fi

# ---- step 3.5: cloud-init -----------------------------------------------------
# Pi OS Bookworm/Trixie ships cloud-init. By default it has
# manage_etc_hosts: true and preserve_hostname: false, which means it
# rewrites /etc/hostname and /etc/hosts on EVERY boot from cached
# instance metadata — undoing step 3. We disable it two ways:
#   1. drop a cloud.cfg.d override (preserves hostname even if cloud-init
#      gets re-enabled later)
#   2. touch /etc/cloud/cloud-init.disabled (skips cloud-init entirely)

if [[ -d "$MNT/etc/cloud" ]]; then
  echo "==> disable cloud-init hostname management"
  mkdir -p "$MNT/etc/cloud/cloud.cfg.d"
  cat > "$MNT/etc/cloud/cloud.cfg.d/99-rebirth-clone.cfg" <<'EOF'
# rebirth-clone.sh: stop cloud-init from re-applying source-image hostname
preserve_hostname: true
manage_etc_hosts: false
EOF
  touch "$MNT/etc/cloud/cloud-init.disabled"
fi

# ---- step 3.6: persistent journald --------------------------------------------
# default Pi OS journald is volatile (Storage=auto, no /var/log/journal),
# so first-boot failures leave no logs. Enable persistent storage.

echo "==> enable persistent journald"
mkdir -p "$MNT/etc/systemd/journald.conf.d" "$MNT/var/log/journal"
cat > "$MNT/etc/systemd/journald.conf.d/persistent.conf" <<'EOF'
[Journal]
Storage=persistent
EOF

# ---- step 3.7: seed authorized_keys --------------------------------------------
# RUVECTOR_REBIRTH_PUBKEY env var lets you inject a pubkey at rebirth
# time so the new Pi is reachable from a known host immediately
# (without needing console + tailscale-up). Useful when you are
# bringing up many nodes from one operator workstation.

if [[ -n "${RUVECTOR_REBIRTH_PUBKEY:-}" ]]; then
  echo "==> seed RUVECTOR_REBIRTH_PUBKEY into ~genesis/.ssh/authorized_keys"
  GEN_HOME="$MNT/home/genesis"
  if [[ -d "$GEN_HOME" ]]; then
    GEN_UID=$(stat -c %u "$GEN_HOME")
    GEN_GID=$(stat -c %g "$GEN_HOME")
    install -d -m 0700 -o "$GEN_UID" -g "$GEN_GID" "$GEN_HOME/.ssh"
    if ! grep -qF "$RUVECTOR_REBIRTH_PUBKEY" "$GEN_HOME/.ssh/authorized_keys" 2>/dev/null; then
      echo "$RUVECTOR_REBIRTH_PUBKEY" >> "$GEN_HOME/.ssh/authorized_keys"
    fi
    chmod 600 "$GEN_HOME/.ssh/authorized_keys"
    chown "$GEN_UID:$GEN_GID" "$GEN_HOME/.ssh/authorized_keys"
  else
    echo "warning: $GEN_HOME not present, skipping pubkey seed" >&2
  fi
fi

# ---- step 4: machine-id -----------------------------------------------------

echo "==> clear /etc/machine-id (systemd will regenerate)"
: > "$MNT/etc/machine-id"
# /var/lib/dbus/machine-id is usually a symlink; if not, clear it too
if [[ -f "$MNT/var/lib/dbus/machine-id" && ! -L "$MNT/var/lib/dbus/machine-id" ]]; then
  : > "$MNT/var/lib/dbus/machine-id"
fi

# ---- step 5: ssh host keys --------------------------------------------------
# IMPORTANT: don't just delete and rely on the Pi OS one-shot regen
# service — on a cloned image that service has already marked itself
# completed and was disabled. So missing host keys = sshd refuses to
# start = no remote shell on first boot. Instead, regenerate the keys
# directly here so the new node has unique keys AND sshd works.

echo "==> regenerate SSH host keys (unique to this clone)"
rm -fv "$MNT"/etc/ssh/ssh_host_*
ssh-keygen -A -f "$MNT"
ls "$MNT/etc/ssh/" | grep ssh_host

# ---- step 6: tailscale state ------------------------------------------------

if [[ -d "$MNT/var/lib/tailscale" ]]; then
  echo "==> clear tailscale state (forces re-auth as new node)"
  rm -fv "$MNT/var/lib/tailscale/tailscaled.state"
  rm -fv "$MNT/var/lib/tailscale/tailscaled.log"*
  # keep the tailscaled binary; only state is identity-bearing
fi

# ---- step 7: bash history ---------------------------------------------------

echo "==> clear bash histories"
rm -fv "$MNT/root/.bash_history" 2>/dev/null || true
for u in "$MNT"/home/*; do
  [[ -d "$u" ]] || continue
  rm -fv "$u/.bash_history" 2>/dev/null || true
done

# ---- step 8: logs -----------------------------------------------------------

echo "==> truncate logs"
rm -rfv "$MNT"/var/log/journal/* 2>/dev/null || true
: > "$MNT/var/log/wtmp" 2>/dev/null || true
: > "$MNT/var/log/btmp" 2>/dev/null || true
: > "$MNT/var/log/lastlog" 2>/dev/null || true
# don't touch syslog/auth.log/dpkg.log — useful breadcrumbs after first boot

# ---- step 9: optional ruvector worker reset ---------------------------------
# the cloned card will keep cognitum-v0's worker config + models. that's
# fine — the worker has no host-specific state. but clear cached metrics.
if [[ -d "$MNT/var/lib/ruvector-hailo" ]]; then
  echo "==> clear ruvector worker runtime state (keep models)"
  find "$MNT/var/lib/ruvector-hailo" \
    -mindepth 1 -maxdepth 1 \
    -not -name models -not -name '.*' \
    -exec rm -rfv {} + 2>/dev/null || true
fi

# ---- finalize ---------------------------------------------------------------

sync
echo
echo "rebirth complete: ${OLD_HOSTNAME:-(unknown)} -> $NEW_HOSTNAME on $DEV"
echo "next steps:"
echo "  1. eject the card: sudo eject $DEV"
echo "  2. boot it on the new Pi"
echo "  3. on the new Pi: sudo tailscale up   (re-auth as new node)"
echo "  4. approve the new node in https://login.tailscale.com/admin/machines"
echo "  5. verify worker: sudo systemctl status ruvector-hailo-worker"
