#!/usr/bin/env bash
# Gate for linux-arm64-gnu native binaries before they are uploaded/published.
#
#   1. No LDAPR (ARMv8.3 RCpc) instructions. Cortex-A72/A53 (Pi 4, Jetson
#      Nano, Pi Zero 2 W) are ARMv8.0 and SIGILL on the first one.
#   2. Highest GLIBC_ symbol version <= MAX_GLIBC (default 2.34), so the
#      binary loads on Ubuntu 22.04 / Debian 12 / JetPack-era userlands.
#
# Usage: check-arm64-node-binary.sh FILE.node [FILE.node ...]
# Needs aarch64-linux-gnu-objdump (binutils-aarch64-linux-gnu, pulled in by
# gcc-aarch64-linux-gnu). Override with OBJDUMP=... if it lives elsewhere.
set -euo pipefail

MAX_GLIBC="${MAX_GLIBC:-2.34}"
OBJDUMP="${OBJDUMP:-aarch64-linux-gnu-objdump}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 FILE.node [FILE.node ...]" >&2
  exit 2
fi
command -v "$OBJDUMP" >/dev/null || { echo "::error::$OBJDUMP not found" >&2; exit 2; }

status=0
for f in "$@"; do
  [ -f "$f" ] || { echo "::error::$f does not exist"; status=1; continue; }

  ldapr=$("$OBJDUMP" -d "$f" | grep -c ldapr || true)
  glibc=$("$OBJDUMP" -T "$f" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -uV | tail -1)
  glibc="${glibc:-none}"
  echo "$f: ldapr=$ldapr max_glibc=$glibc (limit $MAX_GLIBC)"

  if [ "$ldapr" -ne 0 ]; then
    echo "::error file=$f::$ldapr LDAPR instruction(s) found; this binary SIGILLs on ARMv8.0 cores (Pi 4, Jetson)"
    status=1
  fi
  if [ "$glibc" != "none" ] && [ "$(printf '%s\n%s\n' "$glibc" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
    echo "::error file=$f::requires GLIBC_$glibc, above the GLIBC_$MAX_GLIBC floor"
    status=1
  fi
done
exit "$status"
