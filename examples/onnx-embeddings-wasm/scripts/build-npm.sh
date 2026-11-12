#!/usr/bin/env bash
# Build the `ruvector-onnx-embeddings-wasm` npm package into ./pkg.
#
# The package ships ONE .wasm binary and three glue entry points:
#   ruvector_onnx_embeddings_wasm.cjs       wasm-pack --target nodejs (CommonJS, require())
#   ruvector_onnx_embeddings_wasm.node.mjs  ESM wrapper over the .cjs glue (Node import())
#   ruvector_onnx_embeddings_wasm.js        wasm-pack --target web (ESM, browsers/workers/bundlers)
# Both wasm-pack targets emit a byte-identical .wasm; the script fails if they ever diverge.
#
# 0.1.0-0.1.2 shipped the nodejs (CommonJS) glue as `.js` under "type":"module", so neither
# require() nor import() could load it. Do not rename the .cjs glue back to .js.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HERE}/pkg"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
NAME=ruvector_onnx_embeddings_wasm

# Replace (not extend) RUSTFLAGS: mold, set in some shells, cannot link wasm32.
# simd128 is what makes simd_available() true and is supported by Node >= 16.4 and
# all current browsers.
export RUSTFLAGS="-C target-feature=+simd128"

cd "${HERE}"
wasm-pack build --release --target nodejs --out-dir "${TMP}/node" --out-name "${NAME}"
wasm-pack build --release --target web --out-dir "${TMP}/web" --out-name "${NAME}"

if ! cmp -s "${TMP}/node/${NAME}_bg.wasm" "${TMP}/web/${NAME}_bg.wasm"; then
  echo "error: nodejs and web targets produced different .wasm binaries" >&2
  exit 1
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"
cp "${TMP}/web/${NAME}_bg.wasm" "${OUT}/"

# Web / bundler / worker ESM glue.
cp "${TMP}/web/${NAME}.js" "${OUT}/${NAME}.js"
cp "${TMP}/web/${NAME}.d.ts" "${OUT}/${NAME}.d.ts"

# Node CommonJS glue: `.cjs` so it stays CommonJS inside a "type":"module" package.
sed "s#@ts-self-types=\"./${NAME}.d.ts\"#@ts-self-types=\"./${NAME}.d.cts\"#" \
  "${TMP}/node/${NAME}.js" > "${OUT}/${NAME}.cjs"
cp "${TMP}/node/${NAME}.d.ts" "${OUT}/${NAME}.d.cts"

# Node ESM wrapper. The export list is derived from the CommonJS glue so it cannot drift.
EXPORTS="$(grep -E '^exports\.[A-Za-z_][A-Za-z0-9_]* *=' "${OUT}/${NAME}.cjs" \
  | sed -E 's/^exports\.([A-Za-z_][A-Za-z0-9_]*).*/\1/' | sort -u | paste -sd, - | sed 's/,/, /g')"
if [ -z "${EXPORTS}" ]; then
  echo "error: no exports found in ${NAME}.cjs" >&2
  exit 1
fi
cat > "${OUT}/${NAME}.node.mjs" <<EOF
// Node.js ESM entry: loads the CommonJS glue, which compiles the wasm synchronously.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wasm = require('./${NAME}.cjs');

export const { ${EXPORTS} } = wasm;

/** No-op on Node (the module is ready on import); kept for parity with the web entry. */
export function initSync() {
  return wasm;
}

/** No-op on Node (the module is ready on import); kept for parity with the web entry. */
export default async function initWasm() {
  return wasm;
}
EOF
{
  echo "export * from './${NAME}.cjs';"
  echo "export function initSync(): unknown;"
  echo "export default function initWasm(): Promise<unknown>;"
} > "${OUT}/${NAME}.node.d.mts"

cp loader.js parallel-embedder.mjs parallel-worker.mjs README.md "${OUT}/"
cp ../../LICENSE "${OUT}/LICENSE"

VERSION="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' Cargo.toml | head -1)"
node -e '
  const fs = require("fs");
  const pkg = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  pkg.version = process.argv[2];
  fs.writeFileSync(process.argv[3], JSON.stringify(pkg, null, 2) + "\n");
' "${HERE}/npm-package.json" "${VERSION}" "${OUT}/package.json"

echo "built ${OUT} (version ${VERSION})"
