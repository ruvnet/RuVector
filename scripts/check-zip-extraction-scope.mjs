#!/usr/bin/env node
// Temporary reachability guard for the ZIP extraction advisories.
// Revalidates the exact Cargo.lock graph and scans checksum-verified crate sources.
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const lock = await readFile(join(root, "Cargo.lock"), "utf8");
const expectedZip = new Map([
  ["1.1.4", { parent: "candle-core", parentVersion: "0.8.4", parentChecksum: "06ccf5ee3532e66868516d9b315f73aec9f34ea1a37ae98514534d458915dbf1", checksum: "9cc23c04387f4da0374be4533ad1208cbb091d5c11d070dfef13676ad6497164" }],
  ["2.4.2", { parent: "ndarray-npy", parentVersion: "0.9.1", parentChecksum: "7b313788c468c49141a9d9b6131fc15f403e6ef4e8446a0b2e18f664ddb278a9", checksum: "fabe6324e908f85a1c52063ce7aa26b68dcb7eb6dbc83a2d148403c9bc3eba50" }],
  ["7.2.0", { parent: "candle-core", parentVersion: "0.9.2", parentChecksum: "c15b675b80d994b2eadb20a4bbe434eabeb454eac3ee5e2b4cf6f147ee9be091", checksum: "c42e33efc22a0650c311c2ef19115ce232583abbe80850bc8b66509ebef02de0" }],
]);
const blocks = lock.split(/\n\[\[package\]\]\n/).slice(1).map(text => {
  const name = text.match(/^name = "([^"]+)"/m)?.[1];
  const version = text.match(/^version = "([^"]+)"/m)?.[1];
  const checksum = text.match(/^checksum = "([^"]+)"/m)?.[1];
  const deps = [...text.matchAll(/^\s+"([^"]+)",?\s*$/gm)].map(m => m[1]);
  return { name, version, checksum, deps, text };
});
const zips = blocks.filter(p => p.name === "zip");
const gotVersions = zips.map(p => p.version).sort();
const wantVersions = [...expectedZip.keys()].sort();
if (JSON.stringify(gotVersions) !== JSON.stringify(wantVersions)) {
  throw new Error("ZIP lock graph changed; review and update this temporary exception");
}
for (const [version, expected] of expectedZip) {
  const zip = zips.find(p => p.version === version);
  if (zip.checksum !== expected.checksum) throw new Error("ZIP checksum changed: " + version);
  const parents = blocks.filter(p => p.deps.includes("zip " + version));
  if (parents.length !== 1 || parents[0].name !== expected.parent || parents[0].version !== expected.parentVersion) {
    throw new Error("Unexpected reverse dependency for zip " + version + ": " + parents.map(p => p.name + " " + p.version).join(", "));
  }
  if (parents[0].checksum !== expected.parentChecksum) throw new Error("Parent checksum changed: " + expected.parent + " " + expected.parentVersion);
}
if (blocks.some(p => p.name === "zip" && !expectedZip.has(p.version))) throw new Error("Unexpected ZIP version in lock graph");

const temp = await mkdtemp(join(tmpdir(), "ruvector-zip-scope-"));
let scanned = 0;
try {
  for (const [version, expected] of expectedZip) {
    for (const spec of [{ name: "zip", version, checksum: expected.checksum }, { name: expected.parent, version: expected.parentVersion, checksum: expected.parentChecksum }]) {
      const filename = spec.name + "-" + spec.version + ".crate";
      const archive = join(temp, filename);
      const response = await fetch("https://static.crates.io/crates/" + spec.name + "/" + filename);
      if (!response.ok) throw new Error("Crates.io download failed for " + filename + ": HTTP " + response.status);
      const bytes = Buffer.from(await response.arrayBuffer());
      const digest = createHash("sha256").update(bytes).digest("hex");
      if (digest !== spec.checksum) throw new Error("Cargo.lock checksum mismatch for " + filename + ": " + digest);
      await writeFile(archive, bytes);
      const extracted = spawnSync("tar", ["-xzf", archive, "-C", temp], { encoding: "utf8" });
      if (extracted.status !== 0) throw new Error("Could not unpack " + filename + ": " + extracted.stderr);
      console.log("verified " + filename + " sha256=" + digest);
    }
    const sourceRoot = join(temp, expected.parent + "-" + expected.parentVersion, "src");
    const pending = [sourceRoot];
    while (pending.length) {
      const dir = pending.pop();
      for (const entry of await readdir(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name);
        if (entry.isDirectory()) pending.push(path);
        else if (entry.isFile() && entry.name.endsWith(".rs")) {
          scanned++;
          const source = await readFile(path, "utf8");
          if (/\.\s*extract\s*\(|\bZipStreamReader\b/.test(source)) {
            throw new Error("Extraction API reachable in " + expected.parent + " source: " + path);
          }
        }
      }
    }
  }
} finally {
  await rm(temp, { recursive: true, force: true });
}
console.log("PASS: " + scanned + " parent Rust source files scanned; no ZIP extraction API use; exact lock graph and crate checksums verified.");
