# e-means consumer packages

The reviewed consumer integration adds explicit clustering subpaths to
`@ruvector/core@0.1.39`, `ruvector`, `@ruvector/all`, and
`@ruvector/wasm@0.2.2`. These facades use the shared
public `@ruvector/e-means@0.1.0` runtime while the reviewed shared 0.1.1
generation awaits publication. Their existing root APIs remain separate from
the clustering subpaths. These pins do not qualify the 0.1.1 fixes through
the facades; they preserve ordinary registry installs during staging.

The core and WASM npm packages require Node.js 20 or newer because the shared
runtime is a mandatory dependency with that minimum Node version. Their
package metadata must declare the same minimum so engine-strict installs do
not advertise unsupported Node.js 18 compatibility. The WASM package also
supports browser consumers; its Node engine declaration describes npm/Node
usage, not a browser version requirement.

Publish and verify the shared Cargo and npm archives before qualifying or
publishing these consumer packages against the registry. The shared release
uses the [protected leaf release protocol](./e-means-release.md). A local
workspace link or packed archive can support diagnostic checks but does not
prove that an ordinary registry install works.

After both shared 0.1.1 archives are public and byte-verified, promote the four
npm facade dependency pins and the corresponding three workspace-lock edges
to 0.1.1, then qualify that new consumer source. The shared local npm workspace
remains 0.1.1 during staging; it does not substitute for the facades' public
0.1.0 dependency. The workspace lock carries three nested public 0.1.0 records
for those facades, with registry URLs and integrity hashes. Native Rust adapters retain their local path dependency on
shared 0.1.1. No CI skip, file dependency, or registry override is used to hide
an unpublished dependency.

The Rust consumer adapters in `ruvector-core`, `ruvector-pq-search`, and
`ruvector-rairs` retain explicit e-means training entrypoints. They consume
the unchanged shared crate; this integration does not republish or modify
the sealed shared runtime. Consumer publication and registry installation
results require their own exact-source evidence after shared publication.
