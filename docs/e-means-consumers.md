# e-means consumer packages

The reviewed consumer integration adds explicit clustering subpaths to
`@ruvector/core@0.1.38`, `ruvector`, `@ruvector/all`, and
`@ruvector/wasm@0.2.2`. These facades use the shared
`@ruvector/e-means@0.1.1` runtime. Their existing root APIs remain separate
from the clustering subpaths.

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

The Rust consumer adapters in `ruvector-core`, `ruvector-pq-search`, and
`ruvector-rairs` retain explicit e-means training entrypoints. They consume
the unchanged shared crate; this integration does not republish or modify
the sealed shared runtime. Consumer publication and registry installation
results require their own exact-source evidence after shared publication.
