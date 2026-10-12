# RuvNet npm constellation

Observed: 2026-10-11T15:17:39.655Z

Public packages: 398/398. Complete scan: true. Runtime cycles: 1.
Coverage is public maintainer search and latest manifests. Private packages, transitives, source security, and publish permissions are not certified.

| Package | Version | Findings | Runtime ecosystem dependencies |
|---|---|---|---|
| @agentic-robotics/cli | 0.2.3 | no-registry-attestation | @agentic-robotics/core (dependencies) |
| @agentic-robotics/core | 0.2.1 | no-registry-attestation | @agentic-robotics/linux-x64-gnu (optionalDependencies) |
| @agentic-robotics/linux-x64-gnu | 0.2.0 | no-registry-attestation |  |
| @agentic-robotics/mcp | 0.2.2 | no-registry-attestation | @agentic-robotics/core (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @agentic-robotics/self-learning | 1.0.0 | no-registry-attestation | agentdb (dependencies), agentic-flow (dependencies), @agentic-robotics/core (peerDependencies) |
| @agentics.org/agentic-mcp | 1.0.4 | no-registry-attestation |  |
| @agentics.org/sparc2 | 2.0.25 | no-registry-attestation, install-script-review |  |
| @claude-flow/aidefence | 3.0.3 | no-registry-attestation | agentdb (peerDependencies) |
| @claude-flow/browser | 3.0.0 | no-registry-attestation, install-script-review | agentic-flow (dependencies), @claude-flow/cli (peerDependencies) |
| @claude-flow/claims | 3.0.0 | no-registry-attestation | @claude-flow/shared (peerDependencies) |
| @claude-flow/cli | 3.56.5 | no-registry-attestation, install-script-review | @claude-flow/mcp (dependencies), @claude-flow/codex (dependencies), @claude-flow/memory (dependencies), @claude-flow/neural (dependencies), @claude-flow/shared (dependencies), @claude-flow/cli-core (dependencies), @claude-flow/security (dependencies), @ruvector/rabitq-wasm (dependencies), @claude-flow/plugin-agent-federation (dependencies), agentdb (optionalDependencies), ruvector (optionalDependencies), agentic-flow (optionalDependencies), @metaharness/radio (optionalDependencies), @claude-flow/memory (optionalDependencies), @metaharness/darwin (optionalDependencies), @metaharness/flywheel (optionalDependencies), @metaharness/turn-credit (optionalDependencies), metaharness (peerDependencies), @ruvector/typesafe (peerDependencies), @metaharness/router (peerDependencies) |
| @claude-flow/cli-core | 3.7.1 | no-registry-attestation |  |
| @claude-flow/codex | 3.0.3 | no-registry-attestation | @claude-flow/security (dependencies), @claude-flow/cli (peerDependencies) |
| @claude-flow/deployment | 3.0.0 | missing-repository, missing-license, no-registry-attestation | @claude-flow/shared (dependencies) |
| @claude-flow/embeddings | 3.0.0-alpha.45 | missing-repository, no-registry-attestation | agentic-flow (peerDependencies), @ruvector/core (peerDependencies), @ruvector/diskann (peerDependencies), @claude-flow/shared (peerDependencies), @ruvector/attention (peerDependencies), @ruvector/rabitq-wasm (peerDependencies), ruvector-onnx-embeddings-wasm (peerDependencies) |
| @claude-flow/guidance | 3.0.0 | no-registry-attestation | @claude-flow/hooks (dependencies), @claude-flow/memory (dependencies), @claude-flow/shared (dependencies) |
| @claude-flow/hooks | 3.0.0 | no-registry-attestation | @claude-flow/memory (dependencies), @claude-flow/neural (dependencies), @claude-flow/shared (dependencies), @claude-flow/security (dependencies), @claude-flow/shared (peerDependencies) |
| @claude-flow/integration | 3.0.1 | missing-repository, missing-license, no-registry-attestation | agent-booster (optionalDependencies), agentic-flow (peerDependencies) |
| @claude-flow/mcp | 3.1.1 | missing-repository, missing-license, no-registry-attestation |  |
| @claude-flow/memory | 3.0.4 | missing-repository, missing-license, no-registry-attestation | agentdb (dependencies), @claude-flow/security (dependencies) |
| @claude-flow/neural | 3.0.0 | no-registry-attestation | @ruvector/sona (dependencies), @claude-flow/memory (dependencies), agentdb (optionalDependencies) |
| @claude-flow/performance | 3.0.0 | missing-repository, missing-license, no-registry-attestation, dependencies:unbounded:@ruvector/sona, dependencies:unbounded:@ruvector/attention | @ruvector/sona (dependencies), @ruvector/attention (dependencies) |
| @claude-flow/plugin-agent-federation | 1.0.1 | missing-repository, no-registry-attestation | midstreamer (peerDependencies), agentic-flow (peerDependencies) |
| @claude-flow/plugin-agentic-qe | 3.0.0-alpha.5 | no-registry-attestation | @claude-flow/memory (peerDependencies), @claude-flow/plugins (peerDependencies), @claude-flow/security (peerDependencies), @claude-flow/embeddings (peerDependencies) |
| @claude-flow/plugin-code-intelligence | 3.0.0-alpha.3 | no-registry-attestation | @claude-flow/plugins (peerDependencies) |
| @claude-flow/plugin-cognitive-kernel | 3.0.0-alpha.1 | no-registry-attestation |  |
| @claude-flow/plugin-financial-risk | 3.0.0-alpha.1 | no-registry-attestation | ruvector-gnn-wasm (optionalDependencies) |
| @claude-flow/plugin-gastown-bridge | 0.1.4 | no-registry-attestation | @claude-flow/memory (peerDependencies) |
| @claude-flow/plugin-healthcare-clinical | 3.0.0-alpha.2 | no-registry-attestation | ruvector-gnn-wasm (optionalDependencies) |
| @claude-flow/plugin-hyperbolic-reasoning | 3.0.0-alpha.1 | no-registry-attestation | @ruvector/sona (optionalDependencies), @ruvector/gnn-wasm (optionalDependencies), @ruvector/attention-wasm (optionalDependencies) |
| @claude-flow/plugin-iot-cognitum | 1.0.0 | missing-repository, no-registry-attestation | @cognitum-one/sdk (dependencies) |
| @claude-flow/plugin-legal-contracts | 3.0.0-alpha.2 | no-registry-attestation | @claude-flow/plugins (peerDependencies) |
| @claude-flow/plugin-neural-coordination | 3.0.0-alpha.1 | no-registry-attestation |  |
| @claude-flow/plugin-perf-optimizer | 3.0.0-alpha.2 | no-registry-attestation |  |
| @claude-flow/plugin-prime-radiant | 0.1.5 | no-registry-attestation | prime-radiant-advanced-wasm (optionalDependencies), @claude-flow/memory (peerDependencies), @claude-flow/security (peerDependencies) |
| @claude-flow/plugin-quantum-optimizer | 3.0.0-alpha.1 | no-registry-attestation | @ruvector/exotic-wasm (optionalDependencies) |
| @claude-flow/plugin-test-intelligence | 3.0.0-alpha.1 | no-registry-attestation |  |
| @claude-flow/plugins | 3.0.2 | missing-repository, no-registry-attestation | @ruvector/wasm (peerDependencies), @claude-flow/hooks (peerDependencies), @claude-flow/memory (peerDependencies), @ruvector/learning-wasm (peerDependencies) |
| @claude-flow/providers | 3.0.0 | missing-repository, no-registry-attestation | @ruvector/ruvllm (peerDependencies) |
| @claude-flow/security | 3.0.4 | missing-repository, missing-license, no-registry-attestation |  |
| @claude-flow/shared | 3.0.3 | missing-repository, missing-license, no-registry-attestation |  |
| @claude-flow/swarm | 3.0.2 | missing-repository, missing-license, no-registry-attestation |  |
| @claude-flow/teammate-plugin | 1.0.0-alpha.1 | no-registry-attestation | @ruvnet/bmssp (dependencies) |
| @claude-flow/testing | 3.0.0 | missing-repository, missing-license, no-registry-attestation | @claude-flow/swarm (peerDependencies), @claude-flow/memory (peerDependencies), @claude-flow/shared (peerDependencies) |
| @claude-flow/watermark | 0.4.1 | no-registry-attestation |  |
| @cognitum-one/sdk | 0.4.0 |  |  |
| @cognitum/freetokens | 0.1.0 | no-registry-attestation |  |
| @cognitum/gate | 0.1.0 | no-registry-attestation | claude-flow (peerDependencies) |
| @cognitum/ruos | 0.5.0 | no-registry-attestation |  |
| @dream-machine/compile | 0.1.0 | no-registry-attestation |  |
| @dream-machine/ledger | 0.1.0 | no-registry-attestation |  |
| @dream-machine/memory | 0.1.0 | no-registry-attestation, peerDependencies:unbounded:@ruvector/wasm, peerDependencies:unbounded:@ruvector/rvf-wasm | @ruvector/wasm (peerDependencies), @ruvector/rvf-wasm (peerDependencies) |
| @dream-machine/schedule | 0.1.0 | no-registry-attestation | @dream-machine/compile (dependencies) |
| @dream-machine/witness | 0.1.0 | no-registry-attestation |  |
| @foxruv/cf-doctor | 1.1.0 | no-registry-attestation |  |
| @foxruv/e2b-runner | 2.0.1 | no-registry-attestation | agentic-flow (peerDependencies) |
| @foxruv/iris | 1.8.19 | missing-repository, no-registry-attestation, install-script-review | @foxruv/iris-agentic-synth (dependencies), @foxruv/iris-core (dependencies), @foxruv/iris-ultrathink (dependencies), agentdb (dependencies), agentic-flow (dependencies), agentdb (peerDependencies), agentic-flow (peerDependencies) |
| @foxruv/iris-agentic-synth | 1.0.5 | no-registry-attestation, dependencies:unbounded:agentic-robotics | agentic-robotics (dependencies), midstreamer (dependencies), ruvector (optionalDependencies) |
| @foxruv/iris-core | 1.0.0 | no-registry-attestation |  |
| @foxruv/iris-ultrathink | 1.0.0 | no-registry-attestation | agentdb (peerDependencies), agentic-flow (peerDependencies) |
| @foxruv/nova-medicina | 1.0.0 | no-registry-attestation | agentic-flow (dependencies), agentdb (dependencies), claude-flow (dependencies), agentic-flow (peerDependencies) |
| @metaharness/agntcy | 0.1.0 | no-registry-attestation |  |
| @metaharness/avo | 0.2.0 | missing-repository, no-registry-attestation | @metaharness/horizon (dependencies), agenticow (peerDependencies) |
| @metaharness/aws-finops | 0.1.0 | no-registry-attestation |  |
| @metaharness/claude-code | 0.1.1 | no-registry-attestation |  |
| @metaharness/codex | 0.1.0 | no-registry-attestation |  |
| @metaharness/coding | 0.1.0 | no-registry-attestation |  |
| @metaharness/copilot | 0.1.0 | no-registry-attestation |  |
| @metaharness/darwin | 0.10.3 | no-registry-attestation |  |
| @metaharness/devops | 0.1.1 | no-registry-attestation |  |
| @metaharness/education | 0.1.0 | no-registry-attestation |  |
| @metaharness/example-ads | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-aws | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-azure | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-bio | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-datadog | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-fhir | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-gcp | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-github | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-huggingface | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-iot | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-nasa | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-pinecone | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-qiskit | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-slack | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-stripe | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-supabase | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-twilio | 0.2.0 | no-registry-attestation |  |
| @metaharness/example-web3 | 0.2.0 | no-registry-attestation |  |
| @metaharness/field-memory | 0.1.0 | no-registry-attestation |  |
| @metaharness/flywheel | 0.1.12 | missing-repository, no-registry-attestation |  |
| @metaharness/gaming | 0.1.0 | no-registry-attestation |  |
| @metaharness/github-actions | 0.1.0 | no-registry-attestation |  |
| @metaharness/harness | 0.2.0 | no-registry-attestation |  |
| @metaharness/hermes | 0.1.1 | no-registry-attestation |  |
| @metaharness/horizon | 0.2.1 | missing-repository, no-registry-attestation |  |
| @metaharness/host-claude-code | 0.1.3 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-codex | 0.1.3 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-copilot | 0.1.3 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-github-actions | 0.1.4 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-hermes | 0.1.4 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-openclaw | 0.1.4 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-opencode | 0.1.3 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-pi-dev | 0.1.3 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-prime-agent | 0.1.0 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/host-rvm | 0.1.4 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/jujutsu | 0.2.0 | no-registry-attestation | agenticow (peerDependencies), agentic-jujutsu (peerDependencies) |
| @metaharness/kernel | 0.1.3 | no-registry-attestation | @ruvector/emergent-time (dependencies), @ruvector/rvf (peerDependencies) |
| @metaharness/legal | 0.1.0 | no-registry-attestation |  |
| @metaharness/oo-agents | 0.1.0 | missing-repository, no-registry-attestation, dependencies:unbounded:@metaharness/radio | @metaharness/radio (dependencies) |
| @metaharness/openclaw | 0.1.0 | no-registry-attestation |  |
| @metaharness/opencode | 0.1.0 | no-registry-attestation |  |
| @metaharness/pi-dev | 0.1.0 | no-registry-attestation |  |
| @metaharness/projects | 0.1.2 | no-registry-attestation |  |
| @metaharness/radio | 0.1.0 | missing-repository, no-registry-attestation |  |
| @metaharness/redblue | 0.1.6 | no-registry-attestation |  |
| @metaharness/repo-maintainer | 0.1.0 | no-registry-attestation |  |
| @metaharness/research | 0.1.1 | no-registry-attestation |  |
| @metaharness/router | 0.4.0 | no-registry-attestation | @ruvector/tiny-dancer (peerDependencies) |
| @metaharness/rvm | 0.1.0 | no-registry-attestation |  |
| @metaharness/sales | 0.1.0 | no-registry-attestation |  |
| @metaharness/sdk | 0.1.0 | no-registry-attestation | @metaharness/kernel (dependencies) |
| @metaharness/support | 0.1.0 | no-registry-attestation |  |
| @metaharness/trading | 0.1.0 | no-registry-attestation |  |
| @metaharness/turn-credit | 0.1.0 | no-registry-attestation |  |
| @metaharness/vertical-base | 0.1.0 | no-registry-attestation |  |
| @metaharness/vertical-trading | 0.1.0 | no-registry-attestation | @metaharness/vertical-base (dependencies) |
| @metaharness/weight-eft | 0.1.2 | no-registry-attestation |  |
| @metaharness/workspace-lens | 0.1.2 | missing-repository, no-registry-attestation |  |
| @metaharness/workspace-probe | 0.1.1 | missing-repository, no-registry-attestation | @metaharness/workspace-lens (dependencies) |
| @neural-trader/agentic-accounting-agents | 0.1.1 | no-registry-attestation, dependencies:non-registry:@neural-trader/agentic-accounting-core, dependencies:non-registry:@neural-trader/agentic-accounting-types | @neural-trader/agentic-accounting-core (dependencies), @neural-trader/agentic-accounting-types (dependencies), agentic-flow (dependencies), agentdb (dependencies) |
| @neural-trader/agentic-accounting-cli | 0.1.1 | no-registry-attestation, dependencies:non-registry:@neural-trader/agentic-accounting-core, dependencies:non-registry:@neural-trader/agentic-accounting-types | @neural-trader/agentic-accounting-core (dependencies), @neural-trader/agentic-accounting-types (dependencies) |
| @neural-trader/agentic-accounting-core | 0.1.1 | no-registry-attestation, dependencies:non-registry:@neural-trader/agentic-accounting-types, dependencies:non-registry:@neural-trader/agentic-accounting-rust-core | @neural-trader/agentic-accounting-types (dependencies), @neural-trader/agentic-accounting-rust-core (dependencies), agentdb (dependencies) |
| @neural-trader/agentic-accounting-mcp | 0.1.1 | no-registry-attestation, dependencies:non-registry:@neural-trader/agentic-accounting-core, dependencies:non-registry:@neural-trader/agentic-accounting-agents, dependencies:non-registry:@neural-trader/agentic-accounting-types | @neural-trader/agentic-accounting-core (dependencies), @neural-trader/agentic-accounting-agents (dependencies), @neural-trader/agentic-accounting-types (dependencies) |
| @neural-trader/agentic-accounting-rust-core | 0.1.1 | no-registry-attestation |  |
| @neural-trader/agentic-accounting-types | 0.1.1 | no-registry-attestation |  |
| @neural-trader/backend | 2.2.1 | no-registry-attestation, install-script-review |  |
| @neural-trader/backtesting | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/benchoptimizer | 2.1.1 | no-registry-attestation |  |
| @neural-trader/brokers | 2.1.1 | no-registry-attestation | @neural-trader/core (peerDependencies) |
| @neural-trader/core | 2.0.0 | no-registry-attestation, install-script-review |  |
| @neural-trader/e2b-strategies | 1.1.1 | no-registry-attestation | agentic-jujutsu (dependencies), @neural-trader/brokers (optionalDependencies), @neural-trader/execution (optionalDependencies), @neural-trader/features (optionalDependencies), @neural-trader/market-data (optionalDependencies), @neural-trader/neural (optionalDependencies), @neural-trader/portfolio (optionalDependencies), @neural-trader/risk (optionalDependencies), @neural-trader/strategies (optionalDependencies) |
| @neural-trader/example-dynamic-pricing | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/example-energy-forecasting | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), claude-flow (dependencies) |
| @neural-trader/example-energy-grid-optimization | 0.1.0 | no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), sublinear-time-solver (dependencies) |
| @neural-trader/example-evolutionary-game-theory | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor, dependencies:non-registry:@neural-trader/core | @neural-trader/predictor (dependencies), @neural-trader/core (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/example-healthcare-optimization | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/example-logistics-optimization | 1.0.0 | missing-repository, no-registry-attestation | agentic-flow (dependencies), agentdb (dependencies), sublinear-time-solver (dependencies) |
| @neural-trader/example-market-microstructure | 1.0.0 | no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), sublinear-time-solver (dependencies), claude-flow (dependencies) |
| @neural-trader/example-neuromorphic-computing | 1.0.0 | missing-repository, no-registry-attestation, dependencies:unbounded:@neural-trader/agentdb, dependencies:unbounded:@neural-trader/agentic-flow |  |
| @neural-trader/example-portfolio-optimization | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor, dependencies:non-registry:@neural-trader/core | @neural-trader/predictor (dependencies), @neural-trader/core (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/example-quantum-optimization | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/core, dependencies:non-registry:@neural-trader/predictor | @neural-trader/core (dependencies), @neural-trader/predictor (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/example-supply-chain-prediction | 1.0.0 | missing-repository, no-registry-attestation, dependencies:non-registry:@neural-trader/predictor | @neural-trader/predictor (dependencies), agentdb (dependencies), agentic-flow (dependencies) |
| @neural-trader/execution | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/features | 2.1.2 | no-registry-attestation | @neural-trader/core (peerDependencies) |
| @neural-trader/market-data | 2.1.1 | no-registry-attestation | @neural-trader/core (peerDependencies) |
| @neural-trader/mcp | 2.1.0 | no-registry-attestation | @neural-trader/core (dependencies), @neural-trader/mcp-protocol (dependencies) |
| @neural-trader/mcp-protocol | 2.0.0 | no-registry-attestation | @neural-trader/core (dependencies) |
| @neural-trader/neural | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/neuro-divergent | 2.1.1 | no-registry-attestation, install-script-review |  |
| @neural-trader/news-trading | 2.1.1 | no-registry-attestation | @neural-trader/core (peerDependencies) |
| @neural-trader/portfolio | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/prediction-markets | 2.1.1 | no-registry-attestation | @neural-trader/core (peerDependencies) |
| @neural-trader/predictor | 0.1.0 | no-registry-attestation | @neural-trader/neural (peerDependencies) |
| @neural-trader/risk | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/sports-betting | 2.1.1 | no-registry-attestation | @neural-trader/core (peerDependencies), @neural-trader/risk (peerDependencies) |
| @neural-trader/strategies | 2.6.0 | no-registry-attestation | neural-trader (dependencies), neural-trader (peerDependencies) |
| @neural-trader/syndicate | 2.1.1 | no-registry-attestation |  |
| @qudag/cli | 0.1.0 | missing-repository, no-registry-attestation | @qudag/napi-core (peerDependencies) |
| @qudag/mcp-sse | 0.1.0 | missing-repository, no-registry-attestation | @qudag/napi-core (peerDependencies) |
| @qudag/mcp-stdio | 0.1.0 | no-registry-attestation | @qudag/napi-core (peerDependencies) |
| @qudag/napi-core | 0.1.0 | no-registry-attestation |  |
| @ruv/rvcsi | 0.3.1 | no-registry-attestation |  |
| @ruv/sparc-ui | 0.1.4 | missing-repository, missing-license, no-registry-attestation |  |
| @ruvector/acorn-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/agentic-integration | 1.0.2 | no-registry-attestation |  |
| @ruvector/agentic-synth | 0.1.7 | no-registry-attestation | agentic-robotics (peerDependencies), dspy.ts (peerDependencies), midstreamer (peerDependencies), ruvector (peerDependencies) |
| @ruvector/agentic-synth-examples | 0.1.6 | no-registry-attestation | @ruvector/agentic-synth (dependencies), dspy.ts (dependencies), @ruvector/agentic-synth (peerDependencies) |
| @ruvector/attention | 2.2.2 | no-registry-attestation | @ruvector/attention-linux-x64-gnu (optionalDependencies), @ruvector/attention-linux-arm64-gnu (optionalDependencies), @ruvector/attention-darwin-x64 (optionalDependencies), @ruvector/attention-darwin-arm64 (optionalDependencies), @ruvector/attention-win32-x64-msvc (optionalDependencies) |
| @ruvector/attention-darwin-arm64 | 2.2.2 | no-registry-attestation |  |
| @ruvector/attention-darwin-x64 | 2.2.2 | no-registry-attestation |  |
| @ruvector/attention-linux-arm64-gnu | 2.2.2 | no-registry-attestation |  |
| @ruvector/attention-linux-x64-gnu | 2.2.2 | no-registry-attestation |  |
| @ruvector/attention-unified-wasm | 0.1.29 | no-registry-attestation |  |
| @ruvector/attention-wasm | 2.2.2 | no-registry-attestation |  |
| @ruvector/attention-win32-x64-msvc | 2.2.2 | no-registry-attestation |  |
| @ruvector/burst-scaling | 1.0.2 | missing-repository, no-registry-attestation |  |
| @ruvector/cli | 0.1.29 | missing-repository, no-registry-attestation |  |
| @ruvector/cluster | 0.1.0 | no-registry-attestation |  |
| @ruvector/cnn | 0.1.1 | no-registry-attestation |  |
| @ruvector/core | 0.1.32 |  | ruvector-core-linux-x64-gnu (optionalDependencies), ruvector-core-linux-arm64-gnu (optionalDependencies), ruvector-core-darwin-x64 (optionalDependencies), ruvector-core-darwin-arm64 (optionalDependencies), ruvector-core-win32-x64-msvc (optionalDependencies) |
| @ruvector/diskann | 0.1.1 | no-registry-attestation | @ruvector/diskann-linux-x64-gnu (optionalDependencies), @ruvector/diskann-linux-arm64-gnu (optionalDependencies), @ruvector/diskann-darwin-x64 (optionalDependencies), @ruvector/diskann-darwin-arm64 (optionalDependencies), @ruvector/diskann-win32-x64-msvc (optionalDependencies) |
| @ruvector/diskann-darwin-arm64 | 0.1.0 | no-registry-attestation |  |
| @ruvector/diskann-darwin-x64 | 0.1.0 | no-registry-attestation |  |
| @ruvector/diskann-linux-arm64-gnu | 0.1.0 | no-registry-attestation |  |
| @ruvector/diskann-linux-x64-gnu | 0.1.0 | no-registry-attestation |  |
| @ruvector/diskann-win32-x64-msvc | 0.1.0 | no-registry-attestation |  |
| @ruvector/economy-wasm | 0.1.29 | no-registry-attestation |  |
| @ruvector/edge | 0.1.9 | no-registry-attestation | @ruvector/edge-full (optionalDependencies), @ruvector/edge-full (peerDependencies) |
| @ruvector/edge-full | 0.1.0 | no-registry-attestation |  |
| @ruvector/edge-net | 0.5.3 | no-registry-attestation | @ruvector/ruvllm (dependencies) |
| @ruvector/emergent-time | 0.1.0 | no-registry-attestation |  |
| @ruvector/exotic-wasm | 0.1.29 | no-registry-attestation |  |
| @ruvector/gnn | 0.1.25 | no-registry-attestation | @ruvector/gnn-linux-x64-gnu (optionalDependencies), @ruvector/gnn-linux-x64-musl (optionalDependencies), @ruvector/gnn-linux-arm64-gnu (optionalDependencies), @ruvector/gnn-linux-arm64-musl (optionalDependencies), @ruvector/gnn-darwin-x64 (optionalDependencies), @ruvector/gnn-darwin-arm64 (optionalDependencies), @ruvector/gnn-win32-x64-msvc (optionalDependencies) |
| @ruvector/gnn-darwin-arm64 | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-darwin-x64 | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-linux-arm64-gnu | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-linux-arm64-musl | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-linux-x64-gnu | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-linux-x64-musl | 0.1.25 | no-registry-attestation |  |
| @ruvector/gnn-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/gnn-win32-x64-msvc | 0.1.25 | no-registry-attestation |  |
| @ruvector/graph-data-generator | 0.1.0 | no-registry-attestation | @ruvector/agentic-synth (dependencies), @ruvector/agentic-synth (peerDependencies) |
| @ruvector/graph-node | 2.1.1 | no-registry-attestation | @ruvector/graph-node-darwin-x64 (optionalDependencies), @ruvector/graph-node-darwin-arm64 (optionalDependencies), @ruvector/graph-node-linux-x64-gnu (optionalDependencies), @ruvector/graph-node-win32-x64-msvc (optionalDependencies), @ruvector/graph-node-linux-arm64-gnu (optionalDependencies) |
| @ruvector/graph-node-darwin-arm64 | 2.1.1 | no-registry-attestation |  |
| @ruvector/graph-node-darwin-x64 | 2.1.1 | no-registry-attestation |  |
| @ruvector/graph-node-linux-arm64-gnu | 2.1.1 | no-registry-attestation |  |
| @ruvector/graph-node-linux-x64-gnu | 2.1.1 | no-registry-attestation |  |
| @ruvector/graph-node-win32-x64-msvc | 2.1.1 | no-registry-attestation |  |
| @ruvector/graph-transformer | 2.0.4 | no-registry-attestation | @ruvector/graph-transformer-linux-x64-gnu (optionalDependencies), @ruvector/graph-transformer-linux-x64-musl (optionalDependencies), @ruvector/graph-transformer-linux-arm64-gnu (optionalDependencies), @ruvector/graph-transformer-linux-arm64-musl (optionalDependencies), @ruvector/graph-transformer-darwin-x64 (optionalDependencies), @ruvector/graph-transformer-darwin-arm64 (optionalDependencies), @ruvector/graph-transformer-win32-x64-msvc (optionalDependencies) |
| @ruvector/graph-transformer-darwin-arm64 | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-darwin-x64 | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-linux-arm64-gnu | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-linux-arm64-musl | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-linux-x64-gnu | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-linux-x64-musl | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-transformer-win32-x64-msvc | 2.0.4 | no-registry-attestation |  |
| @ruvector/graph-wasm | 2.0.3 | no-registry-attestation |  |
| @ruvector/kge | 0.2.0 | no-registry-attestation |  |
| @ruvector/kge-darwin-arm64 | 0.2.0 | no-registry-attestation |  |
| @ruvector/kge-darwin-x64 | 0.2.0 | no-registry-attestation |  |
| @ruvector/kge-linux-arm64-gnu | 0.2.0 | no-registry-attestation |  |
| @ruvector/kge-linux-x64-gnu | 0.2.0 | no-registry-attestation |  |
| @ruvector/kge-win32-x64-msvc | 0.2.0 | no-registry-attestation |  |
| @ruvector/learning-wasm | 0.1.29 | no-registry-attestation |  |
| @ruvector/math-wasm | 2.2.2 | no-registry-attestation |  |
| @ruvector/mincut-wasm | 2.3.0 | no-registry-attestation |  |
| @ruvector/nervous-system-wasm | 0.1.29 | no-registry-attestation |  |
| @ruvector/neural-trader-wasm | 0.1.1 | missing-repository, no-registry-attestation |  |
| @ruvector/node | 0.1.24 | no-registry-attestation | @ruvector/core (dependencies), @ruvector/gnn (dependencies) |
| @ruvector/node-darwin-arm64 | 0.1.18 | no-registry-attestation |  |
| @ruvector/node-darwin-x64 | 0.1.18 | no-registry-attestation |  |
| @ruvector/node-linux-arm64-gnu | 0.1.18 | no-registry-attestation |  |
| @ruvector/node-linux-x64-gnu | 0.1.18 | no-registry-attestation |  |
| @ruvector/node-win32-x64-msvc | 0.1.18 | no-registry-attestation |  |
| @ruvector/ospipe | 0.1.3 | no-registry-attestation |  |
| @ruvector/ospipe-wasm | 0.1.1 | no-registry-attestation |  |
| @ruvector/pi-brain | 0.1.2 | no-registry-attestation |  |
| @ruvector/postgres-cli | 0.2.9 | no-registry-attestation |  |
| @ruvector/rabitq-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/raft | 0.1.1 | no-registry-attestation |  |
| @ruvector/replication | 0.1.1 | no-registry-attestation |  |
| @ruvector/router | 0.1.32 | no-registry-attestation | @ruvector/router-darwin-x64 (optionalDependencies), @ruvector/router-darwin-arm64 (optionalDependencies), @ruvector/router-linux-x64-gnu (optionalDependencies), @ruvector/router-win32-x64-msvc (optionalDependencies), @ruvector/router-linux-arm64-gnu (optionalDependencies) |
| @ruvector/router-darwin-arm64 | 0.1.32 | no-registry-attestation |  |
| @ruvector/router-darwin-x64 | 0.1.32 | no-registry-attestation |  |
| @ruvector/router-linux-arm64-gnu | 0.1.32 | no-registry-attestation |  |
| @ruvector/router-linux-x64-gnu | 0.1.32 | no-registry-attestation |  |
| @ruvector/router-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/router-win32-x64-msvc | 0.1.32 | no-registry-attestation |  |
| @ruvector/rudag | 0.1.1 | no-registry-attestation |  |
| @ruvector/ruqu | 0.2.0 | no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies) |
| @ruvector/ruqu-wasm | 2.0.6 | no-registry-attestation |  |
| @ruvector/ruvector-consciousness-wasm | 2.1.0 | no-registry-attestation |  |
| @ruvector/ruvllm | 3.0.0 | no-registry-attestation | @ruvector/ruvllm-darwin-x64 (optionalDependencies), @ruvector/ruvllm-darwin-arm64 (optionalDependencies), @ruvector/ruvllm-linux-x64-gnu (optionalDependencies), @ruvector/ruvllm-win32-x64-msvc (optionalDependencies), @ruvector/ruvllm-linux-arm64-gnu (optionalDependencies) |
| @ruvector/ruvllm-cli | 0.1.1 | no-registry-attestation |  |
| @ruvector/ruvllm-darwin-arm64 | 2.0.1 | no-registry-attestation |  |
| @ruvector/ruvllm-darwin-x64 | 2.0.1 | no-registry-attestation |  |
| @ruvector/ruvllm-linux-arm64-gnu | 2.0.1 | no-registry-attestation |  |
| @ruvector/ruvllm-linux-x64-gnu | 2.0.1 | no-registry-attestation |  |
| @ruvector/ruvllm-wasm | 2.0.2 | no-registry-attestation |  |
| @ruvector/ruvllm-win32-x64-msvc | 2.0.1 | no-registry-attestation |  |
| @ruvector/rvagent-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/rvdna | 0.3.1 | no-registry-attestation |  |
| @ruvector/rvf | 0.3.4 | no-registry-attestation | @ruvector/rvf-node (dependencies), @ruvector/rvf-wasm (optionalDependencies), @ruvector/rvf-solver (optionalDependencies) |
| @ruvector/rvf-mcp-server | 0.1.4 | missing-repository, no-registry-attestation | @ruvector/rvf (dependencies) |
| @ruvector/rvf-node | 0.2.3 | missing-repository, no-registry-attestation | @ruvector/rvf-node-darwin-x64 (optionalDependencies), @ruvector/rvf-node-darwin-arm64 (optionalDependencies), @ruvector/rvf-node-linux-x64-gnu (optionalDependencies), @ruvector/rvf-node-win32-x64-msvc (optionalDependencies), @ruvector/rvf-node-linux-arm64-gnu (optionalDependencies) |
| @ruvector/rvf-node-darwin-arm64 | 0.2.2 | missing-repository, no-registry-attestation |  |
| @ruvector/rvf-node-darwin-x64 | 0.2.2 | missing-repository, no-registry-attestation |  |
| @ruvector/rvf-node-linux-arm64-gnu | 0.2.2 | missing-repository, no-registry-attestation |  |
| @ruvector/rvf-node-linux-x64-gnu | 0.2.2 | missing-repository, no-registry-attestation |  |
| @ruvector/rvf-node-win32-x64-msvc | 0.2.2 | missing-repository, no-registry-attestation |  |
| @ruvector/rvf-solver | 0.1.8 | no-registry-attestation |  |
| @ruvector/rvf-wasm | 0.1.9 | no-registry-attestation |  |
| @ruvector/rvforge | 0.2.0 | no-registry-attestation |  |
| @ruvector/rvlite | 0.2.4 | no-registry-attestation |  |
| @ruvector/scipix | 0.1.1 | no-registry-attestation |  |
| @ruvector/server | 0.1.0 | no-registry-attestation |  |
| @ruvector/sona | 0.1.8 | no-registry-attestation | @ruvector/sona-linux-x64-gnu (optionalDependencies), @ruvector/sona-linux-x64-musl (optionalDependencies), @ruvector/sona-linux-arm64-gnu (optionalDependencies), @ruvector/sona-darwin-x64 (optionalDependencies), @ruvector/sona-darwin-arm64 (optionalDependencies), @ruvector/sona-win32-x64-msvc (optionalDependencies), @ruvector/sona-win32-arm64-msvc (optionalDependencies) |
| @ruvector/sona-darwin-arm64 | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-darwin-x64 | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-linux-arm64-gnu | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-linux-x64-gnu | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-linux-x64-musl | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-win32-arm64-msvc | 0.1.8 | no-registry-attestation |  |
| @ruvector/sona-win32-x64-msvc | 0.1.8 | no-registry-attestation |  |
| @ruvector/spiking-neural | 1.0.3 | no-registry-attestation, install-script-review |  |
| @ruvector/tiny-dancer | 0.1.22 | no-registry-attestation | @ruvector/tiny-dancer-linux-x64-gnu (optionalDependencies), @ruvector/tiny-dancer-linux-x64-musl (optionalDependencies), @ruvector/tiny-dancer-linux-arm64-gnu (optionalDependencies), @ruvector/tiny-dancer-linux-arm64-musl (optionalDependencies), @ruvector/tiny-dancer-darwin-x64 (optionalDependencies), @ruvector/tiny-dancer-darwin-arm64 (optionalDependencies), @ruvector/tiny-dancer-win32-x64-msvc (optionalDependencies), @ruvector/tiny-dancer-win32-arm64-msvc (optionalDependencies) |
| @ruvector/tiny-dancer-darwin-arm64 | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-darwin-x64 | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-linux-arm64-gnu | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-linux-arm64-musl | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-linux-x64-gnu | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-linux-x64-musl | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-wasm | 0.1.0 | no-registry-attestation |  |
| @ruvector/tiny-dancer-win32-arm64-msvc | 0.1.22 | no-registry-attestation |  |
| @ruvector/tiny-dancer-win32-x64-msvc | 0.1.22 | no-registry-attestation |  |
| @ruvector/typesafe | 0.2.0 | no-registry-attestation |  |
| @ruvector/typesafe-darwin-arm64 | 0.2.0 | no-registry-attestation |  |
| @ruvector/typesafe-darwin-x64 | 0.2.0 | no-registry-attestation |  |
| @ruvector/typesafe-linux-arm64-gnu | 0.2.0 | no-registry-attestation |  |
| @ruvector/typesafe-linux-x64-gnu | 0.2.0 | no-registry-attestation |  |
| @ruvector/typesafe-win32-x64-msvc | 0.2.0 | no-registry-attestation |  |
| @ruvector/wasm | 0.2.1 | no-registry-attestation |  |
| @ruvector/wasm-unified | 1.0.0 | no-registry-attestation |  |
| @ruvnet/agent-harness-generator | 0.1.3 | no-registry-attestation | metaharness (dependencies) |
| @ruvnet/bmssp | 1.0.0 | no-registry-attestation |  |
| @ruvnet/open-claude-code | 2.1.0 | no-registry-attestation |  |
| @ruvnet/ruvector-verified-wasm | 0.1.1 | no-registry-attestation |  |
| @ruvnet/ruview | 0.9.1 |  | @ruvnet/ruview-kernel (peerDependencies) |
| @ruvnet/ruview-kernel | 0.1.0 | no-registry-attestation |  |
| @ruvnet/ruvn | 0.1.1 | missing-repository, missing-license, no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies), @metaharness/host-codex (dependencies), @metaharness/host-pi-dev (dependencies), @metaharness/host-hermes (dependencies), @metaharness/host-openclaw (dependencies), @metaharness/host-rvm (dependencies), @metaharness/host-copilot (dependencies), @metaharness/host-opencode (dependencies), @metaharness/host-github-actions (dependencies) |
| @ruvnet/rvagent | 0.2.0 | no-registry-attestation |  |
| @ruvnet/strange-loop | 0.3.1 | no-registry-attestation |  |
| agent-booster | 0.2.2 | no-registry-attestation |  |
| agentbbs | 0.2.1 | no-registry-attestation |  |
| agentdb | 3.0.0-alpha.20 | no-registry-attestation, install-script-review | @ruvector/graph-transformer (dependencies), ruvector (optionalDependencies), @ruvector/gnn (optionalDependencies), @ruvector/rvf (optionalDependencies), @ruvector/sona (optionalDependencies), @ruvector/router (optionalDependencies), @ruvector/ruvllm (optionalDependencies), @ruvector/rvf-node (optionalDependencies), @ruvector/rvf-wasm (optionalDependencies), @ruvector/attention (optionalDependencies), @ruvector/graph-node (optionalDependencies), @ruvector/rvf-solver (optionalDependencies), ruvector-attention-wasm (optionalDependencies), ruvector-graph-transformer-wasm (optionalDependencies) |
| agentic-flow | 2.1.4 | no-registry-attestation, install-script-review | agentdb (dependencies), ruvector (dependencies), @ruvector/core (dependencies), @ruvector/router (dependencies), @ruvector/ruvllm (dependencies), @metaharness/router (dependencies), @ruvector/edge-full (dependencies), @ruvector/tiny-dancer (dependencies), ruvector-onnx-embeddings-wasm (dependencies), @ruvector/sona (optionalDependencies), @ruvector/attention (optionalDependencies) |
| agentic-jujutsu | 2.3.6 | no-registry-attestation | @qudag/napi-core (dependencies) |
| agentic-payments | 0.1.13 | no-registry-attestation | agentic-flow (optionalDependencies), agentic-flow (peerDependencies) |
| agentic-robotics | 0.2.4 | no-registry-attestation | @agentic-robotics/core (dependencies), @agentic-robotics/cli (dependencies), @agentic-robotics/mcp (dependencies) |
| agenticow | 0.2.4 | no-registry-attestation | @ruvector/rvf-node (dependencies) |
| agentics-hackathon | 1.3.4 | no-registry-attestation |  |
| agenticsjs | 1.0.5 | no-registry-attestation |  |
| ai-text-watermark | 0.4.0 | no-registry-attestation | @claude-flow/watermark (dependencies) |
| aidefence | 2.3.0 | no-registry-attestation |  |
| aidefense | 2.3.0 | no-registry-attestation |  |
| bmssp-wasm | 1.0.0 | no-registry-attestation |  |
| claude-flow | 3.56.5 | no-registry-attestation | @claude-flow/mcp (dependencies), @claude-flow/codex (dependencies), @claude-flow/neural (dependencies), @claude-flow/shared (dependencies), @claude-flow/cli-core (dependencies), @claude-flow/security (dependencies), @ruvector/rabitq-wasm (dependencies), @claude-flow/plugin-agent-federation (dependencies), agentdb (optionalDependencies), agentic-flow (optionalDependencies), @ruvector/core (optionalDependencies), @ruvector/sona (optionalDependencies), @ruvector/router (optionalDependencies), @ruvector/attention (optionalDependencies), @ruvector/router-linux-x64-gnu (optionalDependencies), @claude-flow/plugin-gastown-bridge (optionalDependencies) |
| claude-flow-v3 | 3.0.0-alpha.1 | no-registry-attestation | agentdb (dependencies) |
| cogmusic | 0.1.1 | missing-repository, no-registry-attestation |  |
| consciousness-explorer | 1.1.2 | no-registry-attestation |  |
| create-sparc | 1.2.4 | missing-repository, no-registry-attestation |  |
| cuda-wasm | 1.1.1 | no-registry-attestation, install-script-review |  |
| dream-machine | 0.1.2 |  |  |
| dspy.ts | 2.2.0 | no-registry-attestation | agentdb (dependencies) |
| flow-nexus | 0.1.128 | no-registry-attestation |  |
| goalie | 1.3.1 | no-registry-attestation |  |
| homecore | 0.1.0 | no-registry-attestation | @metaharness/kernel (dependencies) |
| lean-agentic | 0.3.2 | no-registry-attestation | agentdb (dependencies) |
| llm-stream-reformat | 0.1.0 | no-registry-attestation |  |
| metaharness | 0.4.17 | no-registry-attestation | @metaharness/avo (dependencies), @metaharness/darwin (dependencies), @metaharness/redblue (dependencies), @metaharness/flywheel (dependencies), @metaharness/weight-eft (dependencies), @metaharness/turn-credit (dependencies), @ruvector/ruvllm (optionalDependencies), @metaharness/kernel (peerDependencies) |
| midstreamer | 0.3.1 | no-registry-attestation | agentic-flow (dependencies) |
| neural-trader | 2.8.11 | no-registry-attestation, install-script-review | agentic-payments (dependencies), aidefence (dependencies), midstreamer (dependencies), sublinear-time-solver (dependencies), @neural-trader/core (optionalDependencies), @neural-trader/predictor (optionalDependencies), agentdb (optionalDependencies), agentic-flow (optionalDependencies) |
| prime-radiant-advanced-wasm | 0.1.3 | no-registry-attestation |  |
| psycho-symbolic-integration | 0.2.0 | no-registry-attestation | psycho-symbolic-reasoner (dependencies), @ruvector/agentic-synth (dependencies), ruvector (peerDependencies) |
| psycho-symbolic-reasoner | 1.0.7 | no-registry-attestation |  |
| qudag | 1.2.1 | no-registry-attestation, install-script-review |  |
| radio-moe | 0.3.1 |  | @metaharness/radio (dependencies) |
| research-swarm | 1.2.2 | no-registry-attestation | goalie (dependencies), agentic-flow (peerDependencies) |
| rucelium-harness | 0.1.0 | no-registry-attestation |  |
| ruflo | 3.56.5 | no-registry-attestation | @claude-flow/cli (dependencies) |
| ruflo-graph-intelligence | 0.1.0-alpha.1 | no-registry-attestation | sublinear-time-solver (dependencies), @claude-flow/cli (peerDependencies) |
| rulake-wasm | 2.3.0-alpha.1 | no-registry-attestation |  |
| rupixel | 0.1.0 | no-registry-attestation |  |
| ruv-swarm | 1.0.20 | no-registry-attestation, optionalDependencies:unbounded:worker_threads |  |
| ruvbot | 0.3.3 | no-registry-attestation | aidefence (dependencies), @ruvector/ruvllm (optionalDependencies) |
| ruvdrone | 0.1.0 | no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies), @metaharness/host-codex (dependencies), @metaharness/host-copilot (dependencies), @metaharness/host-github-actions (dependencies), @metaharness/host-hermes (dependencies), @metaharness/host-openclaw (dependencies), @metaharness/host-opencode (dependencies), @metaharness/host-pi-dev (dependencies), @metaharness/host-rvm (dependencies) |
| ruvector | 0.3.3 | no-registry-attestation | @ruvector/gnn (dependencies), @ruvector/core (dependencies), @ruvector/sona (dependencies), @ruvector/attention (dependencies), metaharness (optionalDependencies), @ruvector/rvf (optionalDependencies), @metaharness/darwin (optionalDependencies), @metaharness/router (optionalDependencies), @metaharness/harness (optionalDependencies), @metaharness/redblue (optionalDependencies), @metaharness/flywheel (optionalDependencies), @ruvector/mincut-wasm (optionalDependencies), @ruvector/tiny-dancer (optionalDependencies), @metaharness/weight-eft (optionalDependencies), @metaharness/workspace-lens (optionalDependencies), @metaharness/workspace-probe (optionalDependencies), metaharness (peerDependencies), @ruvector/router (peerDependencies), @ruvector/ruvllm (peerDependencies), @ruvector/diskann (peerDependencies), @ruvector/pi-brain (peerDependencies), @metaharness/darwin (peerDependencies), @metaharness/router (peerDependencies), @metaharness/harness (peerDependencies), @metaharness/redblue (peerDependencies), @metaharness/flywheel (peerDependencies), @metaharness/weight-eft (peerDependencies), @metaharness/workspace-lens (peerDependencies), @metaharness/workspace-probe (peerDependencies) |
| ruvector-attention-unified-wasm | 0.1.0 | no-registry-attestation |  |
| ruvector-attention-wasm | 2.1.0 | no-registry-attestation |  |
| ruvector-core | 0.1.26 | no-registry-attestation | ruvector-core-linux-x64-gnu (optionalDependencies), ruvector-core-linux-arm64-gnu (optionalDependencies), ruvector-core-darwin-x64 (optionalDependencies), ruvector-core-darwin-arm64 (optionalDependencies), ruvector-core-win32-x64-msvc (optionalDependencies) |
| ruvector-core-darwin-arm64 | 0.1.30 |  |  |
| ruvector-core-darwin-x64 | 0.1.30 |  |  |
| ruvector-core-linux-arm64-gnu | 0.1.30 |  |  |
| ruvector-core-linux-x64-gnu | 0.1.30 |  |  |
| ruvector-core-win32-x64-msvc | 0.1.30 |  |  |
| ruvector-extensions | 0.1.2 | no-registry-attestation | ruvector (dependencies) |
| ruvector-gnn-wasm | 2.1.0 | no-registry-attestation |  |
| ruvector-graph-transformer-wasm | 2.0.4 | no-registry-attestation |  |
| ruvector-math-wasm | 0.1.32 | no-registry-attestation |  |
| ruvector-mragent | 0.1.0 | no-registry-attestation | @metaharness/darwin (optionalDependencies), ruvector (optionalDependencies), @metaharness/darwin (peerDependencies) |
| ruvector-onnx-embeddings-wasm | 0.1.2 | no-registry-attestation |  |
| ruvector-sona | 0.1.4 | no-registry-attestation |  |
| ruvector-wasm | 2.1.0 | no-registry-attestation |  |
| ruvi | 1.1.0 | missing-repository, no-registry-attestation |  |
| ruvllm-esp32 | 0.3.3 | no-registry-attestation, install-script-review |  |
| ruvneural | 0.1.0 | missing-repository, missing-license, no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies), @metaharness/host-codex (dependencies), @metaharness/host-pi-dev (dependencies), @metaharness/host-hermes (dependencies), @metaharness/host-openclaw (dependencies), @metaharness/host-rvm (dependencies), @metaharness/host-copilot (dependencies), @metaharness/host-opencode (dependencies), @metaharness/host-github-actions (dependencies) |
| rvlite | 0.2.6 | no-registry-attestation | @ruvector/rvf-wasm (optionalDependencies), @ruvector/rvf-wasm (peerDependencies) |
| spiking-neural | 1.0.1 | no-registry-attestation, install-script-review |  |
| strange-loops | 1.0.3 | no-registry-attestation | sublinear-time-solver (dependencies) |
| strange-loops-mcp | 1.0.0 | no-registry-attestation | strange-loops (dependencies) |
| sublinear-time-solver | 1.7.2 | no-registry-attestation | @ruvnet/strange-loop (dependencies), strange-loops (dependencies), @ruvnet/bmssp (dependencies), temporal-lead-solver (dependencies) |
| symbolic-scribe-harness | 0.1.1 | no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies) |
| temporal-lead-solver | 0.1.0 | no-registry-attestation |  |
| temporal-neural-solver | 0.1.3 | no-registry-attestation |  |
| vscode-remote-mcp | 1.0.4 | missing-repository, no-registry-attestation |  |
| wifi-densepose-sar-harness | 0.1.0 | missing-repository, no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies), @metaharness/router (dependencies), @metaharness/flywheel (dependencies) |
| worldgraphs | 0.1.3 | missing-repository, missing-license, no-registry-attestation | @metaharness/kernel (dependencies), @metaharness/host-claude-code (dependencies), @metaharness/host-codex (dependencies), @metaharness/host-pi-dev (dependencies), @metaharness/host-hermes (dependencies), @metaharness/host-openclaw (dependencies), @metaharness/host-rvm (dependencies), @metaharness/host-copilot (dependencies), @metaharness/host-opencode (dependencies), @metaharness/host-github-actions (dependencies) |

## Prioritized next work

- agentdb: inspect-package-and-repair-metadata-or-release-proof; priority 32.
- agentic-flow: inspect-package-and-repair-metadata-or-release-proof; priority 31.
- @metaharness/kernel: inspect-package-and-repair-metadata-or-release-proof; priority 22.
- @claude-flow/memory: inspect-package-and-repair-metadata-or-release-proof; priority 16.
- @neural-trader/core: inspect-package-and-repair-metadata-or-release-proof; priority 16.
- neural-trader: inspect-package-and-repair-metadata-or-release-proof; priority 16.
- @claude-flow/shared: inspect-package-and-repair-metadata-or-release-proof; priority 15.
- @claude-flow/security: inspect-package-and-repair-metadata-or-release-proof; priority 13.
- @neural-trader/predictor: inspect-package-and-repair-metadata-or-release-proof; priority 12.
- @claude-flow/performance: inspect-package-and-repair-metadata-or-release-proof; priority 10.
- ruvector: inspect-package-and-repair-metadata-or-release-proof; priority 10.
- @metaharness/flywheel: inspect-package-and-repair-metadata-or-release-proof; priority 9.
- @metaharness/host-claude-code: inspect-package-and-repair-metadata-or-release-proof; priority 9.
- @neural-trader/agentic-accounting-core: inspect-package-and-repair-metadata-or-release-proof; priority 9.
- @ruvector/ruvllm: inspect-package-and-repair-metadata-or-release-proof; priority 9.
- @ruvector/sona: inspect-package-and-repair-metadata-or-release-proof; priority 9.
- @claude-flow/cli: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @claude-flow/mcp: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @metaharness/darwin: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @neural-trader/agentic-accounting-mcp: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @neural-trader/example-evolutionary-game-theory: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @neural-trader/example-neuromorphic-computing: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @neural-trader/example-portfolio-optimization: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @neural-trader/example-quantum-optimization: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @ruvector/attention: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- sublinear-time-solver: inspect-package-and-repair-metadata-or-release-proof; priority 8.
- @claude-flow/plugins: inspect-package-and-repair-metadata-or-release-proof; priority 7.
- @claude-flow/swarm: inspect-package-and-repair-metadata-or-release-proof; priority 7.
- @metaharness/radio: inspect-package-and-repair-metadata-or-release-proof; priority 7.
- @metaharness/router: inspect-package-and-repair-metadata-or-release-proof; priority 7.

## Runtime cycles

- strange-loops → sublinear-time-solver
