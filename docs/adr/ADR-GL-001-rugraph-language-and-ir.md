# ADR-GL-001: RuGraph language and canonical program representation

**Status**: Implemented, validation recorded in the acceptance report
**Date**: 2026-10-16

## Context

RuVector exposes graph, vector and learned relation APIs, but an application must
manually distinguish observations from model predictions and control execution.
The objective is a small executable language that makes these distinctions
checkable, with identical behavior across Rust, TypeScript and Python callers.

## Decision

Implement RuGraph version 1 as a bounded, statically checked dataflow language.
Readable `.rvg` source and strict JSON produce the same Program representation.
The operations are nodes, select, traverse, similar, observe, infer, predict and
assess. Bindings have one definition, are ordered, and cannot reference future
bindings. A program emits either decisions or explicitly tagged predictions.
Unknown versions, fields, operators, duplicate JSON keys and invalid type flows
are errors. JSON map order is canonicalized; array order remains significant.

The Rust core lives in its own workspace, and the real RuVector adapters live in
a second workspace. This permits the core's small dependency graph to be built
without the unrelated server, ONNX, hardware or UI crates. Root Cargo exclusions
make this relationship explicit. The native workspace imports existing graph
and KGE crates by path rather than copying their implementations.

The initial language is deliberately a finite graph DSL. It does not claim
general purpose language completeness, Cypher/GQL conformance, arbitrary tensor
autodiff, distributed transactions or arbitrary tool execution. These are not
hidden fallbacks. Unsupported syntax fails instead of being approximated.

## Alternatives and tradeoffs

| Approach | Delivery and adoption | Runtime and assurance |
|---|---|---|
| General purpose language | Largest compiler/tooling burden | Broad expressiveness, much larger effect surface |
| Ordinary SDK only | Lowest adoption burden | Evidence distinctions remain conventions unless separately enforced |
| Bounded DSL with SDKs | Selected; two frontends and one runtime | Finite checks, structured errors, explicit resource limits |

Reuse established ideas from [Jac](https://www.jac-lang.org/reference/language/osp/),
[MeTTa](https://trueagi-io.github.io/hyperon-experimental/metta/),
[Scallop](https://www.scallop-lang.org/doc/introduction.html), and
[egglog](https://arxiv.org/abs/2304.04332). This is an engineering integration,
not an assertion that graph programming, provenance or typed effects are new.

## Acceptance and rollback

Text and JSON must compile to equal programs and run to identical receipts.
Static tests reject predictions flowing to assess or infer. SDK tests exercise
the actual binary. All changes are additive except the explicit root workspace
exclusions. Rollback removes the two language workspaces and their clients;
existing graph and KGE APIs retain their behavior.
