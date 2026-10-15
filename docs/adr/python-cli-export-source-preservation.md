# Preserve source collection files during CLI export

Status: proposed for review

## Problem and evidence

The real export CLI writes <out>.meta.json as an export metadata list. Giving the collection path as the output prefix overwrites its sidecar, making Collection.load reject the collection after a reported successful export. Symlink and hardlink output aliases have the same consequence. An output index alias can also rewrite the source index unnecessarily. Four baseline CLI cases demonstrate missing overlap refusal with one ordinary separate-output control passing.

## Decision

Before either export write, compare both output paths against the collection index and sidecar using resolved paths and existing-file identity. Reject overlap with a ClickException. This keeps the documented separate-file export format and permits ordinary replacement of unrelated export output files. The constant-size checks add no vector-sized allocation.

## Validation and limits

Actual CLI/native tests cover direct, symbolic and hard-link aliases and validate preserved source bytes plus successful reload. Full Python validation passes300 tests with2 skips against freshly built upstream079 native; independent exact-commit producer review is required. These are offline local files, not a remote storage guarantee. This does not prevent an external process changing links after the check, add atomic output pairs, or change import's explicitly documented overwrite behavior.

## Changelog

Reject collection-overlapping CLI export destinations before any export file is written.
