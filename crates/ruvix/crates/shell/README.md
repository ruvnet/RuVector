# ruvix-shell

In-kernel debug shell for the RuVix Cognition Kernel (ADR-087).

## Overview

`ruvix-shell` provides an interactive command-line interface accessible over UART for runtime inspection and debugging of the RuVix kernel.

## Features

- **Command Parser**: History support, argument parsing, whitespace handling
- **13 Commands**: help, info, mem, tasks, caps, vectors, witness, proofs, queues, perf, cpu, trace, reboot
- **Trace Mode**: Enable/disable syscall tracing with visual indicator
- **no_std Compatible**: Works in bare-metal environments with `alloc`

## Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `info` | Display kernel version and boot info |
| `mem` | Memory statistics (physical, regions, slabs) |
| `tasks` | List tasks with state and capabilities |
| `caps [id]` | Capability table dump |
| `queues` | Queue status and statistics |
| `vectors` | Vector store information |
| `proofs` | Proof verification statistics |
| `witness [n]` | Witness log viewer |
| `perf` | Performance counters |
| `cpu` | Per-CPU information (SMP systems) |
| `trace [on/off]` | Enable/disable syscall tracing |
| `reboot` | Restart the kernel |

## Usage

```rust
use ruvix_shell::{Shell, ShellConfig};

let config = ShellConfig::default();
let mut shell = Shell::new(config);

// Process input line
let response = shell.process_line("info");
println!("{}", response);
```

## Example Session

```
RuVix Cognition Kernel v0.1.0
Boot: 2024-01-15 14:32:00 UTC

rvsh> mem
Physical Memory:
  Total:     4096 MB
  Free:      3847 MB (94%)
  Kernel:      64 MB

rvsh> tasks
ID   NAME              STATE    CAPS  PRI
0    idle              RUNNING     1  255
1    init              BLOCKED     8   10
2    vector-service    READY      16    5

rvsh> trace on
Syscall tracing enabled.
rvsh[T]>
```

## License

MIT OR Apache-2.0
