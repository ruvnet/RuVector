# ADR 151: Transition from `searchreplace` to Stateful PTY Agent Loop

## 1. Context and Problem Statement

Our current Darwin Mode architecture relies on a `searchreplace` formatting primitive. The model is provided localized files, an issue description, and a test traceback, and is expected to emit a single, perfectly formatted markdown block representing the entire logical fix.

Through extensive testing (ADR-144 through ADR-150), we proved that wrapping models in a closed-loop `pytest` feedback harness doubles their baseline performance. However, our recent `deepseek-v4-pro` floor test mathematically proved that we have hit the **Primitive Ceiling** of this architecture. Regardless of the underlying model's reasoning density, forcing an LLM to guess the complete, multi-file solution in a single string-replacement block restricts the resolve rate. The June 2026 State-of-the-Art (~60% on SWE-bench Pro via frameworks like `mini-SWE-agent`) relies on multi-step exploration and live tool-use.

To cross our current 58.3% ceiling, we must change how the model interacts with the codebase.

## 2. Decision

We will deprecate the single-shot `searchreplace` primitive and replace it with a **Stateful PTY (Pseudo-Terminal) Agent Loop**. The orchestrator will no longer parse markdown patches; it will act as a routing bridge between the LLM and an active bash session inside the SWE-bench Docker container.

### 2.1 The ReAct Tool Schema

The agent will be prompted to think iteratively and interact with the environment via strict JSON tool calls. The schema will be restricted to four core primitives to prevent infinite-loop hallucinations:

1. `execute_bash(command: str)`: Runs any valid bash command (e.g., `grep -rn "def fault" .`, `pytest tests/test_parser.py`, `ls -la`). Returns `stdout`/`stderr`.
2. `read_file(path: str, start_line: int, end_line: int)`: Extracts specific, numbered chunks of code without blowing up the context window.
3. `edit_file(path: str, start_line: int, end_line: int, content: str)`: Replaces a specific block of code.
4. `finish_task()`: Signals to the orchestrator that the patch is complete and ready for the final, official SWE-bench evaluation.

### 2.2 Trajectory and Context Management

* **Max Turns:** The agent will be given a maximum of **50 environment turns** per instance to prevent budget runaway.
* **Terminal Binding:** The orchestrator will bind a persistent PTY to the `swe-bench` testbed container, allowing stateful operations (like navigating directories via `cd` or setting environment variables).
* **Trajectory Memory (Scratchpad):** The system prompt will require the model to begin every turn with a `thought` block, documenting what it learned from the previous bash execution and what it intends to do next.

## 3. Rationale

* **Matches SOTA Mechanics:** Real developers use `grep`, run partial tests, and explore codebases before writing fixes. By giving the model a bash terminal, we align our architecture with the mechanics used by the current leaderboard leaders (GPT-5 Mini + `mini-SWE-agent`).
* **Shatters the "Emission Wall":** Emitting a 3-line JSON tool call to edit 5 lines of code is vastly more reliable than emitting a 200-line markdown `searchreplace` block. Indentation and markdown-escaping errors will drop to near zero.
* **Leverages High-Context Windows:** Modern cheap models (like DeepSeek V4 Pro) have massive context windows (1M+ tokens). We can now feed the entire `stdout` of a test run directly back to the model without truncation fears.

## 4. Consequences

* **Positive:** Unlocks the physical capability to resolve complex, multi-file refactoring bugs, pushing the resolve-rate ceiling toward 60%+.
* **Negative:** Wall-clock time per instance will increase significantly (from ~2 minutes to potentially ~15 minutes).
* **Economic:** Cost per instance will rise due to higher context accumulation over 50 turns. This necessitates using cost-optimized frontier models (`deepseek-v4-pro` or `gpt-5-mini`) as the primary engines rather than heavy legacy models like Sonnet-4.0.
