---
name: llm-code-review
description: Review a change in this repo and produce actionable review comments (correctness, tests, docs, portability, performance) formatted for copy/paste into a code review tool. Use when the user says “review this change”, “code review”, “CR”, “patchset review”, or wants AI reviewer feedback.
---

# Code review

Use this to review a checked-out change in this repo and produce a set of review comments you can paste into your code review tool.

Windows note: if `python3` isn’t available, use `python` (or `py -3`) for any scripts below.

## Workflow

### 1) Gather review context (local git)

Run the minimal read-only context:

```sh
git status --porcelain
git diff --stat
git diff
```

If the change is a single commit:

```sh
git show --stat
git show
```

### 2) Apply repo expectations (scoped)

- If the change affects build/test/runtime behavior: run build + `ctest` (see `AGENTS.md` “Validation (expected)”).
- If the change affects anything documented: update `README.md` (see `AGENTS.md` “Docs updates (scoped)”).
- If the change affects public API/behavior: follow `skills/llm-change-api-safely/SKILL.md` and its checklist.
- If tests fail due to model output drift/context: use `skills/llm-debug-test-failures/SKILL.md`.

### 3) Review checklist (high-signal)

Focus on:
- Correctness and error handling (bad inputs, missing files, null/empty paths, context overflow paths).
- API contracts and backward compatibility (C++ and JNI).
- Test quality (assertions high-signal; debug output sufficient; avoid weakening to “non-empty”).
- Performance/robustness (token loops, circuit breakers, thread counts, large file IO).
- Portability (Linux/macOS/Windows; avoid bash-only assumptions; path separators; `python3` vs `python`).
- Security/safety basics (don’t log secrets like `HF_TOKEN`; avoid writing into `resources_downloaded/`).

Style/maintainability:
- C++ naming/shape:
  - public API follows existing style in `src/cpp/interface/` (method names, enums, types)
  - avoid surprising behavior changes; keep functions single-responsibility where practical
  - prefer explicit types and clear names over abbreviations
  - check `const` correctness, references vs copies, and move semantics for large strings/containers
- Java naming/shape:
  - follow existing Java style in `src/java/` (methods/fields/casing consistent with file)
  - avoid exposing native-handle misuse (clear init/free lifecycle; meaningful exceptions)
- Naming consistency across layers:
  - if a concept exists in both C++ and Java/JNI, keep names aligned (e.g., `resetContext` vs `ResetContext`)
  - avoid introducing multiple terms for the same thing (model root vs model dir)
- Unused/accidental complexity:
  - unused variables/fields/imports/includes
  - dead code paths and commented-out blocks without rationale
  - duplicated logic that can be centralized (especially in tests)
- Logging ergonomics:
  - error messages should be actionable and stable (tests/debug workflows may rely on substrings)
  - avoid overly noisy logs in hot paths unless behind a debug flag
- Tests and determinism:
  - keep prompts constrained and assertions high-signal
  - ensure new debug flags default to off and don’t slow CI unnecessarily
- Docs alignment:
  - if the change touches flags/options/config keys/supported models, ensure `README.md` is updated per `AGENTS.md` “Docs updates (scoped)”

### 4) Produce review-ready comments

Output in a paste-friendly structure:

```
General:
- <comment>

Files:
- <path>:<line>: <comment> (suggestion: <what to change>)
```

Prefer a small number of high-signal comments (blockers first, then nits).
