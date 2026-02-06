---
name: llm-config-schema-change
description: Safely add or change model config schema keys (JSON) and update parsing, tests, and docs. Use when editing model_configuration_files schema or LlmConfig parsing without doing broader model onboarding.
---

# Config schema change

Use this when adding or modifying JSON config keys under `model_configuration_files/`.

Windows note: if `python3` isn’t available, use `python` (or `py -3`) for any scripts below.

If you are onboarding a model end-to-end, also use `skills/llm-add-model-support/` for config naming, download wiring, and CTest coverage.

## Workflow

### 1) Update parsing and defaults

- Update parsing in `src/cpp/config/` (`LlmConfig.*`).
- Ensure new keys have safe defaults.

### 2) Update tests

- Add/adjust tests in `test/cpp/LlmConfigTest.cpp`.
- If a new key affects runtime behavior, update the relevant integration tests.

### 3) Update example configs

- Update or add JSON files under `model_configuration_files/` that exercise the new keys.

### 4) Update docs

- If users need to set or understand the new key, update `README.md`.
- Add to `TROUBLESHOOTING.md` if there are platform-specific caveats.

### 5) Validate

```sh
cmake --preset=native -B build
cmake --build ./build --parallel
ctest --test-dir ./build --output-on-failure
```
