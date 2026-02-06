<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# API change checklist (this repo)

Use this when changing the public API or behavior to ensure all layers (config, backends, JNI, tests, docs) stay consistent.

## C++ API surface

- Update the public header: `src/cpp/interface/Llm.hpp`
- Prefer additive changes (new methods/params with defaults) over breaking changes
- If you must break API/behavior, document it in `README.md` and update tests to match the new contract
- Maintain invariants:
  - `LlmInit()` must be called before encode/decode paths
  - `NextToken()` returns `LLM::endToken` (`"<eos>"`) for stop tokens
- Confirm error behavior is consistent (exceptions vs return values) for invalid state/arguments

## Implementation glue

- Update implementation: `src/cpp/Llm.cpp`
- If the change affects cross-thread/cancel behavior, review:
  - `LLM::CancellableNextToken()`, `LLM::Cancel()`, `LLM::StopGeneration()`
- If a change impacts resource lifetime, verify `~LLM()` and `FreeLlm()`
- If changing logging/error messages, keep them stable and high-signal (tests/debug tooling may key off substrings)

## Config schema changes

- Update parser: `src/cpp/config/LlmConfig.*`
- Update tests: `test/cpp/LlmConfigTest.cpp`
- Ensure new fields have safe defaults for existing configs
- Update example configs / docs if users are expected to set the new key(s): `model_configuration_files/`, `README.md`

## Backend implementations

- If an API change adds capability requirements, ensure every backend either:
  - Implements it, or
  - Fails clearly with a good error message
- Verify modality support checks remain correct (text/image)
- Consistency check each backend’s mapping of runtime knobs if touched:
  - `contextSize`, `batchSize`, `numThreads`
  - chat templating (`applyDefaultChatTemplate`, templates, stop words)

## JNI bindings (if BUILD_JNI_LIB=ON)

- Update JNI: `src/cpp/LlmJni.cpp` and the Java interface under `src/java/`
- If changing signatures, ensure tests still run (see JNI tests wired in `test/CMakeLists.txt`)
- If JNI is not relevant to the feature, validate at least one build with `-DBUILD_JNI_LIB=OFF` to isolate issues
- If behavior changes, update JNI tests to print enough context on failure (prompt/response/config)

## Tests and proof

- Minimum (for any API/behavior change):
  - `cmake --preset=native -B build`
  - `cmake --build ./build --parallel`
  - `ctest --test-dir ./build --output-on-failure`
- If the change is backend-specific, configure and run CTest for that backend explicitly:
  - `cmake --preset=native -B build -DLLM_FRAMEWORK=<backend>`
  - `ctest --test-dir ./build --output-on-failure`
- If JNI is enabled in your preset, run at least one JNI test target as well (or explicitly justify skipping):
  - `ctest --test-dir ./build -R llm-jni-ctest -V --output-on-failure`

## Common “did you also update…?”

- `README.md` if user-visible usage changed (flags, supported models/backends, config keys, semantics)
- `TROUBLESHOOTING.md` if platform/toolchain behavior changed
- Benchmarks (`src/cpp/benchmark/`) if the API affects tokenization/encode/decode flow or timing
