# Agent guide

This repository builds an Arm KleidiAI-enabled LLM wrapper library with a thin, backend-agnostic C++ API and optional JNI bindings. Supported backends are selected at CMake configure time: `llama.cpp`, `onnxruntime-genai`, `mediapipe`, and `mnn`.

## High-signal paths

- `README.md`: build options, supported frameworks/models, basic usage.
- `skills/`: project-local skills for repeatable workflows.
- `CMakePresets.json`: supported build presets.
- `scripts/cmake/`: CMake modules, toolchains, downloads, framework glue.
- `scripts/dev/`: doctor checks, version reports, debug bundle helpers.
- `scripts/py/download_resources.py` and `scripts/py/requirements.json`: deterministic download definitions.
- `model_configuration_files/`: model config JSONs consumed by tests and examples.
- `src/cpp/interface/`: public C++ API.
- `src/cpp/config/`: config parsing and schema handling.
- `src/cpp/frameworks/`: backend integrations.
- `src/cpp/benchmark/`: benchmarking code.
- `src/java/`: Java/JNI surface.
- `test/`: Catch2 C++ tests and optional JNI tests.
- `TROUBLESHOOTING.md`: platform-specific issues and limitations.

## Skills

Project-specific skills live under `skills/`. Use them when the task clearly matches a documented workflow. Keep `AGENTS.md` for durable repo rules; keep detailed procedures in the skills themselves.

## Validation

When a change affects build, test, runtime behavior, CMake, scripts, or public API/configuration, run a local build and CTest before considering the change done:

```sh
cmake --preset=native -B build
cmake --build ./build
ctest --test-dir ./build --output-on-failure
```

If JNI is not relevant, it is reasonable to iterate with `-DBUILD_JNI_LIB=OFF`.

## Docs

Update `README.md` when a change affects something users or reviewers need to know how to build, configure, run, or use. Update `TROUBLESHOOTING.md` for platform-specific issues or new limitations.

## Downloads

Configure may trigger resource downloads through `scripts/cmake/download-resources.cmake` and `scripts/py/download_resources.py`. Some models are gated on Hugging Face.

Preferred token sources:
- `HF_TOKEN`
- `~/.netrc` entry for `huggingface.co`

Do not commit anything from `resources_downloaded/`. Avoid introducing changes that force large re-downloads unless explicitly needed.

## Patch hygiene

- Do not commit `build*/`, `resources_downloaded/`, or `download.log`.
- Avoid checking in large binaries or model artifacts; prefer stable URLs and `sha256sum` updates in `scripts/py/requirements.json`.
- Treat SPDX maintenance as part of the same change, not follow-up work.
- Add the standard repo SPDX header to new source/doc/script files that support comments.
- When editing an existing source/doc/script file that supports comments, ensure the SPDX header is present and the year/range is current. Edit the year only; do not change holder or license text.
- Do not inject SPDX comments into formats that would break consumers, such as JSON. Use a repo-approved sidecar or documentation approach instead.

## Notes

- If `python3` is unavailable, use `python` or `py -3`.
- Useful debug helpers:
  - `python3 scripts/dev/llm_doctor.py --build-dir build`
  - `python3 scripts/dev/collect_debug_bundle.py build .`
