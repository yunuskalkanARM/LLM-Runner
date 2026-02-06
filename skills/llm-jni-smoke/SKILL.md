---
name: llm-jni-smoke
description: Run a fast JNI-focused build/test smoke check (JNI on, minimal test run), and isolate JNI toolchain issues. Use when changing JNI/Java code or validating JNI setup.
---

# JNI smoke check

Use this for fast JNI validation after Java/JNI changes.

Windows note: if `python3` isn’t available, use `python` (or `py -3`) for any scripts below.

## Offline/build caveat

During configure, CMake may use FetchContent to download dependencies from the network. If your network is blocked, configure can fail before tests run. To enable network access when the process runs in a sandbox, see `skills/README.md`. Otherwise, run the build on your local machine (with network access) and share the resulting build path with the tool (for example: `build dir: /path/to/build`), or point CMake at pre-downloaded dependencies in your environment.

## Workflow

### 1) Configure with JNI enabled

```sh
cmake --preset=native -B build -DBUILD_JNI_LIB=ON
```

### 2) Build

```sh
cmake --build ./build --parallel
```

### 3) Run only the JNI test

```sh
ctest --test-dir ./build -R llm-jni-ctest -V --output-on-failure
```

Tip: list available tests with `ctest --test-dir ./build -N` if the regex doesn't match.

### 4) Troubleshooting

- Set `JAVA_HOME` if `jni.h` is not found.
- If JNI is not relevant, disable with `-DBUILD_JNI_LIB=OFF`.
