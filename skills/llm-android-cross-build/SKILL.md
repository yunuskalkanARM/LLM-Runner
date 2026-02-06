---
name: llm-android-cross-build
description: Cross-compile for Android (x-android-aarch64 preset), optionally build tests/benchmarks, and outline adb push/run steps. Use when building for Android targets or diagnosing Android toolchain issues.
---

# Android cross-build

Use this to build for Android targets using the `x-android-aarch64` preset.

Windows note: if `python3` isn’t available, use `python` (or `py -3`) for any scripts below.

## Offline/build caveat

During configure, CMake may use FetchContent to download dependencies from the network. If your network is blocked, configure can fail before tests run. To enable network access when the process runs in a sandbox, see `skills/README.md`. Otherwise, run the build on your local machine (with network access) and share the resulting build path with the tool (for example: `build dir: /path/to/build`), or point CMake at pre-downloaded dependencies in your environment.

## Workflow

### 1) Configure

```sh
cmake --preset=x-android-aarch64 -B build
```

If configure fails because the NDK is missing, set it explicitly:

```sh
export NDK_PATH=/path/to/ndk
cmake --preset=x-android-aarch64 -B build
```

Note: configure triggers downloads via the standard downloader; set `HF_TOKEN` (or `~/.netrc`) for gated models.

Optional flags:

```sh
cmake --preset=x-android-aarch64 -B build -DLLM_FRAMEWORK=llama.cpp -DBUILD_JNI_LIB=ON
cmake --preset=x-android-aarch64 -B build -DBUILD_BENCHMARK=ON
```

### 2) Build

```sh
cmake --build ./build --parallel
```

### 3) (Optional) Push and run on device

Use `adb` to copy binaries and models/configs to the device, then run them from a shell. Keep paths consistent with your config `modelRoot`.

### 4) Troubleshooting

- Ensure NDK is installed and discoverable (Android Studio or `NDK_HOME`).
- If JNI fails, verify `JAVA_HOME` and JDK version.
- See `TROUBLESHOOTING.md` for platform-specific notes.
