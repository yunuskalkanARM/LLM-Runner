---
name: llm-update-downloads
description: Update scripts/py/requirements.json entries (URLs + sha256sum) for models/tools, validate hash changes, and keep downloads deterministic without committing artifacts. Use when adding or refreshing model/tool downloads.
---

# Update downloads (requirements.json)

Use this when adding or refreshing entries in `scripts/py/requirements.json`.

Windows note: if `python3` isn’t available, use `python` (or `py -3`) for the scripts below.

## Workflow

### 1) Update the manifest

- Edit `scripts/py/requirements.json`.
- Ensure each entry has `url`, `destination`, and `sha256sum`.
- Keep paths under `resources_downloaded/` (but never commit the artifacts).

### 2) Compute SHA256 for local files (if applicable)

```sh
python3 skills/llm-add-model-support/scripts/sha256_file.py <file>
```

### 3) Optional validation

- If you need to confirm the downloader logic, run a configure step that triggers downloads:

```sh
cmake --preset=native -B build
```

Tip: avoid large downloads unless explicitly requested.

### 4) Docs and hygiene

- If downloads affect user-facing setup, update `README.md`.
- Never commit `resources_downloaded/` or `download.log`.
