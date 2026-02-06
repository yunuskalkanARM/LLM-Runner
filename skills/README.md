<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Project-local skills

This folder contains **repo-specific skills** that make common workflows repeatable and safe.

Skills are intentionally:
- **actionable** (exact commands and files),
- **offline-friendly by default** (avoid downloads/network unless explicitly requested),
- **small and focused** (one workflow per skill).

Notes:
- Commands use `python3` by default; on Windows, use `python` (or `py -3`) if `python3` isn’t available.
- Skills assume a build directory named `build`; replace it with your actual build dir as needed.
- Some skills can optionally use network access. If the process runs in a sandboxed environment, enable network access in the sandbox configuration, restart the process, and verify with a simple HTTPS HEAD request (for example, to `https://api.github.com`).
  - Example: `[sandbox_workspace_write] network_access = true`
- Build commands use `cmake --build ... --parallel` which is portable. If you want an explicit job count: Linux `$(nproc)`, macOS `$(sysctl -n hw.ncpu)`, Windows `%NUMBER_OF_PROCESSORS%`.

## Skill anatomy

Each skill lives under `skills/<skill-name>/`:

- `SKILL.md`: the entrypoint. It starts with frontmatter:
  - `name:` the skill ID (used when explicitly requested).
  - `description:` used for automatic triggering (include common user phrases like “build”, “ctest”, “add model”, etc.).
- `scripts/`: runnable helpers (bash/python) referenced by the skill.
- `references/`: deeper docs/checklists the skill can load for detail.
- `assets/`: templates/binaries intended to be copied/used, not loaded as context.

Not every skill needs all subfolders.

## When to add a new skill

Add a skill when a workflow:
- is repeated often (build/test, onboarding a model/backend, upgrade tracking),
- has sharp edges (network gating, large downloads, platform toolchains),
- benefits from a checklist or deterministic helper script.

Keep skills short; point to canonical docs (`README.md`, `TROUBLESHOOTING.md`, `AGENTS.md`) instead of duplicating long explanations.

## Keeping skills up to date

Update skills when a change impacts user-visible workflows that a skill documents (commands, flags, paths, env vars, config keys).
SPDX header upkeep is also part of skill maintenance: new or edited `SKILL.md` files and helper scripts should leave the change with the standard repo SPDX header present and the year current.

Quick workflow:
- List surface changes: `git diff --name-only` then `git diff`
- Search for impacted terms under `skills/` (binary names, flags, config keys)
- Prefer pointing to canonical docs over duplicating long explanations
- Before finishing, scan touched skill files for missing/outdated SPDX headers and fix them in the same patch
