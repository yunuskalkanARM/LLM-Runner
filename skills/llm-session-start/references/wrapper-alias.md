<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Optional: wrapper alias to run checks before your tool session

The tool does not (currently) support “auto-run this command on repo open” based on `AGENTS.md`/skills.

If you want an automatic reminder/check before launching the tool, create a shell function/alias in your shell rc file (outside the repo).

Example (bash/zsh):

```sh
llm-workflow() {
  local repo="/path/to/large-language-models"
  (cd "$repo" && python3 skills/llm-session-start/scripts/start.py build --network) || return $?
  command "$@"
}
```

Then run `llm-workflow <tool-command> ...` for this repo.
