#!/usr/bin/env bash
# Install git hooks from hooks/ into .git/hooks/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

for hook in "$SCRIPT_DIR"/pre-*; do
    name="$(basename "$hook")"
    cp "$hook" "$GIT_HOOKS_DIR/$name"
    chmod +x "$GIT_HOOKS_DIR/$name"
    echo "installed $name"
done

echo "done"
