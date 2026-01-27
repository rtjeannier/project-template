#!/usr/bin/env bash
#
# PreToolUse hook: block git push to main/master branches.
# Exit code 2 = deny the tool call. Exit code 0 = allow.
#
set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# Only inspect commands that contain "git push"
if ! echo "$COMMAND" | grep -qE '\bgit\s+push\b'; then
  exit 0
fi

# Check 1: Explicit branch name in the command
# Matches: git push origin main, git push -u origin main, git push --force origin master
if echo "$COMMAND" | grep -qE '\bgit\s+push\b.*\b(main|master)\b'; then
  echo "BLOCKED: Pushing directly to a protected branch (main/master) is not allowed. Use a feature branch and create a PR instead." >&2
  exit 2
fi

# Check 2: Refspec targeting main/master
# Matches: git push origin HEAD:main, git push origin HEAD:refs/heads/main
if echo "$COMMAND" | grep -qE '\bgit\s+push\b.*:(refs/heads/)?(main|master)\b'; then
  echo "BLOCKED: Pushing via refspec to a protected branch (main/master) is not allowed. Use a feature branch and create a PR instead." >&2
  exit 2
fi

# Check 3: Bare "git push" while on main/master
# A bare push (no explicit branch) pushes the current branch.
# Strip "git push" and flags, count remaining words.
STRIPPED=$(echo "$COMMAND" | sed -E 's/\bgit\s+push\b//' | sed -E 's/\s+--?[a-zA-Z-]+(=\S+)?//g' | xargs)
WORD_COUNT=$(echo "$STRIPPED" | wc -w | xargs)

if [ "$WORD_COUNT" -le 1 ]; then
  CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
    echo "BLOCKED: You are on '$CURRENT_BRANCH' and running a bare git push. Pushing directly to a protected branch is not allowed. Switch to a feature branch first." >&2
    exit 2
  fi
fi

exit 0
