#!/usr/bin/env bash
#
# UserPromptSubmit hook: detect /tags in user prompts and inject context.
# Exit code 0 = allow prompt (stdout JSON adds context for Claude).
# Exit code 2 = block prompt (not used here).
#
set -euo pipefail

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // ""')

# =====================================================================
# TAG REGISTRY
# To add a new tag: copy an if-block, change the grep pattern and heredoc.
# =====================================================================

CONTEXT_PARTS=()

# --- /wait ---
if echo "$PROMPT" | grep -qiE '(^|\s)/wait(\s|$)'; then
  CONTEXT_PARTS+=("$(cat <<'TAGEOF'
[HOOK: /wait tag detected]
After you create a pull request, you MUST do the following:
1. Record the PR number from the gh pr create output.
2. Run the PR poll as a background Bash command using this script:
     for i in $(seq 1 20); do
       STATE=$(gh pr view <PR_NUMBER> --json state -q '.state')
       if [ "$STATE" != "OPEN" ]; then
         MERGED=$(gh pr view <PR_NUMBER> --json merged -q '.merged')
         REVIEWS=$(gh pr view <PR_NUMBER> --json reviews,comments --jq '{reviews: [.reviews[] | {author: .author.login, state: .state, body: .body}], comments: [.comments[] | {author: .author.login, body: .body}]}')
         if [ "$MERGED" = "true" ]; then
           echo "MERGED"
         else
           echo "CLOSED"
         fi
         echo "$REVIEWS"
         exit 0
       fi
       sleep 30
     done
     echo "TIMEOUT"
3. Replace <PR_NUMBER> with the actual PR number.
4. When the poll returns:
   - MERGED: Report "PR #<number> has been merged." with any review comments.
   - CLOSED: Report "PR #<number> was closed without merging." with any review comments.
   - TIMEOUT: Report "PR #<number> is still open after 10 minutes. Let me know when you've reviewed it."
5. Always include any review comments or requested changes in your report.
TAGEOF
  )")
fi

# --- /draft (uncomment to enable) ---
# if echo "$PROMPT" | grep -qiE '(^|\s)/draft(\s|$)'; then
#   CONTEXT_PARTS+=("$(cat <<'TAGEOF'
# [HOOK: /draft tag detected]
# When creating a pull request, use the --draft flag with gh pr create.
# TAGEOF
#   )")
# fi

# --- /no-pr (uncomment to enable) ---
# if echo "$PROMPT" | grep -qiE '(^|\s)/no-pr(\s|$)'; then
#   CONTEXT_PARTS+=("$(cat <<'TAGEOF'
# [HOOK: /no-pr tag detected]
# Do NOT create a pull request. Only make code changes and commit to the feature branch.
# TAGEOF
#   )")
# fi

# =====================================================================
# OUTPUT
# =====================================================================

if [ ${#CONTEXT_PARTS[@]} -eq 0 ]; then
  exit 0
fi

JOINED_CONTEXT=""
for part in "${CONTEXT_PARTS[@]}"; do
  if [ -n "$JOINED_CONTEXT" ]; then
    JOINED_CONTEXT="$JOINED_CONTEXT"$'\n\n'"$part"
  else
    JOINED_CONTEXT="$part"
  fi
done

jq -n --arg ctx "$JOINED_CONTEXT" '{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": $ctx
  }
}'

exit 0
