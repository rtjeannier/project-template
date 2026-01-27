# Claude Code Instructions

This file contains instructions for Claude Code when working on this project.
These standards apply to ALL Claude Code sessions (interactive and automated).

## Code Style

- Follow existing patterns in the codebase
- Keep changes focused and minimal
- Don't refactor unrelated code
- Add comments only where logic isn't self-evident
- Create plans before implementation for non-trivial tasks
- Write clear commit messages - explain the "why", not just the "what"

## Testing

- Run existing tests before submitting PR
- Add tests for new functionality when appropriate
- Ensure the build passes

## Git Operations

**Interactive sessions:** Handle git operations normally (create branches, commit, push, create PRs).

**@claude-code workflow:** Focus only on code changes. The GitHub Actions workflow handles:
- Branching (checks out PR branch or creates new branch from main)
- Committing changes
- Pushing to remote
- Creating/updating PRs
- Posting responses to issues/PRs

## Workflow Rules

1. **Never commit directly to main** - Always use feature branches
2. **Submit changes via PR** - All changes go through pull requests for review
