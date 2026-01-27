# Claude Code Instructions

This file contains instructions for Claude Code when working on this project.

## Workflow Rules

1. **Never commit directly to main** - Always create a feature branch
2. **Create plans before implementation** - For non-trivial tasks, outline your approach first
3. **Submit changes via PR** - Push your branch and create a pull request
4. **Write clear commit messages** - Explain the "why", not just the "what"

## When Triggered via @claude-code

When triggered by a GitHub issue or PR comment containing `@claude-code`:

1. Read and understand the request in the comment
2. Create a new branch from main: `git checkout -b feature/<short-description>`
3. Implement the requested changes
4. Commit with a clear message
5. Push the branch and create a PR
6. Link the PR to the original issue

## Code Style

- Follow existing patterns in the codebase
- Keep changes focused and minimal
- Don't refactor unrelated code
- Add comments only where logic isn't self-evident

## Testing

- Run existing tests before submitting PR
- Add tests for new functionality when appropriate
- Ensure the build passes
