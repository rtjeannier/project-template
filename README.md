# Project Template

A template repository for projects using Claude Code with self-hosted GitHub Actions runners.

## What This Template Provides

- **Claude Code workflow** - Trigger Claude via `@claude-code` in issues/PRs
- **Self-hosted runner setup** - Run on your PC or cloud VM
- **Docker isolation** - Secure, containerized environment
- **Works with Claude Max Pro** - No API key required

## Quick Start

### 1. Create Your Project

Click "Use this template" on GitHub to create a new repository.

### 2. Clone and Set Up

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
cd YOUR_PROJECT
```

### 3. Set Up the Runner

Follow the detailed instructions in [docs/setup-runner.md](docs/setup-runner.md).

**Quick version:**

```bash
# Build and start the container
docker-compose up -d

# Enter the container
docker exec -it claude-runner bash

# Configure the runner (get token from GitHub repo settings)
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_PROJECT --token YOUR_TOKEN

# Exit and restart container
exit
docker-compose restart
```

### 4. Authenticate Claude Code

Inside the container:

```bash
docker exec -it claude-runner bash
claude auth login
```

### 5. Start Using

Create an issue or comment on a PR with `@claude-code` followed by your request:

```
@claude-code add a contact form to the homepage
```

## How It Works

```
You comment "@claude-code [task]"
         ↓
GitHub Actions triggers
         ↓
Your self-hosted runner receives the job
         ↓
Claude Code runs inside the container
         ↓
Claude implements the feature
         ↓
PR is created for your review
```

## Security

- Only responds to comments from `rtjeannier` (configured in workflow)
- Code runs isolated in Docker container
- Runner tokens are never committed to the repo

## Documentation

- [Setting Up the Runner](docs/setup-runner.md)
- [Moving to Cloud](docs/moving-to-cloud.md)

## Workflow Files

- `CLAUDE.md` - Instructions for Claude Code
- `.github/workflows/claude-local.yml` - GitHub Actions workflow
- `.devcontainer/` - Docker configuration
- `docker-compose.yml` - Container orchestration
