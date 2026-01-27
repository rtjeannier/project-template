# Setting Up the Self-Hosted Runner

This guide walks you through setting up the GitHub Actions runner on your local PC.

## Prerequisites

- Docker and Docker Compose installed
- VS Code with Dev Containers extension
- GitHub account with access to your repository
- Claude Max Pro subscription (for Claude Code authentication)

## Understanding the Architecture

**Important:** The VS Code devcontainer and the GitHub Actions runner are the same container. When you open this project in VS Code's devcontainer, you're working inside the environment that will execute your GitHub Actions workflows. This means:

- Authentication you set up (GitHub CLI, Claude Code) persists for workflow runs
- The runner process runs in the same container where you develop
- Your `~/.claude` and `~/.config/gh` directories are mounted from your host machine

## Option A: VS Code Devcontainer Setup (Recommended)

This is the simplest approach - VS Code handles the container for you.

### Step 1: Open in VS Code

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
code .
```

When prompted, click "Reopen in Container" (or run `Dev Containers: Reopen in Container` from the command palette).

### Step 2: Authenticate GitHub CLI

In the VS Code terminal (which is now inside the container):

```bash
gh auth login
```

Follow the prompts to authenticate with GitHub.

### Step 3: Authenticate Claude Code

```bash
claude auth login
```

Follow the prompts to log in with your Claude Max Pro account.

### Step 4: Configure the Runner Using GitHub CLI

Use the GitHub CLI to automatically get a registration token and configure the runner:

```bash
# Get repository info
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')

# Get a runner registration token via GitHub API
TOKEN=$(gh api -X POST repos/$REPO/actions/runners/registration-token -q '.token')

# Configure the runner
cd /home/runner && ./config.sh --url https://github.com/$REPO --token $TOKEN --unattended --name claude-runner --labels self-hosted,Linux,X64

# Return to workspace
cd /home/runner/work
```

### Step 5: Start the Runner

```bash
# Start the runner in the background
cd /home/runner && nohup ./run.sh > /tmp/runner.log 2>&1 &
```

### Step 6: Verify the Runner is Online

```bash
# Check runner status via GitHub API
gh api repos/$(gh repo view --json nameWithOwner -q '.nameWithOwner')/actions/runners --jq '.runners[] | {name, status, busy}'
```

You should see:
```json
{"busy":false,"name":"claude-runner","status":"online"}
```

## Option B: Docker Compose Setup (Without VS Code)

If you prefer to run the container directly without VS Code.

### Step 1: Build and Start the Container

```bash
docker-compose build
docker-compose up -d
```

### Step 2: Enter the Container

```bash
docker exec -it claude-runner bash
```

### Step 3: Authenticate (Inside Container)

```bash
# Authenticate GitHub CLI
gh auth login

# Authenticate Claude Code
claude auth login
```

### Step 4: Configure the Runner

**Using GitHub CLI (recommended):**

```bash
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
TOKEN=$(gh api -X POST repos/$REPO/actions/runners/registration-token -q '.token')
cd /home/runner && ./config.sh --url https://github.com/$REPO --token $TOKEN --unattended --name claude-runner --labels self-hosted,Linux,X64
```

**Or manually:**

1. Go to your GitHub repository → Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Copy the token shown (starts with `A...`)
4. Run:
   ```bash
   cd /home/runner
   ./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
   ```

### Step 5: Exit and Restart

```bash
exit
docker-compose restart
```

The runner will now start automatically on container startup.

## Testing the Setup

Create an issue in your repository:

```
Title: Test Claude Code
Body: @claude-code say hello by creating a file called hello.txt
```

Watch the Actions tab - you should see a workflow run.

## Running on Startup

The `docker-compose.yml` is configured with `restart: unless-stopped`, so the container will automatically start when your PC boots (as long as Docker starts automatically).

To ensure Docker starts on boot:
- **Windows:** Docker Desktop starts automatically by default
- **macOS:** Docker Desktop starts automatically by default
- **Linux:** `sudo systemctl enable docker`

## Troubleshooting

### Runner shows as "Offline"

```bash
# Check container is running
docker ps

# Check logs
docker logs claude-runner

# Restart
docker-compose restart
```

### Authentication expired

```bash
docker exec -it claude-runner bash
claude auth login
gh auth login
exit
```

### Need to reconfigure runner

```bash
docker exec -it claude-runner bash
cd /home/runner
./config.sh remove
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token NEW_TOKEN
exit
docker-compose restart
```

## Multiple Runners

You can run multiple runners for load balancing:

1. On another machine (or cloud VM), clone the repo
2. Follow the same setup steps
3. Both runners will appear in GitHub, jobs go to whichever is free

GitHub automatically distributes jobs to available runners with matching labels.
