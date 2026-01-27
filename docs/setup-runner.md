# Setting Up the Self-Hosted Runner

This guide walks you through setting up the GitHub Actions runner on your local PC.

## Prerequisites

- Docker and Docker Compose installed
- GitHub account with access to your repository
- Claude Max Pro subscription (for Claude Code authentication)

## Step 1: Build the Container

From your project root:

```bash
docker-compose build
```

This creates an image with:
- GitHub Actions runner
- Claude Code CLI
- Git and GitHub CLI
- Node.js

## Step 2: Start the Container

```bash
docker-compose up -d
```

The container will start but the runner won't be configured yet.

## Step 3: Get a Runner Token

1. Go to your GitHub repository
2. Settings → Actions → Runners
3. Click "New self-hosted runner"
4. Copy the token shown (it starts with `A...`)

**Note:** Tokens expire after about 1 hour. If yours expires, generate a new one.

## Step 4: Configure the Runner

Enter the container:

```bash
docker exec -it claude-runner bash
```

Configure the runner:

```bash
cd /home/runner
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
```

You'll be prompted for:
- **Runner group:** Press Enter for default
- **Runner name:** Press Enter or give it a name like "local-pc"
- **Labels:** Press Enter for default (self-hosted, Linux, X64)
- **Work folder:** Press Enter for default

Exit the container:

```bash
exit
```

## Step 5: Restart the Container

```bash
docker-compose restart
```

The runner will now start automatically and connect to GitHub.

## Step 6: Verify the Runner

1. Go to your GitHub repository
2. Settings → Actions → Runners
3. You should see your runner listed as "Idle"

## Step 7: Authenticate Claude Code

Enter the container:

```bash
docker exec -it claude-runner bash
```

Authenticate Claude Code:

```bash
claude auth login
```

Follow the prompts to log in with your Claude Max Pro account.

Also authenticate GitHub CLI:

```bash
gh auth login
```

Exit:

```bash
exit
```

## Step 8: Test It

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
