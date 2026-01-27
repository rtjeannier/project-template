# Moving the Runner to Cloud

This guide explains how to migrate your self-hosted runner from your local PC to a cloud VM.

## Why Move to Cloud?

- **Always on** - Doesn't depend on your PC being awake
- **Accessible anywhere** - Works even when you're away from home
- **Redundancy** - Run alongside your local runner for load balancing

## Recommended Providers

| Provider | Cheapest Option | Notes |
|----------|-----------------|-------|
| Hetzner | ~$4/month (CX22) | Great value, EU/US locations |
| DigitalOcean | $6/month (Basic) | Simple, good docs |
| Linode | $5/month (Nanode) | Reliable |
| Vultr | $5/month | Many locations |
| Oracle Cloud | Free tier | 4 ARM cores free forever |

**Minimum specs:** 1 vCPU, 2GB RAM, 20GB storage

## Step 1: Create a VM

1. Sign up with your chosen provider
2. Create a VM with Ubuntu 22.04
3. Note the IP address
4. SSH into the VM: `ssh root@YOUR_IP`

## Step 2: Install Docker

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Start Docker on boot
systemctl enable docker
```

## Step 3: Clone Your Project

```bash
# Create directory
mkdir -p /opt/runner
cd /opt/runner

# Clone your project
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

## Step 4: Build and Start

```bash
docker compose build
docker compose up -d
```

## Step 5: Configure the Runner

Get a new runner token from GitHub (Settings → Actions → Runners → New self-hosted runner).

```bash
# Enter container
docker exec -it claude-runner bash

# Configure
cd /home/runner
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN

# Exit
exit

# Restart
docker compose restart
```

## Step 6: Authenticate

```bash
docker exec -it claude-runner bash
claude auth login
gh auth login
exit
```

## Alternative: Transfer Configuration

Instead of reconfiguring, you can copy the runner configuration from your local machine:

### On your local PC:

```bash
# Export volumes
docker run --rm -v runner-config:/data -v $(pwd):/backup alpine tar czf /backup/runner-config.tar.gz -C /data .
docker run --rm -v claude-config:/data -v $(pwd):/backup alpine tar czf /backup/claude-config.tar.gz -C /data .
docker run --rm -v gh-config:/data -v $(pwd):/backup alpine tar czf /backup/gh-config.tar.gz -C /data .
```

### Copy to cloud VM:

```bash
scp runner-config.tar.gz claude-config.tar.gz gh-config.tar.gz root@YOUR_IP:/opt/runner/
```

### On cloud VM:

```bash
cd /opt/runner

# Create volumes and import
docker volume create runner-config
docker volume create claude-config
docker volume create gh-config

docker run --rm -v runner-config:/data -v $(pwd):/backup alpine tar xzf /backup/runner-config.tar.gz -C /data
docker run --rm -v claude-config:/data -v $(pwd):/backup alpine tar xzf /backup/claude-config.tar.gz -C /data
docker run --rm -v gh-config:/data -v $(pwd):/backup alpine tar xzf /backup/gh-config.tar.gz -C /data

# Start container
docker compose up -d
```

**Note:** If transferring runner config, remove the old runner from GitHub settings first to avoid conflicts.

## Running Both Local and Cloud

You can run both runners simultaneously:

1. Keep your local runner as-is
2. Set up cloud runner as described above
3. Both show up in GitHub Actions → Runners
4. Jobs automatically go to whichever is available

This gives you:
- **Redundancy** - If one is offline, the other handles jobs
- **Speed** - Two jobs can run in parallel
- **Flexibility** - Use local when home, cloud when away

## Auto-Updates

To keep your cloud runner updated:

```bash
# Create update script
cat > /opt/runner/update.sh << 'EOF'
#!/bin/bash
cd /opt/runner/YOUR_REPO
git pull
docker compose build
docker compose up -d
EOF

chmod +x /opt/runner/update.sh

# Add to crontab (daily at 3am)
(crontab -l 2>/dev/null; echo "0 3 * * * /opt/runner/update.sh") | crontab -
```

## Security Notes

- Keep your VM updated: `apt update && apt upgrade -y`
- Consider setting up a firewall (only SSH needed)
- Use SSH keys, disable password auth
- The runner token is stored in the Docker volume, not in git
