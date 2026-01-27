#!/bin/bash
set -e

# GitHub Actions Runner Entrypoint Script
# This script configures and starts the runner

cd /home/runner

# Check if runner is already configured
if [ ! -f ".runner" ]; then
    echo "Runner not configured. Please run:"
    echo "  ./config.sh --url https://github.com/OWNER/REPO --token YOUR_TOKEN"
    echo ""
    echo "Get your token from:"
    echo "  GitHub Repo -> Settings -> Actions -> Runners -> New self-hosted runner"
    echo ""
    echo "Starting shell for manual configuration..."
    exec /bin/bash
fi

# Start the runner
echo "Starting GitHub Actions Runner..."
exec ./run.sh
