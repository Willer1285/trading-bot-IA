#!/bin/bash

# AI Trading Bot - Update Script
# Updates the bot to the latest version

set -e

echo "========================================="
echo "AI Trading Bot - Updating..."
echo "========================================="

# Stop the bot
echo "Stopping bot..."
docker-compose down

# Pull latest code (if using git)
if [ -d .git ]; then
    echo "Pulling latest code..."
    git pull
fi

# Rebuild
echo "Rebuilding application..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

echo ""
echo "Update completed successfully!"
echo ""

# Show logs
docker-compose logs -f trading-bot
