#!/bin/bash

# AI Trading Bot - Stop Script

set -e

echo "========================================="
echo "AI Trading Bot - Stopping..."
echo "========================================="

# Stop all services
docker-compose down

echo ""
echo "Bot stopped successfully!"
echo ""
