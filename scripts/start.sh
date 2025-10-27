#!/bin/bash

# AI Trading Bot - Start Script
# This script starts the trading bot in production mode

set -e

echo "========================================="
echo "AI Trading Bot - Starting..."
echo "========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed!"
    echo "Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed!"
    echo "Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs data models

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose pull

# Build the application
echo "Building application..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

# Show logs
echo ""
echo "========================================="
echo "Bot started successfully!"
echo "========================================="
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f trading-bot"
echo "  - Stop bot: docker-compose down"
echo "  - Restart bot: docker-compose restart trading-bot"
echo "  - Check status: docker-compose ps"
echo ""

# Follow logs
echo "Following logs (Ctrl+C to exit, bot will keep running)..."
docker-compose logs -f trading-bot
