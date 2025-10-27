#!/bin/bash

# AI Trading Bot - VPS Installation Script
# For Hostinger VPS or any Ubuntu/Debian VPS

set -e

echo "========================================="
echo "AI Trading Bot - VPS Installation"
echo "========================================="
echo ""

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed successfully!"
else
    echo "Docker already installed"
fi

# Install Docker Compose
echo "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully!"
else
    echo "Docker Compose already installed"
fi

# Install Git
echo "Installing Git..."
sudo apt-get install -y git

# Install other utilities
echo "Installing utilities..."
sudo apt-get install -y \
    htop \
    curl \
    wget \
    vim \
    screen

# Setup firewall (optional)
echo "Setting up firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 22/tcp  # SSH
    sudo ufw --force enable
    echo "Firewall configured"
fi

# Clone repository (if not already cloned)
if [ ! -d "trading-bot-IA" ]; then
    echo "Cloning repository..."
    # git clone <your-repo-url> trading-bot-IA
    echo "Please manually clone your repository"
fi

# Create .env file
echo ""
echo "========================================="
echo "Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Navigate to the bot directory: cd trading-bot-IA"
echo "2. Copy .env.example to .env: cp .env.example .env"
echo "3. Edit .env with your credentials: nano .env"
echo "4. Make scripts executable: chmod +x scripts/*.sh"
echo "5. Start the bot: ./scripts/start.sh"
echo ""
echo "Note: You may need to logout and login again for Docker permissions to take effect"
echo ""
