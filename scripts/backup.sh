#!/bin/bash

# AI Trading Bot - Backup Script
# Creates backup of logs, data, and models

set -e

BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="trading_bot_backup_${TIMESTAMP}.tar.gz"

echo "========================================="
echo "AI Trading Bot - Creating Backup..."
echo "========================================="

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
echo "Creating backup: $BACKUP_FILE"
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='venv' \
    logs/ data/ models/ .env config.yaml

# List backups
echo ""
echo "Backup created successfully!"
echo ""
echo "Available backups:"
ls -lh ${BACKUP_DIR}/

# Clean old backups (keep last 7 days)
find ${BACKUP_DIR} -name "trading_bot_backup_*.tar.gz" -mtime +7 -delete
echo ""
echo "Old backups cleaned (kept last 7 days)"
