#!/bin/bash
# Setup cron job for Redis connection cleanup

echo "Setting up Redis cleanup cron job..."

# Get the full path to the cwmai directory
CWMAI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create the cron job command
CRON_CMD="*/30 * * * * cd $CWMAI_DIR && /usr/bin/python3 cleanup_redis_now.py >/dev/null 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "cleanup_redis_now.py"; then
    echo "✅ Cron job already exists"
    exit 0
fi

# Add the cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ Cron job added successfully!"
    echo "   Redis connections will be cleaned every 30 minutes"
    echo ""
    echo "Current cron jobs:"
    crontab -l | grep redis
    echo ""
    echo "To remove this cron job later, run:"
    echo "  crontab -e"
    echo "  # Then delete the line containing 'cleanup_redis_now.py'"
else
    echo "❌ Failed to add cron job"
    exit 1
fi