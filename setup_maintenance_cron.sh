#!/bin/bash

# Setup cron job for repository maintenance
# This script sets up automatic repository maintenance to run every 6 hours

echo "Setting up repository maintenance cron job..."

# Get the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create the cron command
CRON_CMD="cd $SCRIPT_DIR && /usr/bin/python3 scheduled_repository_maintenance.py >> maintenance_cron.log 2>&1"

# Check if the cron job already exists
if crontab -l 2>/dev/null | grep -q "scheduled_repository_maintenance.py"; then
    echo "Cron job already exists. Removing old entry..."
    # Remove existing cron job
    (crontab -l 2>/dev/null | grep -v "scheduled_repository_maintenance.py") | crontab -
fi

# Add new cron job (runs every 6 hours)
(crontab -l 2>/dev/null; echo "0 */6 * * * $CRON_CMD") | crontab -

echo "✅ Repository maintenance cron job set up successfully!"
echo "   - Runs every 6 hours (at 00:00, 06:00, 12:00, 18:00)"
echo "   - Logs output to: $SCRIPT_DIR/maintenance_cron.log"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove this cron job: crontab -l | grep -v 'scheduled_repository_maintenance.py' | crontab -"
echo ""
echo "You can also run the maintenance manually:"
echo "  python3 $SCRIPT_DIR/scheduled_repository_maintenance.py"