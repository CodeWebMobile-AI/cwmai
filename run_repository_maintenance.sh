#!/bin/bash
# Run Repository Maintenance
# This script can be scheduled via cron to run periodic maintenance

# Set up environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env.local ]; then
    export $(cat .env.local | grep -v '^#' | xargs)
fi

# Log file
LOG_FILE="maintenance_cron.log"

echo "========================================" >> "$LOG_FILE"
echo "Starting maintenance at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the maintenance script
python3 scheduled_repository_maintenance.py >> "$LOG_FILE" 2>&1

# Check exit code
if [ $? -eq 0 ]; then
    echo "Maintenance completed successfully at $(date)" >> "$LOG_FILE"
else
    echo "Maintenance failed at $(date)" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"

# Example cron entries:
# Run every 6 hours:
# 0 */6 * * * /path/to/run_repository_maintenance.sh
#
# Run daily at 2 AM:
# 0 2 * * * /path/to/run_repository_maintenance.sh
#
# Run every 12 hours:
# 0 0,12 * * * /path/to/run_repository_maintenance.sh