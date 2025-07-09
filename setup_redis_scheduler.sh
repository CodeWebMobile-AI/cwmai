#!/bin/bash
# Setup scheduled Redis cleanup for macOS using launchd

echo "Setting up Redis cleanup scheduler for macOS..."

PLIST_NAME="com.cwmai.redis-cleanup"
PLIST_FILE="$PLIST_NAME.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCHD_DIR"

# Copy plist file to LaunchAgents
cp "$PLIST_FILE" "$LAUNCHD_DIR/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy plist file"
    exit 1
fi

# Load the job
launchctl load "$LAUNCHD_DIR/$PLIST_FILE" 2>/dev/null

# Check if already loaded
if launchctl list | grep -q "$PLIST_NAME"; then
    echo "✅ Redis cleanup scheduler is active!"
    echo "   Cleanup will run every 30 minutes"
    echo ""
    echo "To check status:"
    echo "  launchctl list | grep $PLIST_NAME"
    echo ""
    echo "To view logs:"
    echo "  tail -f /tmp/cwmai-redis-cleanup.log"
    echo ""
    echo "To stop the scheduler:"
    echo "  launchctl unload ~/Library/LaunchAgents/$PLIST_FILE"
    echo "  rm ~/Library/LaunchAgents/$PLIST_FILE"
else
    echo "⚠️  Scheduler might not be active. Try:"
    echo "  launchctl load -w ~/Library/LaunchAgents/$PLIST_FILE"
fi