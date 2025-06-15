#!/bin/bash

# Install systemd service for repository maintenance
# This provides an alternative to cron with better logging and control

echo "Installing repository maintenance systemd service..."

# Check if running with sufficient privileges
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo ./install_maintenance_service.sh"
    exit 1
fi

# Get the current directory and user
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CURRENT_USER=$(logname)

# Update paths in service file
sed -i "s|/workspaces/cwmai|$SCRIPT_DIR|g" repository-maintenance.service
sed -i "s|%i|$CURRENT_USER|g" repository-maintenance.service

# Copy service files to systemd directory
cp repository-maintenance.service /etc/systemd/system/
cp repository-maintenance.timer /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable and start the timer
systemctl enable repository-maintenance.timer
systemctl start repository-maintenance.timer

echo "✅ Repository maintenance service installed successfully!"
echo ""
echo "Useful commands:"
echo "  - Check timer status: systemctl status repository-maintenance.timer"
echo "  - Check service status: systemctl status repository-maintenance.service"
echo "  - View logs: journalctl -u repository-maintenance.service -f"
echo "  - Run manually: systemctl start repository-maintenance.service"
echo "  - Stop timer: systemctl stop repository-maintenance.timer"
echo "  - Disable timer: systemctl disable repository-maintenance.timer"