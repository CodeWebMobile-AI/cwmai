#!/bin/bash
# Setup automatic repository registration for CWMAI

echo "Setting up automatic repository registration..."

# Get the current directory
CWMAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create the cron script
cat > "$CWMAI_DIR/check_repositories.sh" << 'EOF'
#!/bin/bash
# Auto-register any new repositories found in the filesystem

# Set up environment
export CWMAI_DIR="/Users/fortressarigo/PhpstormProjects/cwmai"
cd "$CWMAI_DIR"

# Load environment variables
if [ -f "$CWMAI_DIR/.env.local" ]; then
    export $(cat "$CWMAI_DIR/.env.local" | grep -v '^#' | xargs)
fi

# Set Python path
export PYTHONPATH="$CWMAI_DIR"

# Run the auto-registration script
/usr/bin/python3 "$CWMAI_DIR/scripts/auto_register_repositories.py" >> "$CWMAI_DIR/logs/repository_registration.log" 2>&1

# Log the check
echo "[$(date)] Repository registration check completed" >> "$CWMAI_DIR/logs/repository_registration.log"
EOF

# Make the script executable
chmod +x "$CWMAI_DIR/check_repositories.sh"

# Create logs directory if it doesn't exist
mkdir -p "$CWMAI_DIR/logs"

# Add to crontab (runs every 5 minutes)
echo "Adding cron job to run every 5 minutes..."

# Check if the cron job already exists
if ! crontab -l 2>/dev/null | grep -q "check_repositories.sh"; then
    # Add the cron job
    (crontab -l 2>/dev/null; echo "*/5 * * * * $CWMAI_DIR/check_repositories.sh") | crontab -
    echo "✅ Cron job added successfully"
else
    echo "ℹ️  Cron job already exists"
fi

# Create a systemd service as an alternative (for systems using systemd)
if command -v systemctl &> /dev/null; then
    echo "Creating systemd timer as an alternative..."
    
    # Create service file
    sudo tee /etc/systemd/system/cwmai-repository-check.service > /dev/null << EOF
[Unit]
Description=CWMAI Repository Registration Check
After=network.target

[Service]
Type=oneshot
ExecStart=$CWMAI_DIR/check_repositories.sh
User=$USER
WorkingDirectory=$CWMAI_DIR

[Install]
WantedBy=multi-user.target
EOF

    # Create timer file
    sudo tee /etc/systemd/system/cwmai-repository-check.timer > /dev/null << EOF
[Unit]
Description=Run CWMAI Repository Check every 5 minutes
Requires=cwmai-repository-check.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start the timer
    sudo systemctl daemon-reload
    sudo systemctl enable cwmai-repository-check.timer
    sudo systemctl start cwmai-repository-check.timer
    
    echo "✅ Systemd timer created and started"
fi

# Run an initial check
echo "Running initial repository check..."
"$CWMAI_DIR/check_repositories.sh"

echo ""
echo "✅ Automatic repository registration setup complete!"
echo ""
echo "The system will now check for new repositories every 5 minutes."
echo "Logs are available at: $CWMAI_DIR/logs/repository_registration.log"
echo ""
echo "To manually run a check:"
echo "  $CWMAI_DIR/check_repositories.sh"
echo ""
echo "To disable the cron job:"
echo "  crontab -e  # Then remove the line with check_repositories.sh"
echo ""
if command -v systemctl &> /dev/null; then
    echo "To check systemd timer status:"
    echo "  sudo systemctl status cwmai-repository-check.timer"
fi