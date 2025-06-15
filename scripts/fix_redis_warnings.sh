#!/bin/bash

# Fix Redis Performance Warnings Script
# This script addresses the two common Redis warnings:
# 1. overcommit_memory setting
# 2. Transparent Huge Pages (THP)

echo "Redis Performance Optimization Script"
echo "===================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires root privileges. Please run with sudo:"
    echo "sudo $0"
    exit 1
fi

echo "Fixing Redis performance warnings..."
echo ""

# Fix 1: Set vm.overcommit_memory to 1
echo "1. Setting vm.overcommit_memory = 1..."
sysctl vm.overcommit_memory=1

# Make it persistent across reboots
if ! grep -q "vm.overcommit_memory = 1" /etc/sysctl.conf; then
    echo "vm.overcommit_memory = 1" >> /etc/sysctl.conf
    echo "   Added to /etc/sysctl.conf for persistence"
else
    echo "   Already in /etc/sysctl.conf"
fi

echo ""

# Fix 2: Disable Transparent Huge Pages
echo "2. Disabling Transparent Huge Pages..."
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/defrag

# Make THP settings persistent across reboots
RC_LOCAL="/etc/rc.local"
if [ ! -f "$RC_LOCAL" ]; then
    # Create rc.local if it doesn't exist
    cat > "$RC_LOCAL" << 'EOF'
#!/bin/bash
# rc.local - executed at the end of each multiuser runlevel
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/defrag
exit 0
EOF
    chmod +x "$RC_LOCAL"
    echo "   Created $RC_LOCAL with THP settings"
else
    # Add to existing rc.local if not already present
    if ! grep -q "transparent_hugepage" "$RC_LOCAL"; then
        # Insert before the exit 0 line
        sed -i '/^exit 0/i echo madvise > /sys/kernel/mm/transparent_hugepage/enabled\necho madvise > /sys/kernel/mm/transparent_hugepage/defrag' "$RC_LOCAL"
        echo "   Added THP settings to $RC_LOCAL"
    else
        echo "   THP settings already in $RC_LOCAL"
    fi
fi

echo ""
echo "Verification:"
echo "============="
echo "vm.overcommit_memory = $(sysctl -n vm.overcommit_memory)"
echo "THP enabled = $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "THP defrag = $(cat /sys/kernel/mm/transparent_hugepage/defrag)"

echo ""
echo "✅ Redis performance optimizations applied!"
echo ""
echo "Note: If Redis is currently running, you should restart it:"
echo "  sudo systemctl restart redis"
echo "  or"
echo "  sudo service redis-server restart"