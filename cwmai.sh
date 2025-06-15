#!/bin/bash
# CWMAI Shell Wrapper - Smart Conversational AI Assistant
#
# This script provides a convenient way to run CWMAI from anywhere
# by automatically finding the CWMAI installation directory.

# Find the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in a symlinked location
if [ -L "${BASH_SOURCE[0]}" ]; then
    # Follow the symlink to find the real location
    REAL_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$( cd "$( dirname "$REAL_SCRIPT" )" && pwd )"
fi

# Change to the CWMAI directory
cd "$SCRIPT_DIR" || {
    echo "Error: Cannot change to CWMAI directory"
    exit 1
}

# Run the Python script with all arguments
exec python3 cwmai "$@"