#!/bin/bash
# Script to run the system from within the zip folder
# This ensures all paths are relative to the zip folder, not the parent directory

# Save current directory
ORIGINAL_DIR=$(pwd)
ZIP_DIR="/Users/fortressarigo/PhpstormProjects/cwmai/zip"

echo "==========================================="
echo "Running system from within zip folder"
echo "==========================================="

# Check if zip directory exists
if [ ! -d "$ZIP_DIR" ]; then
    echo "Error: Zip directory not found at $ZIP_DIR"
    exit 1
fi

# Change to zip directory
cd "$ZIP_DIR"
echo "Changed to directory: $(pwd)"

# Check for local_repos in zip folder
if [ -d "local_repos" ]; then
    echo "✓ Found local_repos directory in zip folder"
    if [ -d "local_repos/repositories" ]; then
        REPO_COUNT=$(find local_repos/repositories -mindepth 2 -maxdepth 2 -type d | wc -l)
        echo "✓ Found $REPO_COUNT repositories in zip/local_repos/repositories"
    else
        echo "⚠️  No repositories directory found in zip/local_repos"
        mkdir -p local_repos/repositories/CodeWebMobile-AI
        echo "✓ Created empty repositories directory"
    fi
else
    echo "⚠️  No local_repos directory found in zip folder"
    mkdir -p local_repos/repositories/CodeWebMobile-AI
    echo "✓ Created local_repos directory structure"
fi

# Clean up ALL state files to ensure fresh start
echo "Cleaning all state files..."
rm -f system_state.json task_state.json work_queue_backup.json production_context.json 2>/dev/null
rm -f claude_execution_queue.json dynamic_claude_pool_stats.json 2>/dev/null
rm -f continuous_orchestrator_state.json task_history.json 2>/dev/null
rm -f *.db *.db-shm *.db-wal 2>/dev/null
echo "✓ All state files cleaned"

echo ""
echo "Setting up environment variables..."

# Export environment variables from parent's .env.local
if [ -f "../.env.local" ]; then
    export $(cat ../.env.local | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env.local"
else
    echo "⚠️  Warning: .env.local not found in parent directory"
fi

# Set PYTHONPATH to current (zip) directory
export PYTHONPATH="$ZIP_DIR"
echo "✓ Set PYTHONPATH=$PYTHONPATH"

# Run the command with all arguments passed to this script
echo ""
echo "Starting system with command: arch -arm64 /usr/bin/python3 $@"
echo "==========================================="
echo ""

# Run the Python script with all arguments
arch -arm64 /usr/bin/python3 "$@"

# Return to original directory
cd "$ORIGINAL_DIR"