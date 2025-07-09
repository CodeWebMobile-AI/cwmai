#!/bin/bash
# Script to prepare the zip folder for moving out of the parent directory

echo "=========================================="
echo "Preparing zip folder for independence"
echo "=========================================="

ZIP_DIR="/Users/fortressarigo/PhpstormProjects/cwmai/zip"

# Change to zip directory
cd "$ZIP_DIR" || exit 1

echo "1. Cleaning all state files..."
rm -f system_state.json task_state.json work_queue_backup.json production_context.json
rm -f claude_execution_queue.json dynamic_claude_pool_stats.json continuous_orchestrator_state.json
rm -f task_history.json benchmark_history.json bug_history.json
rm -f cwmai_state.db cwmai_state.db-shm cwmai_state.db-wal ultimate_orchestrator.db
rm -f *.json 2>/dev/null
rm -f scripts/*.json scripts/*.db 2>/dev/null
rm -rf local_repos/state/* 2>/dev/null

echo "✓ Cleaned all state files"

echo ""
echo "2. Verifying local repositories..."
REPO_COUNT=$(find local_repos/repositories -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')
echo "✓ Found $REPO_COUNT repositories in local_repos"

if [ -d "local_repos/repositories/CodeWebMobile-AI" ]; then
    ls -la local_repos/repositories/CodeWebMobile-AI/
fi

echo ""
echo "3. Creating README for zip folder..."
cat > README_ZIP.md << 'EOF'
# Zip Folder - Standalone CWMAI System

This is a self-contained copy of the CWMAI system that can be moved anywhere and run independently.

## Contents
- All scripts and system files from the parent
- Local repository: video-scribe-ai
- No dependencies on parent directory

## Running the System

1. Make sure you have the .env.local file with your API keys
2. Run the system:
   ```bash
   export $(cat .env.local | grep -v '^#' | xargs)
   export PYTHONPATH=$(pwd)
   arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10
   ```

## Important Notes
- The system will only discover repositories in the local_repos directory
- All state files are created within this directory
- No references to parent directory paths
EOF

echo "✓ Created README_ZIP.md"

echo ""
echo "4. Creating standalone run script..."
cat > run_standalone.sh << 'EOF'
#!/bin/bash
# Standalone run script for zip folder

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running CWMAI from: $(pwd)"

# Check for .env.local
if [ ! -f ".env.local" ]; then
    echo "Error: .env.local not found!"
    echo "Please copy your .env.local file to this directory"
    exit 1
fi

# Export environment variables
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH="$SCRIPT_DIR"

# Run the system
echo "Starting system..."
arch -arm64 /usr/bin/python3 run_hybrid_continuous.py "$@"
EOF

chmod +x run_standalone.sh
echo "✓ Created run_standalone.sh"

echo ""
echo "5. Testing clean state..."
export $(cat ../.env.local | grep -v '^#' | xargs) 2>/dev/null
export PYTHONPATH="$ZIP_DIR"

python3 -c "
from scripts.state_manager import StateManager
sm = StateManager()
state = sm.load_state_with_repository_discovery()
print(f'Projects in clean state: {len(state.get(\"projects\", {}))}')
for proj in state.get('projects', {}).keys():
    print(f'  - {proj}')
" 2>/dev/null

echo ""
echo "=========================================="
echo "✅ Zip folder is ready to be moved!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your .env.local file to the zip folder:"
echo "   cp ../.env.local ."
echo ""
echo "2. Move the zip folder wherever you want:"
echo "   mv $ZIP_DIR ~/Desktop/cwmai-standalone"
echo ""
echo "3. Run from the new location:"
echo "   cd ~/Desktop/cwmai-standalone"
echo "   ./run_standalone.sh --workers 10"
echo ""