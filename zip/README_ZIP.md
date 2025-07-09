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
