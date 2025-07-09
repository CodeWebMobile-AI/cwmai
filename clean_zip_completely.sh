#!/bin/bash
# Complete cleanup script for zip folder

echo "=== Complete Zip Folder Cleanup ==="
cd /Users/fortressarigo/PhpstormProjects/cwmai/zip

echo "1. Removing ALL JSON files..."
find . -name "*.json" -type f -exec rm -f {} \; 2>/dev/null

echo "2. Removing ALL database files..."
find . -name "*.db" -o -name "*.db-*" | xargs rm -f 2>/dev/null

echo "3. Cleaning state directories..."
rm -rf scripts/work_queue_backup.json 2>/dev/null
rm -rf local_repos/state/* 2>/dev/null

echo "4. Verifying clean state..."
JSON_COUNT=$(find . -name "*.json" -type f 2>/dev/null | wc -l)
DB_COUNT=$(find . -name "*.db*" -type f 2>/dev/null | wc -l)

echo "   - JSON files remaining: $JSON_COUNT"
echo "   - Database files remaining: $DB_COUNT"

echo "5. Repository count..."
REPO_COUNT=$(find local_repos/repositories -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
echo "   - Repositories in local_repos: $REPO_COUNT"

echo ""
echo "✅ Zip folder completely cleaned!"
echo ""
echo "To test clean state:"
echo "cd /Users/fortressarigo/PhpstormProjects/cwmai/zip"
echo 'export $(cat ../.env.local | grep -v "^#" | xargs)'
echo "export PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip"
echo "python3 -c 'from scripts.state_manager import StateManager; sm = StateManager(); state = sm.load_state_with_repository_discovery(); print(f\"Projects: {len(state.get(\\\"projects\\\", {}))}\")'"