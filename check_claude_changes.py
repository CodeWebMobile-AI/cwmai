#!/usr/bin/env python3
"""
Check if Claude tasks actually made changes to the repositories.
"""

import os
import subprocess
from datetime import datetime, timezone
import json

def check_repository_changes(repo_path, task_start_time):
    """Check if repository has changes after task start time."""
    os.chdir(repo_path)
    repo_name = os.path.basename(repo_path)
    
    print(f"\n{'='*60}")
    print(f"Repository: {repo_name}")
    print(f"Task started: {task_start_time}")
    print('='*60)
    
    # Get git status
    result = subprocess.run(['git', 'status', '--porcelain'], 
                          capture_output=True, text=True)
    
    if not result.stdout:
        print("❌ No uncommitted changes")
        return False
    
    print(f"✅ Found {len(result.stdout.splitlines())} changed files")
    
    # Check timestamps of changed files
    changes_after_task = 0
    task_start_timestamp = datetime.fromisoformat(task_start_time.replace('Z', '+00:00')).timestamp()
    
    for line in result.stdout.splitlines()[:10]:  # Check first 10 files
        status = line[:2]
        filename = line[3:]
        
        if status == '??' and os.path.exists(filename):
            # New file
            stat = os.stat(filename)
            mtime = stat.st_mtime
            if mtime > task_start_timestamp:
                changes_after_task += 1
                mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  📝 {filename} - created at {mod_time}")
        elif os.path.exists(filename):
            # Modified file
            stat = os.stat(filename)
            mtime = stat.st_mtime
            if mtime > task_start_timestamp:
                changes_after_task += 1
                mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  ✏️  {filename} - modified at {mod_time}")
    
    if changes_after_task > 0:
        print(f"\n✅ Found {changes_after_task} files changed AFTER task started")
        return True
    else:
        print(f"\n⚠️  Files were changed BEFORE task started")
        return False

# Check each repository
base_path = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI"
tasks = [
    ("micronutrient-optimizer", "2025-06-26T15:14:31.275300+00:00"),
    ("subtrack-ai", "2025-06-26T15:14:32.011542+00:00"),
    ("ai-text-summarizer", "2025-06-26T16:49:51.715103+00:00"),
    ("social-spark-ai", "2025-06-26T16:49:52.474852+00:00")
]

results = {}
for repo, start_time in tasks:
    repo_path = os.path.join(base_path, repo)
    results[repo] = check_repository_changes(repo_path, start_time)

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
for repo, has_changes in results.items():
    status = "✅ Made changes" if has_changes else "❌ No changes after task start"
    print(f"{repo}: {status}")

# Return to original directory
os.chdir("/Users/fortressarigo/PhpstormProjects/cwmai")