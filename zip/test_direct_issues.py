#!/usr/bin/env python3
"""Direct test of issue creation"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.append('.')
os.environ['STORAGE_MODE'] = 'LOCAL'

async def test():
    from scripts.project_creator import ProjectCreator
    from scripts.local_github_simulator import LocalRepository
    
    # Create instances
    pc = ProjectCreator(None, None)
    
    # Use actual verisafe
    repo_path = Path("local_repos/repositories/CodeWebMobile-AI/verisafe").resolve()
    repo = LocalRepository(
        'verisafe', 
        'CodeWebMobile-AI/verisafe', 
        'AI-powered smart contract auditing', 
        str(repo_path)
    )
    
    # Simple details
    details = {
        'name': 'verisafe',
        'initial_features': ['Contract Auditing', 'Vulnerability Detection'],
        'target_audience': 'Web3 developers',
        'architecture': {}
    }
    
    print(f"Repo path: {repo.path}")
    print(f"Repo exists: {Path(repo.path).exists()}")
    
    # Call the method
    result = await pc._create_initial_issues(repo, details)
    print(f"\nResult: {result}")
    
    # Check what was created
    issues_dir = Path(repo.path) / "issues"
    if issues_dir.exists():
        issues = list(issues_dir.glob("*.json"))
        print(f"\nCreated {len(issues)} issues:")
        for issue in sorted(issues):
            print(f"  - {issue.name}")

asyncio.run(test())