#!/usr/bin/env python3
"""
Verify Local Mode Setup

This script verifies that the local mode is properly configured and working.
"""

import os
import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.storage_factory import StorageFactory


def check_environment():
    """Check environment configuration."""
    print("🔍 Checking environment configuration...")
    
    storage_mode = os.getenv('STORAGE_MODE', 'github')
    print(f"   STORAGE_MODE: {storage_mode}")
    
    if storage_mode != 'local':
        print("   ⚠️  Not in local mode. Run: python3 cwmai_local.py switch-local")
        return False
    
    local_path = os.getenv('LOCAL_REPOS_PATH', './local_repos')
    print(f"   LOCAL_REPOS_PATH: {local_path}")
    
    if not Path(local_path).exists():
        print(f"   ❌ Local repos path does not exist: {local_path}")
        return False
    
    print("   ✅ Environment configured correctly")
    return True


def check_repositories():
    """Check local repositories."""
    print("\n🔍 Checking local repositories...")
    
    simulator = LocalGitHubSimulator()
    repos = simulator.list_repositories()
    
    print(f"   Found {len(repos)} repositories")
    
    if len(repos) == 0:
        print("   ⚠️  No repositories found")
        return False
    
    # Check a few key repositories
    key_repos = ['project-analytics-dashboard', 'eco-track-ai', 'hydra-health-ai']
    found_repos = [r.name for r in repos]
    
    for repo_name in key_repos:
        if repo_name in found_repos:
            print(f"   ✅ Found {repo_name}")
        else:
            print(f"   ❌ Missing {repo_name}")
    
    return True


def test_create_issue():
    """Test creating an issue locally."""
    print("\n🔍 Testing issue creation...")
    
    simulator = LocalGitHubSimulator()
    
    # Create a test issue
    try:
        issue = simulator.create_issue(
            "CodeWebMobile-AI/eco-track-ai",
            "Test Issue from Verification",
            "This is a test issue created by the verification script.",
            ["test", "verification"]
        )
        
        print(f"   ✅ Created issue #{issue.number}: {issue.title}")
        
        # Check if we can retrieve it
        issues = simulator.get_issues("CodeWebMobile-AI/eco-track-ai")
        if any(i.number == issue.number for i in issues):
            print(f"   ✅ Successfully retrieved issue")
            return True
        else:
            print(f"   ❌ Could not retrieve created issue")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to create issue: {e}")
        return False


def test_storage_factory():
    """Test storage factory."""
    print("\n🔍 Testing storage factory...")
    
    factory = StorageFactory()
    
    # Check issue creator
    issue_creator = factory.create_issue_creator()
    print(f"   Issue creator type: {type(issue_creator).__name__}")
    
    if factory.storage_mode == 'local':
        from scripts.local_issue_creator import LocalIssueCreator
        if isinstance(issue_creator, LocalIssueCreator):
            print("   ✅ Correct issue creator for local mode")
        else:
            print("   ❌ Wrong issue creator type")
            return False
    
    # Check project creator
    project_creator = factory.create_project_creator()
    print(f"   Project creator type: {type(project_creator).__name__}")
    
    return True


def main():
    """Run all verification checks."""
    print("🚀 Local Mode Verification")
    print("=" * 60)
    
    # Load environment
    env_file = Path('.env.local')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    all_good = True
    
    # Run checks
    if not check_environment():
        all_good = False
    
    if not check_repositories():
        all_good = False
    
    if not test_create_issue():
        all_good = False
    
    if not test_storage_factory():
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! Local mode is working correctly.")
        print("\nYou can now run the Ultimate Orchestrator without GitHub API limits:")
        print("  export $(cat .env.local | grep -v '^#' | xargs) && python3 scripts/ultimate_orchestrator.py")
    else:
        print("❌ Some checks failed. Please review the output above.")
        print("\nTo switch to local mode:")
        print("  python3 cwmai_local.py switch-local")


if __name__ == '__main__':
    main()