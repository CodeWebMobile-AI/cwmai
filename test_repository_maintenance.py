#!/usr/bin/env python3
"""Test script for repository maintenance integration."""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add scripts to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fix_repository_customizations import RepositoryCustomizationFixer
from scripts.ai_brain import AIBrain
from scripts.work_item_types import WorkItem, TaskPriority


async def test_repository_check():
    """Test checking repositories for issues."""
    print("\n🔍 Testing repository issue detection...")
    
    try:
        # Initialize fixer
        fixer = RepositoryCustomizationFixer(
            github_token=os.environ.get('GITHUB_TOKEN'),
            organization='CodeWebMobile-AI'
        )
        
        # Check repositories
        issues = await fixer.check_repositories()
        
        if issues:
            print(f"✅ Found issues in {len(issues)} repositories:")
            for repo_name, repo_issues in issues.items():
                print(f"  - {repo_name}: {len(repo_issues)} issues")
                for issue in repo_issues[:2]:  # Show first 2 issues
                    print(f"    • {issue['type']}: {issue['description']}")
        else:
            print("✅ No issues found in any repositories!")
            
        return issues
        
    except Exception as e:
        print(f"❌ Error checking repositories: {e}")
        return {}


async def test_work_item_generation(issues):
    """Test generating work items from issues."""
    print("\n📝 Testing work item generation...")
    
    work_items = []
    current_time = datetime.now(timezone.utc)
    
    for repo_name, repo_issues in issues.items():
        if repo_name in ['cwmai', '.github']:
            continue
            
        work_item = WorkItem(
            id=f"test-repo-fix-{repo_name}-{int(current_time.timestamp())}",
            task_type="maintenance",
            title=f"Fix customization issues in {repo_name}",
            description=f"Repository {repo_name} has issues: {', '.join(issue['type'] for issue in repo_issues)}",
            priority=TaskPriority.MEDIUM,
            repository=repo_name,
            estimated_cycles=1,
            metadata={
                'auto_generated': True,
                'fix_type': 'repository_customization',
                'issues': repo_issues
            }
        )
        work_items.append(work_item)
        print(f"  ✅ Generated work item for {repo_name}")
    
    print(f"\n✅ Generated {len(work_items)} work items")
    return work_items


async def test_single_fix():
    """Test fixing a single repository."""
    print("\n🔧 Testing single repository fix...")
    
    try:
        # Initialize AI brain
        ai_brain = AIBrain()
        
        # Initialize fixer with AI
        fixer = RepositoryCustomizationFixer(
            github_token=os.environ.get('GITHUB_TOKEN'),
            organization='CodeWebMobile-AI',
            ai_brain=ai_brain
        )
        
        # Check for a repository that needs fixes
        issues = await fixer.check_repositories()
        
        if issues:
            # Get first repository with issues
            test_repo = next(iter(issues.keys()))
            print(f"\n🔧 Testing fix on repository: {test_repo}")
            
            # Fix the repository
            result = await fixer.fix_repository(test_repo)
            
            if result.get('status') == 'fixed':
                print(f"✅ Successfully fixed {test_repo}")
                print(f"   Fixes applied: {len(result.get('fixes_applied', []))}")
            else:
                print(f"❌ Failed to fix {test_repo}: {result.get('message', 'Unknown error')}")
        else:
            print("✅ No repositories need fixing!")
            
    except Exception as e:
        print(f"❌ Error during fix test: {e}")


async def main():
    """Run all tests."""
    print("🧪 Testing Repository Maintenance Integration\n")
    
    # Test 1: Check repositories
    issues = await test_repository_check()
    
    if issues:
        # Test 2: Generate work items
        work_items = await test_work_item_generation(issues)
        
        # Test 3: Fix a single repository
        await test_single_fix()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    # Run tests
    asyncio.run(main())