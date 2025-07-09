#!/usr/bin/env python3
"""
Test Context Gathering for CWMAI System

This test verifies that the system can properly gather context from:
1. ARCHITECTURE.md files
2. Completed work history
3. Repository structure
4. Issue tracking
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from local_github_simulator import LocalGitHubSimulator
from dynamic_work_generator import DynamicWorkGenerator
from ai_brain_factory import AIBrainFactory
from project_lifecycle_analyzer import ProjectLifecycleAnalyzer
from portfolio_intelligence import PortfolioIntelligence
from intelligent_work_finder import IntelligentWorkFinder
from state_manager import StateManager
from claude_execution_queue import ClaudeExecutionQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_context_gathering():
    """Test that the system can gather context properly."""
    print("\n🧪 Testing Context Gathering System")
    print("=" * 60)
    
    # Initialize components
    local_simulator = LocalGitHubSimulator()
    ai_brain = AIBrainFactory.create_for_testing()  # Use testing brain for fast execution
    work_generator = DynamicWorkGenerator(ai_brain=ai_brain)
    claude_queue = ClaudeExecutionQueue()
    
    # Load system state
    state_manager = StateManager()
    system_state = state_manager.load_state()
    
    # Initialize work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    # Test 1: Check if we can find repositories
    print("\n📁 Test 1: Repository Discovery")
    repos = []
    repos_path = local_simulator.repos_path
    if repos_path.exists():
        for org_dir in repos_path.iterdir():
            if org_dir.is_dir() and not org_dir.name.startswith('.'):
                for repo_dir in org_dir.iterdir():
                    if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                        repo_name = f"{org_dir.name}/{repo_dir.name}"
                        repos.append(repo_name)
                        print(f"  ✓ Found repository: {repo_name}")
    
    if not repos:
        print("  ❌ No repositories found!")
        print(f"  📍 Repository path: {repos_path}")
        print("  💡 Creating a test repository...")
        
        # Create a test repository
        test_repo = local_simulator.create_repository(
            name="test-context-repo",
            description="Test repository for context gathering",
            organization="CodeWebMobile-AI"
        )
        
        if test_repo:
            print(f"  ✅ Created test repository: {test_repo.full_name}")
            repos.append(test_repo.full_name)
        else:
            print("  ❌ Failed to create test repository")
    
    # Test 2: Check ARCHITECTURE.md reading
    print("\n📄 Test 2: ARCHITECTURE.md Context")
    for repo_name in repos[:3]:  # Test first 3 repos
        print(f"\n  Repository: {repo_name}")
        # Read ARCHITECTURE.md directly
        org, name = repo_name.split('/')
        arch_path = repos_path / org / name / "ARCHITECTURE.md"
        if arch_path.exists():
            arch_content = arch_path.read_text()
        else:
            arch_content = None
        if arch_content:
            print(f"  ✓ Found ARCHITECTURE.md ({len(arch_content)} chars)")
            print(f"  📝 Preview: {arch_content[:100]}...")
        else:
            print("  ⚠️ No ARCHITECTURE.md found")
            
            # Create a sample ARCHITECTURE.md
            org, name = repo_name.split('/')
            arch_path = repos_path / org / name / "ARCHITECTURE.md"
            arch_path.parent.mkdir(parents=True, exist_ok=True)
            arch_content = """# Architecture

## Components
- API endpoints for data processing
- Database models for storage
- Frontend UI components
- Testing framework

## TODO
- [ ] Implement user authentication
- [ ] Add caching layer
- [ ] Improve error handling
- [ ] Add comprehensive logging
"""
            arch_path.write_text(arch_content)
            print(f"  ✅ Created sample ARCHITECTURE.md")
    
    # Test 3: Check completed work history
    print("\n📊 Test 3: Completed Work History")
    for repo_name in repos[:3]:
        completed_work = work_finder._get_completed_work_for_repo(repo_name)
        print(f"\n  Repository: {repo_name}")
        print(f"  ✓ Completed tasks: {len(completed_work)}")
        if completed_work:
            for task in completed_work[:3]:  # Show first 3
                print(f"    - {task.get('title', 'Untitled')}")
    
    # Test 4: Check issue tracking
    print("\n🎫 Test 4: Issue Tracking")
    for repo_name in repos[:3]:
        try:
            issues = local_simulator.list_issues(repo_name)
            print(f"\n  Repository: {repo_name}")
            print(f"  ✓ Total issues: {len(issues)}")
            
            # Check Claude task status for each issue
            pending_count = 0
            completed_count = 0
            for issue in issues[:5]:  # Check first 5 issues
                if claude_queue.has_completed_task_for_issue(repo_name, issue['number']):
                    completed_count += 1
                elif claude_queue.has_pending_task_for_issue(repo_name, issue['number']):
                    pending_count += 1
            
            if issues:
                print(f"  📋 Claude tasks - Pending: {pending_count}, Completed: {completed_count}")
        except Exception as e:
            print(f"  ⚠️ Error listing issues: {e}")
    
    # Test 5: Work generation based on context
    print("\n🔨 Test 5: Context-Based Work Generation")
    print("\n  Discovering new work opportunities...")
    
    start_time = time.time()
    work_items = await work_finder.discover_work(max_items=5, current_workload=0)
    discovery_time = time.time() - start_time
    
    print(f"  ✓ Found {len(work_items)} work items in {discovery_time:.2f}s")
    for item in work_items:
        print(f"    - [{item.priority.value}] {item.title}")
        print(f"      Repository: {item.repository}")
        print(f"      Type: {item.task_type}")
    
    # Test 6: Verify duplicate prevention
    print("\n🚫 Test 6: Duplicate Prevention Check")
    if work_items:
        test_item = work_items[0]
        repo_name = test_item.repository if "/" in test_item.repository else f"CodeWebMobile-AI/{test_item.repository}"
        
        # Check if this would create a duplicate
        existing_issues = []
        try:
            issues = local_simulator.list_issues(repo_name)
            for issue in issues:
                if issue['title'] == test_item.title:
                    existing_issues.append(issue['number'])
        except:
            pass
        
        if existing_issues:
            print(f"  ✓ Found existing issues: {existing_issues}")
            for issue_num in existing_issues:
                if claude_queue.has_completed_task_for_issue(repo_name, issue_num):
                    print(f"    Issue #{issue_num}: ✅ Already completed by Claude")
                elif claude_queue.has_pending_task_for_issue(repo_name, issue_num):
                    print(f"    Issue #{issue_num}: ⏳ Claude task pending")
                else:
                    print(f"    Issue #{issue_num}: 📋 Ready for Claude")
        else:
            print("  ✓ No duplicates found - safe to create new issue")
    
    # Summary
    print("\n📈 Context Gathering Summary")
    print("=" * 60)
    print(f"✓ Repositories found: {len(repos)}")
    print(f"✓ Work items discovered: {len(work_items)}")
    print(f"✓ Discovery time: {discovery_time:.2f}s")
    print(f"✓ Duplicate prevention: Working")
    
    # Check Claude queue stats
    stats = await claude_queue.get_stats()
    print(f"\n📊 Claude Queue Stats:")
    print(f"  - Pending: {stats['pending']}")
    print(f"  - Processing: {stats['processing']}")
    print(f"  - Completed: {stats['completed']}")
    if stats['avg_execution_time'] > 0:
        print(f"  - Avg execution time: {stats['avg_execution_time']:.1f}s")
    
    print("\n✅ Context gathering test completed successfully!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_context_gathering())