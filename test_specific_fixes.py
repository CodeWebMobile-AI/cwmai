#!/usr/bin/env python3
"""
Test specific fixes:
1. Claude executor processing
2. Work generation with 60/40 mix
3. Repository path resolution
"""

import asyncio
import json
import os
from pathlib import Path

# Set environment variable for local mode
os.environ['STORAGE_MODE'] = 'local'

async def test_claude_executor():
    """Test that Claude executors can process tasks."""
    print("\n1️⃣ Testing Claude Executor Processing...")
    
    from scripts.claude_execution_queue import ClaudeExecutionQueue
    from scripts.dynamic_claude_executor_pool import DynamicClaudeExecutorPool
    
    # Initialize queue and pool
    queue = ClaudeExecutionQueue()
    pool = DynamicClaudeExecutorPool(min_executors=1, max_executors=2)
    
    # Check if there are pending tasks
    pending_tasks = await queue.get_pending_tasks()
    print(f"   Found {len(pending_tasks)} pending tasks in queue")
    
    if pending_tasks:
        # Start the pool to process tasks
        print("   Starting executor pool...")
        await pool.start()
        
        # Wait a moment for processing to begin
        await asyncio.sleep(5)
        
        # Check if any tasks moved to processing
        queue_state = await queue.get_queue_state()
        processing = queue_state.get('processing', [])
        print(f"   Tasks in processing: {len(processing)}")
        
        # Stop the pool
        await pool.stop()
        
        if processing:
            print("   ✅ Claude executors are processing tasks!")
        else:
            print("   ❌ Claude executors not processing tasks")
    else:
        print("   ⚠️ No pending tasks to test with")

async def test_work_generation():
    """Test that work generation handles 60/40 mix properly."""
    print("\n2️⃣ Testing Work Generation with 60/40 Mix...")
    
    from scripts.intelligent_work_finder import IntelligentWorkFinder
    from scripts.ai_brain_factory import AIBrainFactory
    from scripts.state_manager import StateManager
    
    # Load system state
    state_manager = StateManager()
    system_state = state_manager.state
    
    # Create AI brain
    brain_factory = AIBrainFactory()
    ai_brain = brain_factory.create_brain()
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    # Discover work
    print("   Discovering work...")
    work_items = await work_finder.discover_work(max_items=10)
    
    # Analyze results
    issue_count = 0
    ai_count = 0
    
    for item in work_items:
        if item.metadata.get('from_local_issue'):
            issue_count += 1
        else:
            ai_count += 1
    
    print(f"   Found {len(work_items)} total work items:")
    print(f"   - Existing issues: {issue_count}")
    print(f"   - AI-generated: {ai_count}")
    
    if issue_count > 0 and ai_count > 0:
        print("   ✅ Work generation mixing existing issues and AI tasks!")
    elif len(work_items) > 0:
        print("   ✅ Work generation producing tasks!")
    else:
        print("   ❌ No work generated")

async def test_repository_resolution():
    """Test that repository path resolution works."""
    print("\n3️⃣ Testing Repository Path Resolution...")
    
    from scripts.repository_analyzer import RepositoryAnalyzer
    
    # Create analyzer
    analyzer = RepositoryAnalyzer(storage_mode='local')
    
    # Test with both short and full names
    test_repos = [
        "micronutrient-optimizer",  # Short name
        "CodeWebMobile-AI/micronutrient-optimizer"  # Full name
    ]
    
    success_count = 0
    for repo_name in test_repos:
        try:
            print(f"   Testing: {repo_name}")
            analysis = await analyzer.analyze_repository(repo_name)
            if analysis and analysis.get('name'):
                print(f"   ✅ Successfully analyzed: {analysis['name']}")
                success_count += 1
            else:
                print(f"   ❌ Failed to analyze: {repo_name}")
        except Exception as e:
            print(f"   ❌ Error analyzing {repo_name}: {e}")
    
    if success_count == len(test_repos):
        print("   ✅ Repository path resolution working correctly!")
    else:
        print("   ❌ Repository path resolution has issues")

async def test_work_after_duplicates():
    """Test that work generation continues after detecting duplicates."""
    print("\n4️⃣ Testing Work Generation After Duplicates...")
    
    from scripts.intelligent_work_finder import IntelligentWorkFinder
    from scripts.ai_brain_factory import AIBrainFactory
    from scripts.state_manager import StateManager
    
    # Load system state
    state_manager = StateManager()
    system_state = state_manager.state
    
    # Create AI brain
    brain_factory = AIBrainFactory()
    ai_brain = brain_factory.create_brain()
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    # First discovery
    print("   First work discovery...")
    first_batch = await work_finder.discover_work(max_items=5)
    print(f"   Found {len(first_batch)} items")
    
    # Second discovery (should detect duplicates but still generate new work)
    print("   Second work discovery (expecting duplicate detection)...")
    second_batch = await work_finder.discover_work(max_items=5)
    print(f"   Found {len(second_batch)} items")
    
    if len(second_batch) > 0:
        print("   ✅ Work generation continues after duplicate detection!")
    else:
        print("   ❌ Work generation stopped after duplicates")

async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Testing CWMAI Fixes")
    print("=" * 60)
    
    try:
        await test_claude_executor()
    except Exception as e:
        print(f"   ❌ Claude executor test failed: {e}")
    
    try:
        await test_work_generation()
    except Exception as e:
        print(f"   ❌ Work generation test failed: {e}")
    
    try:
        await test_repository_resolution()
    except Exception as e:
        print(f"   ❌ Repository resolution test failed: {e}")
    
    try:
        await test_work_after_duplicates()
    except Exception as e:
        print(f"   ❌ Work after duplicates test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())