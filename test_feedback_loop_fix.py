#!/usr/bin/env python3
"""
Test the complete feedback loop between Claude execution and issue state updates.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from scripts.local_issue_creator import LocalIssueCreator
from scripts.claude_execution_queue import ClaudeExecutionQueue
from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.ai_brain import IntelligentAIBrain
from scripts.work_item_types import WorkItem, TaskPriority


async def test_feedback_loop():
    """Test the complete feedback loop."""
    print("=" * 60)
    print("Testing Complete Feedback Loop Fix")
    print("=" * 60)
    
    # Initialize components
    issue_creator = LocalIssueCreator(enable_claude_execution=False)
    claude_queue = ClaudeExecutionQueue()
    ai_brain = IntelligentAIBrain()
    work_finder = IntelligentWorkFinder(ai_brain=ai_brain, system_state={})
    
    # Step 1: Create a test work item
    print("\n1. Creating test work item...")
    test_work_item = WorkItem(
        id=f"test_feedback_{int(time.time())}",
        task_type="TESTING",
        title="Test Feedback Loop Implementation",
        description="Test task to verify Claude execution updates issue state correctly",
        priority=TaskPriority.HIGH,
        repository="CodeWebMobile-AI/micronutrient-optimizer",
        estimated_cycles=1,
        metadata={"test": "feedback_loop"}
    )
    
    # Execute work item (creates issue)
    result = await issue_creator.execute_work_item(test_work_item)
    print(f"Issue creation result: {json.dumps(result, indent=2)}")
    
    if not result.get('success'):
        print("❌ Failed to create issue")
        return
    
    issue_number = result.get('issue_number')
    print(f"✅ Created issue #{issue_number}")
    
    # Step 2: Check that work finder sees the issue
    print("\n2. Checking work discovery...")
    discovered_work = await work_finder.discover_work(max_items=5)
    
    found_issue = False
    for work in discovered_work:
        if work.metadata.get('issue_number') == issue_number:
            found_issue = True
            print(f"✅ Work finder discovered issue #{issue_number}")
            break
    
    if not found_issue:
        print("❌ Work finder did not discover the created issue")
    
    # Step 3: Simulate Claude completing the task
    print("\n3. Simulating Claude task completion...")
    
    # Get the Claude task from the queue
    pending_tasks = claude_queue._queue
    claude_task = None
    
    for task in pending_tasks:
        if task.issue_number == issue_number:
            claude_task = task
            break
    
    if claude_task:
        print(f"Found Claude task: {claude_task.id}")
        
        # Simulate claiming and completing the task
        claimed_task = await claude_queue.claim_task("test_executor")
        if claimed_task and claimed_task.id == claude_task.id:
            print("✅ Claimed Claude task")
            
            # Simulate successful execution
            await claude_queue.complete_task(
                claimed_task.id,
                {
                    'success': True,
                    'execution_time': 5.0,
                    'files_changed': 3,
                    'output': 'Test completion'
                }
            )
            print("✅ Marked Claude task as completed")
            
            # Give it a moment to update
            await asyncio.sleep(1)
            
            # Step 4: Check that issue now has claude-implemented label
            print("\n4. Checking issue state after completion...")
            from scripts.local_github_simulator import LocalGitHubSimulator
            simulator = LocalGitHubSimulator()
            
            issues = simulator.get_issues("CodeWebMobile-AI/micronutrient-optimizer", state="all")
            for issue in issues:
                if issue.number == issue_number:
                    if 'claude-implemented' in issue.labels:
                        print(f"✅ Issue #{issue_number} has claude-implemented label!")
                    else:
                        print(f"❌ Issue #{issue_number} is missing claude-implemented label")
                        print(f"   Current labels: {issue.labels}")
                    break
            
            # Step 5: Check that work finder now skips this issue
            print("\n5. Checking work discovery after completion...")
            discovered_work_2 = await work_finder.discover_work(max_items=5)
            
            found_completed_issue = False
            for work in discovered_work_2:
                if work.metadata.get('issue_number') == issue_number:
                    found_completed_issue = True
                    print(f"❌ Work finder still returns completed issue #{issue_number}")
                    break
            
            if not found_completed_issue:
                print(f"✅ Work finder correctly skips completed issue #{issue_number}")
                
                # Check if AI-generated work is included
                ai_generated_count = sum(1 for w in discovered_work_2 
                                       if not w.metadata.get('from_local_issue'))
                if ai_generated_count > 0:
                    print(f"✅ Work finder generated {ai_generated_count} AI tasks to fill slots")
                else:
                    print("⚠️ No AI-generated work found (might be normal if many issues exist)")
        else:
            print("❌ Failed to claim Claude task")
    else:
        print("❌ No Claude task found in queue")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_feedback_loop())