#!/usr/bin/env python3
"""
Test that setup task types have been removed and Initial Project Setup still works
"""
import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

async def test_no_setup_tasks():
    """Test that individual setup tasks are gone"""
    print("Testing removal of individual setup task types...")
    
    try:
        from scripts.task_types import SmartTaskType, TaskTypeSelector
        
        # Check that setup task types no longer exist
        print("\n1. Checking SmartTaskType enum...")
        setup_types_found = []
        for task_type in SmartTaskType:
            if 'setup' in task_type.value.lower():
                setup_types_found.append(task_type.value)
        
        if setup_types_found:
            print(f"❌ FAIL: Found setup task types: {setup_types_found}")
        else:
            print("✅ SUCCESS: No setup task types found in SmartTaskType enum")
        
        # Test that inception stage projects don't get setup tasks
        print("\n2. Testing task selection for inception stage...")
        appropriate_tasks = TaskTypeSelector.get_appropriate_task_types(
            architecture=None,
            lifecycle_stage="inception",
            current_needs=[],
            completed_tasks=set()
        )
        
        setup_tasks = [t for t in appropriate_tasks if 'setup' in t.value.lower()]
        
        if setup_tasks:
            print(f"❌ FAIL: Setup tasks suggested for inception: {[t.value for t in setup_tasks]}")
        else:
            print("✅ SUCCESS: No setup tasks suggested for inception stage")
        
        print(f"   Available tasks for inception: {[t.value for t in appropriate_tasks[:3]]}")
        
        # Test Initial Project Setup issue creation still works
        print("\n3. Testing Initial Project Setup issue creation...")
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        from scripts.ai_brain import IntelligentAIBrain
        from scripts.state_manager import StateManager
        from scripts.local_github_simulator import LocalGitHubSimulator
        
        state_manager = StateManager()
        state = state_manager.load_state()
        ai_brain = IntelligentAIBrain(state, context={})
        
        # Create test project
        simulator = LocalGitHubSimulator()
        test_repo = 'fortressarigo/test-setup-removed'
        try:
            simulator.create_repository(test_repo)
        except:
            pass  # Already exists
        
        system_state = {
            'projects': {
                test_repo: {
                    'name': 'test-setup-removed',
                    'description': 'Test project',
                    'full_name': test_repo,
                    'lifecycle_stage': 'inception'
                }
            }
        }
        
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # This should create Initial Project Setup issue
        await work_finder.discover_work(max_items=5)
        
        # Check if issue was created
        issues = simulator.get_issues(test_repo)
        setup_issue = None
        for issue in issues:
            if "Initial Project Setup" in issue.title:
                setup_issue = issue
                break
        
        if setup_issue:
            print("✅ SUCCESS: Initial Project Setup issue still being created")
            print(f"   Issue: #{setup_issue.number} - {setup_issue.title}")
        else:
            print("❌ FAIL: Initial Project Setup issue not created")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_no_setup_tasks())