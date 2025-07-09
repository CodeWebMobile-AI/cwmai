#!/usr/bin/env python3
"""
Test work discovery directly
"""
import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager

async def test_work_discovery():
    print("Testing work discovery system...")
    print("=" * 80)
    
    # Initialize components
    state_manager = StateManager()
    state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(state, context={})
    
    # Load system state
    system_state = state_manager.load_state()
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    # Print current projects
    projects = system_state.get('projects', {})
    repositories = system_state.get('repositories', {})
    all_projects = {**projects, **repositories}
    
    print(f"\nFound {len(all_projects)} projects in system state:")
    for name, data in all_projects.items():
        print(f"  - {name}")
    
    # Discover work
    print("\n" + "=" * 80)
    print("Discovering work opportunities...")
    print("=" * 80)
    
    work_items = await work_finder.discover_work(max_items=10)
    
    print(f"\nDiscovered {len(work_items)} work items:")
    for i, item in enumerate(work_items, 1):
        print(f"\n{i}. {item.title}")
        print(f"   Type: {item.task_type}")
        print(f"   Repository: {item.repository}")
        print(f"   Priority: {item.priority}")
        if hasattr(item, 'description'):
            print(f"   Description: {item.description[:100]}...")
    
    # Check for specific task types
    print("\n" + "=" * 80)
    print("Task Type Analysis:")
    print("=" * 80)
    
    documentation_tasks = [w for w in work_items if 'documentation' in w.task_type.lower() or 'documentation' in w.title.lower()]
    setup_tasks = [w for w in work_items if 'Initial Project Setup' in w.title]
    research_tasks = [w for w in work_items if 'research' in w.task_type.lower()]
    
    print(f"\nDocumentation tasks: {len(documentation_tasks)}")
    if documentation_tasks:
        print("  ❌ FOUND DOCUMENTATION TASKS:")
        for task in documentation_tasks:
            print(f"    - {task.title}")
    else:
        print("  ✅ No documentation tasks")
    
    print(f"\nInitial Project Setup issues: {len(setup_tasks)}")
    if setup_tasks:
        print("  ✅ FOUND SETUP ISSUES:")
        for task in setup_tasks:
            print(f"    - {task.title}")
    else:
        print("  ℹ️  No setup issues")
    
    print(f"\nResearch tasks: {len(research_tasks)}")
    if research_tasks:
        print("  Tasks found:")
        for task in research_tasks:
            print(f"    - {task.title}")
            # Check if it's project-related
            if not any(word in task.title.lower() for word in ['project', 'repository', 'codebase', 'code', 'implementation']):
                print("      ⚠️  NON-PROJECT RESEARCH TASK")

if __name__ == "__main__":
    asyncio.run(test_work_discovery())