#!/usr/bin/env python3
"""
Test script to monitor system for 3 minutes
"""
import asyncio
import time
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

from scripts.continuous_orchestrator import ContinuousOrchestrator

async def monitor_system_3min():
    print(f'Starting 3-minute system test at {datetime.now()}')
    print('=' * 80)
    
    orchestrator = ContinuousOrchestrator()
    start_time = time.time()
    test_duration = 180  # 3 minutes
    
    # Track discovered tasks
    documentation_tasks = []
    setup_issues = []
    research_tasks = []
    all_tasks = []
    
    print('Monitoring for:')
    print('- Documentation tasks (should be NONE)')
    print('- Initial Project Setup issues (should be created for new projects)')
    print('- Non-project research tasks (should be NONE)')
    print('=' * 80)
    
    orchestrator_task = None
    try:
        # Start the orchestrator
        orchestrator_task = asyncio.create_task(orchestrator.start())
        
        while time.time() - start_time < test_duration:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Read current state
            try:
                with open('continuous_orchestrator_state.json', 'r') as f:
                    state = json.load(f)
                
                # Check work items
                if 'active_work_items' in state:
                    for item in state['active_work_items']:
                        task_type = item.get('task_type', '')
                        title = item.get('title', '')
                        
                        # Track all tasks
                        task_info = {
                            'time': time.time() - start_time,
                            'type': task_type,
                            'title': title
                        }
                        
                        # Don't count duplicates
                        if not any(t['title'] == title for t in all_tasks):
                            all_tasks.append(task_info)
                        
                        # Check for documentation tasks
                        if 'documentation' in task_type.lower() or 'documentation' in title.lower():
                            if not any(t.get('title') == title for t in documentation_tasks):
                                documentation_tasks.append(item)
                                print(f'\n⚠️  FOUND DOCUMENTATION TASK: {title}')
                        
                        # Check for Initial Project Setup
                        if 'Initial Project Setup' in title:
                            if not any(t.get('title') == title for t in setup_issues):
                                setup_issues.append(item)
                                print(f'\n✅ Found Initial Project Setup: {title}')
                        
                        # Check for research tasks
                        if 'research' in task_type.lower():
                            # Check if it's project-related
                            if not any(word in title.lower() for word in ['project', 'repository', 'codebase', 'code', 'implementation']):
                                if not any(t.get('title') == title for t in research_tasks):
                                    research_tasks.append(item)
                                    print(f'\n⚠️  FOUND NON-PROJECT RESEARCH: {title}')
                
                elapsed = int(time.time() - start_time)
                active_count = len(state.get("active_work_items", []))
                print(f'\r[{elapsed:3d}s] Active work items: {active_count} | Total unique tasks: {len(all_tasks)}', end='', flush=True)
                
            except Exception as e:
                pass
        
        print('\n\n' + '=' * 80)
        print('TEST RESULTS:')
        print('=' * 80)
        
        print(f'\n1. Documentation Tasks: {len(documentation_tasks)}')
        if documentation_tasks:
            print('   ❌ FAIL: Found documentation tasks:')
            for task in documentation_tasks[:5]:
                print(f'      - {task.get("title")}')
        else:
            print('   ✅ PASS: No documentation tasks generated')
        
        print(f'\n2. Initial Project Setup Issues: {len(setup_issues)}')
        if setup_issues:
            print('   ✅ Found setup issues:')
            for issue in setup_issues[:5]:
                print(f'      - {issue.get("title")}')
        else:
            print('   ℹ️  No setup issues (might not have found projects without issues)')
        
        print(f'\n3. Non-Project Research Tasks: {len(research_tasks)}')
        if research_tasks:
            print('   ❌ FAIL: Found non-project research tasks:')
            for task in research_tasks[:5]:
                print(f'      - {task.get("title")}')
        else:
            print('   ✅ PASS: No non-project research tasks')
        
        print(f'\n4. Total Unique Tasks Discovered: {len(all_tasks)}')
        if all_tasks:
            # Show task type distribution
            task_types = {}
            for task in all_tasks:
                t = task['type']
                task_types[t] = task_types.get(t, 0) + 1
            
            print('   Task Type Distribution:')
            for t, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
                print(f'      - {t}: {count}')
            
            # Show some sample tasks
            print('\n   Sample Tasks:')
            for task in all_tasks[:5]:
                print(f'      - [{task["type"]}] {task["title"]}')
        
    finally:
        if orchestrator_task:
            orchestrator_task.cancel()
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
        await orchestrator.stop()
        
    print('\n' + '=' * 80)
    print('Test completed successfully')

if __name__ == "__main__":
    asyncio.run(monitor_system_3min())