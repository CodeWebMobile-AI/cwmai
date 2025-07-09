#!/usr/bin/env python3
"""
Check if initial project setup tasks are being created
"""
import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

async def check_setup_tasks():
    """Check for initial project setup task generation"""
    print("Checking for initial project setup task generation...")
    
    try:
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        from scripts.ai_brain import IntelligentAIBrain
        from scripts.state_manager import StateManager
        
        # Initialize state manager
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # Mock system state with different project scenarios
        system_state = {
            'projects': {
                # New project (just created)
                'fortressarigo/new-project': {
                    'name': 'new-project',
                    'description': 'Brand new project',
                    'full_name': 'fortressarigo/new-project',
                    'created_at': '2025-01-15T00:00:00Z',
                    'updated_at': '2025-01-15T00:00:00Z',
                    'lifecycle_stage': 'inception'
                },
                # Existing project without setup
                'fortressarigo/existing-no-setup': {
                    'name': 'existing-no-setup', 
                    'description': 'Project that needs initial setup',
                    'full_name': 'fortressarigo/existing-no-setup',
                    'created_at': '2024-06-01T00:00:00Z',
                    'updated_at': '2024-12-01T00:00:00Z',
                    'lifecycle_stage': 'early_development'
                },
                # Mature project
                'fortressarigo/mature-project': {
                    'name': 'mature-project',
                    'description': 'Well-established project',
                    'full_name': 'fortressarigo/mature-project', 
                    'created_at': '2023-01-01T00:00:00Z',
                    'updated_at': '2025-01-14T00:00:00Z',
                    'lifecycle_stage': 'mature'
                }
            },
            'repositories': {}
        }
        
        # Initialize AI brain with state
        ai_brain = IntelligentAIBrain(state, context={})
        
        # Create work finder
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # Discover work
        print("\nDiscovering work opportunities...")
        work_items = await work_finder.discover_work(max_items=20)
        
        print(f"\nFound {len(work_items)} total work items")
        
        # Check for setup tasks
        setup_tasks = []
        doc_tasks = []
        
        for item in work_items:
            # Check for setup tasks
            task_type = item.task_type if hasattr(item, 'task_type') else ''
            title = item.title if hasattr(item, 'title') else ''
            
            if any(word in task_type.lower() for word in ['setup', 'initial']) or \
               any(word in title.lower() for word in ['setup', 'initial']):
                setup_tasks.append(item)
                
            # Also check for any documentation tasks
            if any(word in str(item).lower() for word in ['documentation', 'readme', 'docs']):
                doc_tasks.append(item)
        
        print("\n" + "=" * 60)
        print("SETUP TASK ANALYSIS:")
        print("=" * 60)
        
        print(f"\nSetup tasks found: {len(setup_tasks)}")
        if setup_tasks:
            print("\nSetup tasks by project:")
            for task in setup_tasks:
                repo = task.metadata.get('repository', task.repository) if hasattr(task, 'metadata') else task.repository if hasattr(task, 'repository') else 'unknown'
                print(f"\n  Project: {repo}")
                print(f"  Task Type: {task.task_type if hasattr(task, 'task_type') else 'N/A'}")
                print(f"  Title: {task.title if hasattr(task, 'title') else 'N/A'}")
                print(f"  Priority: {task.priority if hasattr(task, 'priority') else 'N/A'}")
                print(f"  Stage: {task.metadata.get('lifecycle_stage', 'unknown') if hasattr(task, 'metadata') else 'unknown'}")
        else:
            print("\n⚠️  No setup tasks found!")
            
        # Check which projects got setup tasks
        projects_with_setup = set()
        for task in setup_tasks:
            repo = task.metadata.get('repository', task.repository)
            if repo:
                projects_with_setup.add(repo)
                
        print(f"\nProjects with setup tasks: {len(projects_with_setup)}")
        for proj in projects_with_setup:
            print(f"  - {proj}")
            
        print("\n" + "=" * 60)
        print("DOCUMENTATION CHECK:")
        print("=" * 60)
        print(f"Documentation tasks found: {len(doc_tasks)}")
        if doc_tasks:
            print("⚠️  WARNING: Documentation tasks still being created!")
        else:
            print("✅ SUCCESS: No documentation tasks created!")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_setup_tasks())