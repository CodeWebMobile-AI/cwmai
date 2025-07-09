#!/usr/bin/env python3
"""
Quick test to check if documentation tasks are being generated
"""
import asyncio
import os
import sys

# Add scripts directory to path
sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')

# Set environment variables
os.environ['STORAGE_MODE'] = 'local'

async def test_work_discovery():
    """Test the work discovery process"""
    print("Testing work discovery for documentation tasks...")
    
    try:
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        from scripts.ai_brain import IntelligentAIBrain
        from scripts.http_ai_client import HTTPAIClient
        
        # Initialize components
        from scripts.state_manager import StateManager
        
        state_manager = StateManager()
        ai_client = HTTPAIClient()
        ai_brain = IntelligentAIBrain(ai_client, state_manager)
        
        # Mock system state with some repositories
        system_state = {
            'projects': {
                'fortressarigo/hydra': {
                    'name': 'hydra',
                    'description': 'Project created from Laravel React starter kit',
                    'full_name': 'fortressarigo/hydra',
                    'created_at': '2024-01-01T00:00:00Z',
                    'updated_at': '2024-12-01T00:00:00Z'
                }
            },
            'repositories': {}
        }
        
        # Create work finder
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # Discover work
        print("\nDiscovering work opportunities...")
        work_items = await work_finder.discover_work(max_items=10)
        
        print(f"\nFound {len(work_items)} work items:")
        
        # Check for documentation tasks
        doc_tasks = []
        setup_tasks = []
        
        for item in work_items:
            print(f"\n- Type: {item.type}")
            print(f"  Title: {item.title}")
            print(f"  Repository: {item.metadata.get('repository', 'N/A')}")
            
            # Check for documentation
            if any(word in str(item).lower() for word in ['documentation', 'readme', 'docs', 'api_documentation', 'developer_guide', 'admin_guide']):
                doc_tasks.append(item)
                
            # Check for setup tasks
            if 'setup' in item.type.lower() or 'initial' in item.title.lower():
                setup_tasks.append(item)
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Documentation tasks found: {len(doc_tasks)}")
        if doc_tasks:
            print("⚠️  WARNING: Documentation tasks still being created!")
            for task in doc_tasks:
                print(f"  - {task.type}: {task.title}")
        else:
            print("✅ SUCCESS: No documentation tasks created!")
            
        print(f"\nSetup tasks found: {len(setup_tasks)}")
        if setup_tasks:
            print("Setup tasks:")
            for task in setup_tasks:
                print(f"  - {task.type}: {task.title}")
                
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_work_discovery())