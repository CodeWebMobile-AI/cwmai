#!/usr/bin/env python3
"""
Test the dynamic task loading system
"""
import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

async def test_dynamic_task_loading():
    """Test that dynamic task loading works correctly"""
    print("Testing dynamic task loading system...")
    
    try:
        # Test 1: Load task types from YAML
        print("\n1. Loading task types from YAML...")
        from scripts.dynamic_task_loader import get_dynamic_task_loader
        
        loader = get_dynamic_task_loader()
        all_tasks = loader.get_all_task_names()
        
        print(f"✅ Loaded {len(all_tasks)} task types from YAML")
        print(f"   Sample tasks: {list(all_tasks)[:5]}")
        
        # Test 2: Check metadata loading
        print("\n2. Checking task metadata...")
        task_name = "feature_api_endpoint"
        metadata = loader.get_task_metadata(task_name)
        
        if metadata:
            print(f"✅ Metadata for {task_name}:")
            print(f"   Category: {metadata.category}")
            print(f"   Complexity: {metadata.complexity}")
            print(f"   Duration: {metadata.typical_duration_hours} hours")
        else:
            print(f"❌ Failed to load metadata for {task_name}")
        
        # Test 3: Test dynamic task selector
        print("\n3. Testing dynamic task selector...")
        from scripts.dynamic_task_selector import DynamicTaskTypeSelector
        from scripts.task_types import ArchitectureType
        
        appropriate_tasks = DynamicTaskTypeSelector.get_appropriate_task_types(
            architecture=ArchitectureType.LARAVEL_REACT,
            lifecycle_stage="early_development",
            current_needs=[],
            completed_tasks=set()
        )
        
        print(f"✅ Found {len(appropriate_tasks)} appropriate tasks for early_development")
        print(f"   Tasks: {appropriate_tasks[:5]}")
        
        # Test 4: Test IntelligentWorkFinder integration
        print("\n4. Testing IntelligentWorkFinder integration...")
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        from scripts.ai_brain import IntelligentAIBrain
        from scripts.state_manager import StateManager
        
        state_manager = StateManager()
        state = state_manager.load_state()
        ai_brain = IntelligentAIBrain(state, context={})
        
        # Mock system state
        system_state = {
            'projects': {
                'fortressarigo/test-dynamic': {
                    'name': 'test-dynamic',
                    'description': 'Test project for dynamic tasks',
                    'full_name': 'fortressarigo/test-dynamic',
                    'lifecycle_stage': 'early_development'
                }
            }
        }
        
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # This should use the dynamic task system
        print("   Creating work opportunities...")
        # Just test the task selection part without full discovery
        from scripts.repository_analyzer import RepositoryAnalyzer
        analyzer = RepositoryAnalyzer(ai_brain=ai_brain)
        
        # Test task selection
        completed_tasks = set()
        smart_tasks = DynamicTaskTypeSelector.get_appropriate_task_types(
            architecture=ArchitectureType.LARAVEL_REACT,
            lifecycle_stage="early_development",
            current_needs=[],
            completed_tasks=completed_tasks
        )
        
        if smart_tasks:
            print(f"✅ Dynamic task selection working!")
            print(f"   Selected tasks: {smart_tasks[:3]}")
        else:
            print("❌ No tasks selected")
            
        # Test 5: Verify no hardcoded enums are used
        print("\n5. Verifying dynamic loading (no hardcoded enums)...")
        # The fact that we got string task types instead of enum values proves it's working
        if all(isinstance(task, str) for task in smart_tasks):
            print("✅ SUCCESS: Using dynamic string-based task types!")
        else:
            print("❌ FAIL: Still using enum-based task types")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dynamic_task_loading())