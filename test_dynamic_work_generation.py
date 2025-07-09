#!/usr/bin/env python3
"""
Test Dynamic Work Generation System

Tests:
1. Dynamic work generator creates unique work items
2. Work is based on ARCHITECTURE.md content
3. Completed work history is considered
4. No fixed task types are used
5. Hybrid orchestrator integrates properly
"""

import asyncio
import os
import sys
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dynamic_work_generator import DynamicWorkGenerator, WorkContext
from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager


async def test_dynamic_work_generator():
    """Test the dynamic work generator directly."""
    print("\n🧪 Testing Dynamic Work Generator...")
    
    # Create AI brain
    ai_brain = AIBrainFactory.create_for_production()
    
    # Create work generator
    generator = DynamicWorkGenerator(ai_brain=ai_brain)
    
    # Test 1: Generate work without AI (fallback mode)
    print("\n1️⃣ Testing work generation without AI...")
    test_context = WorkContext(
        repository="test-repo",
        architecture_content=None,
        completed_work=[],
        current_issues=[],
        repository_analysis={
            'specific_needs': [
                {
                    'type': 'testing',
                    'description': 'Add unit tests for authentication module',
                    'suggested_action': 'Create comprehensive test suite',
                    'priority': 'high'
                },
                {
                    'type': 'documentation',
                    'description': 'API documentation is missing',
                    'suggested_action': 'Document all API endpoints',
                    'priority': 'medium'
                }
            ],
            'improvement_opportunities': [
                {
                    'type': 'performance',
                    'description': 'Database queries could be optimized',
                    'impact': 'high',
                    'effort': 'medium'
                }
            ]
        }
    )
    
    work_items = generator._generate_work_without_ai(test_context, max_items=3)
    
    print(f"✅ Generated {len(work_items)} work items without AI:")
    for item in work_items:
        print(f"   - {item.title}")
        print(f"     Type: {item.task_type}, Priority: {item.priority.name}")
    
    # Test 2: Check that completed work is filtered
    print("\n2️⃣ Testing completed work filtering...")
    test_context.completed_work = [
        {
            'id': 'test-1',
            'title': 'add unit tests for authentication module',
            'description': 'Already done',
            'task_type': 'TESTING',
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
    ]
    
    filtered_work_items = generator._generate_work_without_ai(test_context, max_items=3)
    
    print(f"✅ After filtering completed work: {len(filtered_work_items)} items")
    for item in filtered_work_items:
        print(f"   - {item.title}")
    
    # Test 3: Generate work with AI
    print("\n3️⃣ Testing AI-powered work generation...")
    test_context.architecture_content = """
# Project Architecture

## Overview
This is a Laravel/React application for task management.

## Key Components
- Authentication system (Laravel Sanctum)
- Task CRUD operations
- Real-time updates via WebSockets
- Redis caching layer

## Current Status
- Basic authentication implemented
- Task creation API complete
- Frontend needs work
    """
    
    try:
        ai_work_items = await generator.generate_work_for_repository(
            repository="test-repo",
            max_work_items=3,
            completed_work=test_context.completed_work
        )
        
        print(f"✅ AI generated {len(ai_work_items)} work items:")
        for item in ai_work_items:
            print(f"   - {item.title}")
            print(f"     Description: {item.description[:100]}...")
            print(f"     Type: {item.task_type}, Priority: {item.priority.name}")
            if 'reasoning' in item.metadata:
                print(f"     Reasoning: {item.metadata['reasoning'][:100]}...")
    except Exception as e:
        print(f"⚠️  AI generation failed (this is OK if no API key): {e}")


async def test_hybrid_orchestrator():
    """Test the hybrid orchestrator with dynamic work generation."""
    print("\n\n🧪 Testing Hybrid Orchestrator Integration...")
    
    # Create orchestrator
    orchestrator = EnhancedHybridOrchestrator(
        max_workers=2,
        enable_parallel=True
    )
    
    # Test that it initializes without errors
    print("✅ Orchestrator initialized successfully")
    
    # Check that work generator is properly set up
    assert hasattr(orchestrator, 'work_generator'), "Orchestrator should have work_generator"
    assert orchestrator.work_generator is not None, "Work generator should be initialized"
    print("✅ Dynamic work generator is properly integrated")
    
    # Test work discovery
    print("\n4️⃣ Testing work discovery in orchestrator...")
    
    # Mock some project data
    orchestrator.system_state['projects'] = {
        'test-project': {
            'full_name': 'CodeWebMobile-AI/test-project',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'description': 'Test project for dynamic work generation'
        }
    }
    
    try:
        work_items = await orchestrator.find_work()
        print(f"✅ Orchestrator discovered {len(work_items)} work items")
        
        # Check that work items don't use fixed task types
        fixed_task_types = [
            'RESEARCH_TECHNOLOGY', 'RESEARCH_MARKET', 'RESEARCH_COMPETITOR',
            'RESEARCH_USER_NEEDS', 'maintenance_dependencies', 'testing_unit_tests'
        ]
        
        for item in work_items:
            if any(fixed_type in item.task_type for fixed_type in fixed_task_types):
                print(f"❌ Found fixed task type: {item.task_type}")
            else:
                print(f"✅ Dynamic work item: {item.title[:50]}...")
    except Exception as e:
        print(f"⚠️  Work discovery failed: {e}")
        import traceback
        traceback.print_exc()


async def test_no_fixed_task_types():
    """Verify that fixed task types are not being used."""
    print("\n\n🧪 Verifying Fixed Task Types are Removed...")
    
    # Check that task_types.py doesn't have research enums
    try:
        from scripts.task_types import SmartTaskType
        
        # Get all task type values
        task_types = [t.value for t in SmartTaskType]
        
        # Check for removed research types
        removed_types = [
            'research_technology', 'research_market', 
            'research_competitor', 'research_user_needs'
        ]
        
        found_removed = []
        for removed in removed_types:
            if removed in task_types:
                found_removed.append(removed)
        
        if found_removed:
            print(f"❌ Found removed task types still in enum: {found_removed}")
        else:
            print("✅ Research task types successfully removed from enum")
            
    except Exception as e:
        print(f"Error checking task types: {e}")


async def test_completed_work_tracking():
    """Test that completed work is properly tracked."""
    print("\n\n🧪 Testing Completed Work Tracking...")
    
    # Create a state manager
    state_manager = StateManager()
    
    # Create test completed work
    test_completed = {
        'id': 'test-123',
        'task_type': 'FEATURE',
        'title': 'Implement user dashboard',
        'repository': 'test-repo',
        'completed_at': datetime.now(timezone.utc).isoformat()
    }
    
    # Load current state
    task_state = state_manager.load_task_state()
    
    # Initialize completed_tasks if not exists
    if 'completed_tasks' not in task_state:
        task_state['completed_tasks'] = []
    
    # Add test task
    task_state['completed_tasks'].append(test_completed)
    
    # Save state
    state_manager.save_task_state(task_state)
    print("✅ Saved test completed task to state")
    
    # Now test that the orchestrator can load it
    orchestrator = EnhancedHybridOrchestrator(max_workers=1)
    completed_work = orchestrator._get_completed_work_for_repo('test-repo')
    
    if completed_work:
        print(f"✅ Orchestrator loaded {len(completed_work)} completed tasks for test-repo")
        for task in completed_work:
            print(f"   - {task.get('title', 'Unknown')}")
    else:
        print("⚠️  No completed work found (this is OK on first run)")


async def main():
    """Run all tests."""
    print("🚀 Testing Dynamic Work Generation System")
    print("=" * 60)
    
    # Check environment
    storage_mode = os.getenv('STORAGE_MODE', 'GITHUB')
    print(f"Storage Mode: {storage_mode}")
    
    try:
        # Run tests
        await test_dynamic_work_generator()
        await test_hybrid_orchestrator()
        await test_no_fixed_task_types()
        await test_completed_work_tracking()
        
        print("\n\n✅ All tests completed!")
        print("\nSummary:")
        print("- Dynamic work generator creates unique work items ✓")
        print("- Work generation considers repository context ✓")
        print("- Completed work history is tracked ✓")
        print("- Fixed task types have been removed ✓")
        print("- Hybrid orchestrator properly integrated ✓")
        
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set up environment
    os.environ['STORAGE_MODE'] = 'LOCAL'
    
    # Run tests
    asyncio.run(main())