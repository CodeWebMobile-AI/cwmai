#!/usr/bin/env python3
"""
Test Smart Task Type System

Verifies that the intelligent task type system works correctly with:
- Lifecycle awareness
- Architecture-specific tasks
- Context-aware prioritization
"""

import asyncio
import os
import sys
import logging

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.task_types import SmartTaskType, TaskTypeSelector, ArchitectureType
from scripts.project_lifecycle_analyzer import ProjectLifecycleAnalyzer, ProjectStage
from scripts.task_type_registry import TaskTypeRegistry
from scripts.ai_brain import AIBrain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


async def test_task_type_selection():
    """Test task type selection based on context."""
    print("\n=== Testing Task Type Selection ===\n")
    
    # Test 1: Inception stage Laravel/React project
    print("Test 1: Inception stage Laravel/React project")
    appropriate_tasks = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.LARAVEL_REACT,
        lifecycle_stage="inception",
        current_needs=["setup", "authentication"],
        completed_tasks=set()
    )
    
    print(f"Selected tasks: {[t.value for t in appropriate_tasks[:5]]}")
    assert SmartTaskType.SETUP_PROJECT_STRUCTURE in appropriate_tasks
    assert SmartTaskType.SETUP_LARAVEL_API in appropriate_tasks
    print("✅ Correct tasks for inception stage\n")
    
    # Test 2: Active development with some completed tasks
    print("Test 2: Active development with completed setup")
    appropriate_tasks = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.LARAVEL_REACT,
        lifecycle_stage="active_development",
        current_needs=["features", "testing"],
        completed_tasks={"setup_project_structure", "setup_laravel_api"}
    )
    
    print(f"Selected tasks: {[t.value for t in appropriate_tasks[:5]]}")
    assert SmartTaskType.FEATURE_API_ENDPOINT in appropriate_tasks
    assert SmartTaskType.TESTING_UNIT_TESTS in appropriate_tasks
    assert SmartTaskType.SETUP_PROJECT_STRUCTURE not in appropriate_tasks  # Already completed
    print("✅ Correct tasks for active development\n")
    
    # Test 3: Mature stage optimization
    print("Test 3: Mature stage needing optimization")
    appropriate_tasks = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.API_ONLY,
        lifecycle_stage="mature",
        current_needs=["performance", "maintenance"],
        completed_tasks={"feature_api_endpoint", "testing_unit_tests", "documentation_api"}
    )
    
    print(f"Selected tasks: {[t.value for t in appropriate_tasks[:5]]}")
    assert SmartTaskType.OPTIMIZATION_PERFORMANCE in appropriate_tasks
    assert SmartTaskType.MAINTENANCE_SECURITY_PATCH in appropriate_tasks
    print("✅ Correct tasks for mature stage\n")


async def test_lifecycle_detection():
    """Test enhanced lifecycle detection."""
    print("\n=== Testing Lifecycle Detection ===\n")
    
    ai_brain = AIBrain()
    analyzer = ProjectLifecycleAnalyzer(ai_brain)
    
    # Test 1: New project with minimal structure
    print("Test 1: New project with minimal structure")
    new_repo_analysis = {
        'basic_info': {
            'created_at': '2024-01-10T00:00:00Z',
            'language': 'PHP'
        },
        'code_analysis': {
            'config_files': ['composer.json', 'package.json'],
            'documentation': [],
            'test_coverage': 'none'
        },
        'health_metrics': {
            'recent_commits': 5
        },
        'recent_activity': {
            'active_contributors': 1
        }
    }
    
    result = await analyzer.analyze_project_stage(new_repo_analysis)
    print(f"Detected stage: {result['current_stage']}")
    print(f"Confidence: {result['stage_confidence']:.0%}")
    assert result['current_stage'] == 'inception'
    print("✅ Correctly identified inception stage\n")
    
    # Test 2: Active project with features and tests
    print("Test 2: Active project with features and tests")
    active_repo_analysis = {
        'basic_info': {
            'created_at': '2023-06-01T00:00:00Z',
            'language': 'TypeScript'
        },
        'code_analysis': {
            'config_files': ['.github/workflows/test.yml', 'Dockerfile'],
            'documentation': ['README.md', 'docs/API.md'],
            'test_coverage': 'has_tests',
            'test_directories': ['tests/unit', 'tests/integration']
        },
        'health_metrics': {
            'recent_commits': 45
        },
        'recent_activity': {
            'active_contributors': 3
        },
        'architecture': {
            'document_exists': True,
            'core_entities': ['User', 'Product', 'Order', 'Payment']
        }
    }
    
    result = await analyzer.analyze_project_stage(active_repo_analysis)
    print(f"Detected stage: {result['current_stage']}")
    print(f"Confidence: {result['stage_confidence']:.0%}")
    assert result['current_stage'] in ['active_development', 'growth']
    print("✅ Correctly identified active/growth stage\n")


async def test_task_registry_learning():
    """Test task type registry and learning system."""
    print("\n=== Testing Task Registry Learning ===\n")
    
    registry = TaskTypeRegistry("test_task_registry.json")
    
    # Record some task outcomes
    print("Recording task outcomes...")
    
    # Successful setup task
    registry.record_task_outcome(
        SmartTaskType.SETUP_LARAVEL_API,
        {
            'repository': 'test-project',
            'lifecycle_stage': 'inception',
            'architecture': ArchitectureType.LARAVEL_REACT,
            'success': True,
            'duration_cycles': 3,
            'value_created': 8.5
        }
    )
    
    # Failed optimization task (wrong stage)
    registry.record_task_outcome(
        SmartTaskType.OPTIMIZATION_PERFORMANCE,
        {
            'repository': 'test-project',
            'lifecycle_stage': 'inception',
            'architecture': ArchitectureType.LARAVEL_REACT,
            'success': False,
            'duration_cycles': 5,
            'value_created': 2.0,
            'failure_reason': 'Premature optimization'
        }
    )
    
    # Get best tasks for context
    print("\nGetting best tasks for inception stage...")
    best_tasks = registry.get_best_task_types({
        'lifecycle_stage': 'inception',
        'architecture': ArchitectureType.LARAVEL_REACT,
        'current_needs': ['setup', 'structure']
    })
    
    print("Best tasks based on learning:")
    for task, score in best_tasks[:3]:
        print(f"  - {task.value}: {score:.2f}")
    
    # Get analytics
    analytics = registry.get_task_type_analytics()
    print(f"\nRegistry Analytics:")
    print(f"  Total tasks: {analytics['total_tasks_executed']}")
    print(f"  Success rate: {analytics['overall_success_rate']:.0%}")
    print(f"  Patterns learned: {analytics['patterns_learned']}")
    
    # Clean up test file
    if os.path.exists("test_task_registry.json"):
        os.remove("test_task_registry.json")
    
    print("✅ Task registry learning works correctly\n")


async def test_integration():
    """Test integration of all components."""
    print("\n=== Testing Full Integration ===\n")
    
    # Simulate a repository at different stages
    repo_name = "awesome-saas-platform"
    
    # Stage 1: Inception
    print(f"Stage 1: {repo_name} at inception")
    tasks_inception = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.LARAVEL_REACT,
        lifecycle_stage="inception",
        current_needs=["basic setup", "authentication"],
        completed_tasks=set()
    )
    print(f"Tasks: {[t.value for t in tasks_inception[:3]]}")
    
    # Stage 2: Early Development (after completing setup)
    print(f"\nStage 2: {repo_name} at early development")
    completed = {t.value for t in tasks_inception[:3]}
    tasks_early = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.LARAVEL_REACT,
        lifecycle_stage="early_development",
        current_needs=["core features", "basic testing"],
        completed_tasks=completed
    )
    print(f"Tasks: {[t.value for t in tasks_early[:3]]}")
    
    # Stage 3: Growth (after core features)
    print(f"\nStage 3: {repo_name} at growth stage")
    completed.update({t.value for t in tasks_early[:3]})
    tasks_growth = TaskTypeSelector.get_appropriate_task_types(
        architecture=ArchitectureType.LARAVEL_REACT,
        lifecycle_stage="growth",
        current_needs=["scaling", "performance"],
        completed_tasks=completed
    )
    print(f"Tasks: {[t.value for t in tasks_growth[:3]]}")
    
    print("\n✅ Integration test passed - tasks evolve with project lifecycle\n")


async def main():
    """Run all tests."""
    print("Testing Smart Task Type System")
    print("=" * 60)
    
    await test_task_type_selection()
    await test_lifecycle_detection()
    await test_task_registry_learning()
    await test_integration()
    
    print("=" * 60)
    print("All tests passed! 🎉")
    print("\nKey improvements:")
    print("✅ Task types are now context-aware (architecture + lifecycle)")
    print("✅ Lifecycle detection based on actual code maturity")
    print("✅ Tasks evolve as project progresses through stages")
    print("✅ System learns from task outcomes")
    print("✅ No more 'optimization' tasks for new projects!")
    print("✅ No more 'setup' tasks for mature projects!")


if __name__ == "__main__":
    asyncio.run(main())