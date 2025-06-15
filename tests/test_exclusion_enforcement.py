#!/usr/bin/env python3
"""
Test that repository exclusion is properly enforced across the system.
Validates all the fixes for worker specialization and resource efficiency.
"""

import json
import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.repository_exclusion import RepositoryExclusion
from scripts.state_manager import StateManager
from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.resource_manager import ResourceManager
from scripts.task_persistence import TaskPersistence
from scripts.work_item_types import WorkItem, TaskPriority


def test_state_exclusion():
    """Test that excluded repos don't appear in state."""
    print("\n🧪 Testing state exclusion filtering...")
    
    # Create test state with contamination
    test_state = {
        'projects': {
            '.github': {'name': '.github', 'description': 'Organization config'},
            'cwmai': {'name': 'cwmai', 'description': 'AI system'},
            'ai-creative-studio': {'name': 'ai-creative-studio', 'description': 'Creative AI'},
            'moderncms-with-ai-powered-content-recommendations': {'name': 'moderncms', 'description': 'CMS'}
        }
    }
    
    # Apply filtering
    filtered = RepositoryExclusion.filter_excluded_repos_dict(test_state['projects'])
    
    assert '.github' not in filtered, ".github should be excluded"
    assert 'cwmai' not in filtered, "cwmai should be excluded"
    assert 'ai-creative-studio' in filtered, "ai-creative-studio should be included"
    assert 'moderncms-with-ai-powered-content-recommendations' in filtered, "moderncms should be included"
    
    print(f"✅ State filtering test passed - {len(filtered)} valid projects from {len(test_state['projects'])} total")


def test_worker_specialization():
    """Test that workers don't get assigned to excluded repos."""
    print("\n🧪 Testing worker specialization assignment...")
    
    # Mock state with excluded repos
    mock_state = {
        'projects': {
            '.github': {},
            'cwmai': {},
            'ai-creative-studio': {},
            'moderncms-with-ai-powered-content-recommendations': {}
        }
    }
    
    # Simulate worker assignment
    projects = list(mock_state['projects'].keys())
    filtered_projects = RepositoryExclusion.filter_excluded_repos(projects)
    
    # Check first non-system worker gets valid project
    assert len(filtered_projects) == 2, f"Expected 2 valid projects, got {len(filtered_projects)}"
    assert filtered_projects[0] not in ['.github', 'cwmai'], f"Worker assigned to excluded repo: {filtered_projects[0]}"
    
    print(f"✅ Worker specialization test passed - workers assigned to: {filtered_projects}")


def test_state_manager_filtering():
    """Test that StateManager filters excluded repos on load."""
    print("\n🧪 Testing StateManager filtering on load...")
    
    # Create a temporary state file with contamination
    test_state_file = "test_state_temp.json"
    contaminated_state = {
        'version': '1.0.0',
        'projects': {
            '.github': {'name': '.github'},
            'cwmai': {'name': 'cwmai'},
            'valid-project': {'name': 'valid-project'}
        },
        'charter': {
            'primary_goal': 'Test',
            'secondary_goal': 'Test',
            'constraints': []
        },
        'system_performance': {
            'learning_metrics': {}
        },
        'task_queue': []
    }
    
    with open(test_state_file, 'w') as f:
        json.dump(contaminated_state, f)
    
    try:
        # Load with StateManager
        state_manager = StateManager()
        state_manager.state_file = test_state_file
        loaded_state = state_manager.load_state()
        
        # Check filtering was applied
        assert '.github' not in loaded_state['projects'], ".github should be filtered on load"
        assert 'cwmai' not in loaded_state['projects'], "cwmai should be filtered on load"
        assert 'valid-project' in loaded_state['projects'], "valid-project should remain"
        
        print(f"✅ StateManager filtering test passed - {len(loaded_state['projects'])} valid projects loaded")
        
    finally:
        # Cleanup
        if os.path.exists(test_state_file):
            os.remove(test_state_file)


async def test_resource_efficiency_updates():
    """Test that resource efficiency metrics update correctly."""
    print("\n🧪 Testing resource efficiency metric updates...")
    
    # Create test state manager
    state_manager = StateManager()
    state_manager.state = {
        'metrics': {
            'resource_efficiency': 0.0
        }
    }
    
    # Create resource manager and connect state manager
    resource_manager = ResourceManager()
    resource_manager.set_state_manager(state_manager)
    
    # Update metrics
    await resource_manager.update_resource_metrics()
    
    # Check that efficiency is no longer 0.0
    updated_state = state_manager.get_state()
    efficiency = updated_state['metrics']['resource_efficiency']
    
    assert efficiency > 0.0, f"Resource efficiency should be > 0.0, got {efficiency}"
    assert 0.0 <= efficiency <= 1.0, f"Resource efficiency should be between 0 and 1, got {efficiency}"
    
    print(f"✅ Resource efficiency test passed - efficiency updated to {efficiency:.3f}")


def test_task_persistence_skipping():
    """Test that task persistence tracks skipped tasks correctly."""
    print("\n🧪 Testing task persistence skip tracking...")
    
    # Create task persistence instance
    persistence = TaskPersistence(storage_file="test_persistence_temp.json")
    
    try:
        # Record a skipped task multiple times
        test_title = "Optimize system resource efficiency"
        
        for i in range(15):
            persistence.record_skipped_task(test_title)
        
        # Check skip stats
        assert test_title in persistence.skip_stats, "Task should be in skip stats"
        assert persistence.skip_stats[test_title]['count'] == 15, "Skip count should be 15"
        
        # Check cooldown increased
        assert persistence.cooldown_period > 300, f"Cooldown should increase after 10 skips, got {persistence.cooldown_period}"
        
        # Test duplicate detection with skip cooldown
        work_item = WorkItem(
            id="test-1",
            task_type="SYSTEM_IMPROVEMENT",
            title=test_title,
            description="Test task",
            priority=TaskPriority.HIGH
        )
        
        is_duplicate = persistence.is_duplicate_task(work_item)
        assert is_duplicate, "Task should be detected as duplicate due to skip cooldown"
        
        print(f"✅ Task persistence test passed - skip tracking and cooldown working correctly")
        
    finally:
        # Cleanup
        if os.path.exists("test_persistence_temp.json"):
            os.remove("test_persistence_temp.json")


def test_exclusion_consistency():
    """Test that exclusion is consistent across all patterns."""
    print("\n🧪 Testing exclusion consistency...")
    
    # Test all exclusion methods
    test_repos = ['.github', 'cwmai', 'cwmai.git', 'valid-repo']
    
    # Test is_excluded_repo
    assert RepositoryExclusion.is_excluded_repo('.github'), ".github should be excluded"
    assert RepositoryExclusion.is_excluded_repo('cwmai'), "cwmai should be excluded"
    assert RepositoryExclusion.is_excluded_repo('cwmai.git'), "cwmai.git should be excluded"
    assert not RepositoryExclusion.is_excluded_repo('valid-repo'), "valid-repo should not be excluded"
    
    # Test filter_excluded_repos
    filtered_list = RepositoryExclusion.filter_excluded_repos(test_repos)
    assert 'valid-repo' in filtered_list, "valid-repo should remain"
    assert '.github' not in filtered_list, ".github should be filtered"
    assert 'cwmai' not in filtered_list, "cwmai should be filtered"
    
    # Test filter_excluded_repos_dict
    test_dict = {repo: {'name': repo} for repo in test_repos}
    filtered_dict = RepositoryExclusion.filter_excluded_repos_dict(test_dict)
    assert 'valid-repo' in filtered_dict, "valid-repo should remain in dict"
    assert '.github' not in filtered_dict, ".github should be filtered from dict"
    
    print("✅ Exclusion consistency test passed - all methods filter correctly")


async def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("🔬 CWMAI Repository Exclusion Validation Tests")
    print("=" * 60)
    
    try:
        # Synchronous tests
        test_state_exclusion()
        test_worker_specialization()
        test_state_manager_filtering()
        test_exclusion_consistency()
        
        # Asynchronous tests
        await test_resource_efficiency_updates()
        
        # Task persistence test
        test_task_persistence_skipping()
        
        print("\n" + "=" * 60)
        print("✨ All tests passed! Repository exclusion is properly enforced.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the cleanup script: python scripts/cleanup_contamination.py")
        print("2. Restart the orchestrator: python run_continuous_ai.py --mode development --workers 2")
        print("3. Verify workers get proper specializations (not .github)")
        print("4. Verify resource efficiency updates (not stuck at 0.0)")
        print("5. Verify no infinite loops of duplicate tasks")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())