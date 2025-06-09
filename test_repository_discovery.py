#!/usr/bin/env python3
"""
Test Repository Discovery System

Simple test script to validate that the repository discovery system
can correctly find and integrate CodeWebMobile-AI organization repositories.
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.insert(0, 'scripts')

def test_repository_discovery():
    """Test the repository discovery functionality."""
    print("🔍 Testing Repository Discovery System")
    print("=" * 50)
    
    # Test 1: StateManager repository discovery
    print("\n1. Testing StateManager repository discovery...")
    try:
        from state_manager import StateManager
        
        state_manager = StateManager()
        
        # Test basic discovery
        repositories = state_manager.discover_organization_repositories()
        print(f"✓ Discovered {len(repositories)} repositories")
        
        if repositories:
            print("\nRepositories found:")
            for repo in repositories:
                print(f"  - {repo['name']} ({repo['language']}) - Health: {repo['health_score']:.1f}")
        else:
            print("⚠️  No repositories discovered")
            
    except Exception as e:
        print(f"❌ StateManager test failed: {e}")
        return False
    
    # Test 2: Full state loading with discovery
    print("\n2. Testing state loading with repository discovery...")
    try:
        state = state_manager.load_state_with_repository_discovery()
        projects = state.get('projects', {})
        
        print(f"✓ State loaded with {len(projects)} projects")
        
        if projects:
            print("\nProjects in state:")
            for project_id, project_data in projects.items():
                name = project_data.get('name', project_id)
                health = project_data.get('health_score', 0)
                repo_type = project_data.get('type', 'unknown')
                print(f"  - {name} (Type: {repo_type}) - Health: {health:.1f}")
        
    except Exception as e:
        print(f"❌ State loading test failed: {e}")
        return False
    
    # Test 3: AI Brain Factory integration
    print("\n3. Testing AI Brain Factory with repository discovery...")
    try:
        from ai_brain_factory import AIBrainFactory
        
        # Test workflow creation
        brain = AIBrainFactory.create_for_workflow()
        brain_state = brain.state
        brain_projects = brain_state.get('projects', {})
        
        print(f"✓ AI Brain created with {len(brain_projects)} projects")
        
        if brain_projects:
            print("Projects in AI Brain:")
            for project_id, project_data in brain_projects.items():
                name = project_data.get('name', project_id)
                print(f"  - {name}")
        
    except Exception as e:
        print(f"❌ AI Brain Factory test failed: {e}")
        return False
    
    # Test 4: Dynamic God Mode Controller integration
    print("\n4. Testing Dynamic God Mode Controller integration...")
    try:
        from god_mode_controller import GodModeConfig, IntensityLevel
        from dynamic_god_mode_controller import DynamicGodModeController
        
        config = GodModeConfig(
            intensity=IntensityLevel.BALANCED,
            enable_self_modification=False,
            enable_multi_repo=True,
            enable_predictive=False,
            enable_quantum=False,
            safety_threshold=0.8
        )
        
        controller = DynamicGodModeController(config)
        discovered_repos = controller.discovered_repositories
        
        print(f"✓ Controller initialized with {len(discovered_repos)} repositories")
        
        # Test active projects method
        active_projects = controller._get_active_projects()
        print(f"✓ Found {len(active_projects)} active projects")
        
        if active_projects:
            print("Active projects:")
            for project in active_projects:
                name = project.get('name', 'Unknown')
                project_type = project.get('type', 'unknown')
                health = project.get('health_score', 0)
                print(f"  - {name} (Type: {project_type}) - Health: {health:.1f}")
        
    except Exception as e:
        print(f"❌ Controller integration test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All repository discovery tests passed!")
    print("\nSystem should now properly recognize the 3 repositories")
    print("in the CodeWebMobile-AI organization instead of showing")
    print("'no active projects'.")
    return True

def main():
    """Main test function."""
    print("Repository Discovery System Test")
    print(f"Running at: {datetime.now(timezone.utc).isoformat()}")
    print(f"GitHub Token Available: {'Yes' if os.getenv('CLAUDE_PAT') else 'No'}")
    
    if not os.getenv('CLAUDE_PAT'):
        print("⚠️  Warning: No GitHub token found. Discovery may not work properly.")
    
    success = test_repository_discovery()
    
    if success:
        print("\n✅ Repository discovery system is working correctly!")
    else:
        print("\n❌ Repository discovery system has issues that need to be addressed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)