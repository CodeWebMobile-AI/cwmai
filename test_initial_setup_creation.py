#!/usr/bin/env python3
"""
Test Initial Project Setup issue creation for projects without issues
"""
import asyncio
import os
import sys

sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai')
os.environ['STORAGE_MODE'] = 'local'

async def test_setup_issue_creation():
    """Test that Initial Project Setup issues are created for projects without issues"""
    print("Testing Initial Project Setup issue creation...")
    
    try:
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        from scripts.ai_brain import IntelligentAIBrain
        from scripts.state_manager import StateManager
        from scripts.local_github_simulator import LocalGitHubSimulator
        
        # Initialize components
        state_manager = StateManager()
        state = state_manager.load_state()
        ai_brain = IntelligentAIBrain(state, context={})
        
        # Create simulator to check issues
        simulator = LocalGitHubSimulator()
        
        # Mock system state with a project that has no issues
        test_project_name = 'fortressarigo/test-no-issues'
        system_state = {
            'projects': {
                test_project_name: {
                    'name': 'test-no-issues',
                    'description': 'Test project with no issues to verify setup creation',
                    'full_name': test_project_name,
                    'created_at': '2025-01-15T00:00:00Z',
                    'updated_at': '2025-01-15T00:00:00Z',
                    'lifecycle_stage': 'inception'
                }
            },
            'repositories': {}
        }
        
        # Ensure the project exists in local simulator
        try:
            repo = simulator.create_repository(test_project_name)
            print(f"✅ Created test repository: {test_project_name}")
        except Exception as e:
            print(f"Repository might already exist: {e}")
        
        # Check initial issue count
        initial_issues = simulator.get_issues(test_project_name, state="open")
        print(f"\nInitial issue count for {test_project_name}: {len(initial_issues)}")
        
        # Create work finder
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        
        # Discover work - this should create Initial Project Setup issue
        print("\n🔍 Running work discovery...")
        work_items = await work_finder.discover_work(max_items=10)
        
        print(f"\nFound {len(work_items)} work items")
        
        # Check if Initial Project Setup issue was created
        print("\n📋 Checking for Initial Project Setup issue...")
        updated_issues = simulator.get_issues(test_project_name, state="open")
        print(f"Updated issue count: {len(updated_issues)}")
        
        setup_issue_found = False
        for issue in updated_issues:
            print(f"\n  Issue #{issue.number}: {issue.title}")
            if "Initial Project Setup" in issue.title:
                setup_issue_found = True
                print("  ✅ This is the Initial Project Setup issue!")
                print(f"  Labels: {issue.labels if hasattr(issue, 'labels') else 'N/A'}")
                # Show first few lines of description
                if hasattr(issue, 'body') and issue.body:
                    lines = issue.body.split('\n')[:10]
                    print("  Description preview:")
                    for line in lines:
                        print(f"    {line}")
        
        if setup_issue_found:
            print("\n✅ SUCCESS: Initial Project Setup issue was created for project without issues!")
        else:
            print("\n❌ FAIL: Initial Project Setup issue was NOT created!")
            
        # Also check work items to see if setup issue appears there
        print("\n📊 Work items that include setup issues:")
        for item in work_items:
            if hasattr(item, 'metadata') and item.metadata.get('initial_setup_issue'):
                print(f"  - {item.title} (from initial setup)")
            elif 'setup' in str(item).lower() and hasattr(item, 'title'):
                print(f"  - {item.title}")
                
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_setup_issue_creation())