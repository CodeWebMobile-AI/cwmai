#!/usr/bin/env python3
"""
Test script to verify local issue creation functionality.
Tests the fix for the missing create_issue method in LocalIssueCreator.
"""

import os
import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.local_issue_creator import LocalIssueCreator
from scripts.task_manager import TaskManager
from scripts.storage_mode import StorageMode


def test_local_issue_creation():
    """Test creating an issue in a local repository."""
    print("Testing local issue creation...")
    
    # Set storage mode to LOCAL
    os.environ['STORAGE_MODE'] = 'LOCAL'
    
    # Use an existing repository
    test_repo = "verisafe"
    test_repo_full = f"CodeWebMobile-AI/{test_repo}"
    
    # Initialize components
    print(f"\n1. Initializing components for repository: {test_repo}")
    task_manager = TaskManager(repository=test_repo_full)
    
    # Create a test issue similar to "Initial Project Setup"
    test_title = "Test Initial Project Setup"
    test_description = """@claude This is a test issue to verify local issue creation.

## Initial Project Setup

Based on the project architecture for **verisafe**, complete the following setup tasks:

1. **Configure API authentication**
   - [ ] Set up Sanctum/API authentication middleware
   - [ ] Create auth controllers
   - [ ] Add protected API routes

2. **Create initial data models**
   - [ ] Create models based on the problem domain
   - [ ] Create corresponding migrations
   - [ ] Set up model relationships

3. **Update starter kit dashboard component**
   - [ ] Customize the dashboard for verisafe
   - [ ] Add data display widgets
   - [ ] Update navigation

This is a test issue to verify the fix for local issue creation."""
    
    test_labels = ['setup', 'test', 'api', 'frontend']
    
    print(f"\n2. Creating test issue: '{test_title}'")
    
    try:
        # Create the issue
        issue_number = task_manager.create_ai_task_issue(
            title=test_title,
            description=test_description,
            labels=test_labels,
            priority="high",
            task_type="setup",
            repository=test_repo_full
        )
    except Exception as e:
        print(f"❌ Error creating issue: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if issue_number:
        print(f"✅ Successfully created issue #{issue_number}")
        
        # Verify the issue file exists
        repo_path = Path(f"local_repos/repositories/{test_repo_full}")
        issues_dir = repo_path / "issues"
        issue_file = issues_dir / f"issue-{issue_number}.json"
        
        print(f"\n3. Verifying issue file: {issue_file}")
        
        if issue_file.exists():
            print("✅ Issue file exists")
            
            # Read and display issue content
            with open(issue_file, 'r') as f:
                issue_data = json.load(f)
            
            print("\n4. Issue content:")
            print(f"   - Number: {issue_data.get('number')}")
            print(f"   - Title: {issue_data.get('title')}")
            print(f"   - State: {issue_data.get('state')}")
            print(f"   - Labels: {issue_data.get('labels')}")
            print(f"   - Created: {issue_data.get('created_at')}")
            print(f"   - Body preview: {issue_data.get('body', '')[:100]}...")
            
            # Check repository metadata
            metadata_file = repo_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print(f"\n5. Repository metadata:")
                print(f"   - Next issue number: {metadata.get('next_issue_number', 'N/A')}")
                if 'statistics' in metadata and 'issues' in metadata['statistics']:
                    print(f"   - Total issues: {metadata['statistics']['issues'].get('total', 0)}")
                    print(f"   - Open issues: {metadata['statistics']['issues'].get('open', 0)}")
            
            print("\n✅ TEST PASSED: Local issue creation is working correctly!")
            return True
        else:
            print(f"❌ Issue file not found at: {issue_file}")
            return False
    else:
        print("❌ Failed to create issue - returned None")
        return False


def test_direct_local_issue_creator():
    """Test the LocalIssueCreator directly."""
    print("\n\n=== Testing LocalIssueCreator directly ===")
    
    try:
        # Create LocalIssueCreator instance
        issue_creator = LocalIssueCreator()
        
        # Check if method exists
        if hasattr(issue_creator, 'create_issue'):
            print("✅ create_issue method exists")
        else:
            print("❌ create_issue method not found")
            print(f"Available methods: {[m for m in dir(issue_creator) if not m.startswith('_')]}")
            return
        
        # Test the create_issue method directly
        test_repo = "verisafe"
        issue_number = issue_creator.create_issue(
            repository=test_repo,
            title="Direct Test Issue",
            body="@claude This is a direct test of the create_issue method.",
            labels=["test", "direct"]
        )
    except Exception as e:
        print(f"❌ Error during direct test: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if issue_number:
        print(f"✅ Direct test successful - created issue #{issue_number}")
        
        # List all issues in the repository
        repo_path = Path(f"local_repos/repositories/CodeWebMobile-AI/{test_repo}/issues")
        if repo_path.exists():
            issue_files = list(repo_path.glob("issue-*.json"))
            print(f"\nTotal issues in {test_repo}: {len(issue_files)}")
            for issue_file in sorted(issue_files):
                print(f"  - {issue_file.name}")
    else:
        print("❌ Direct test failed - create_issue returned None")


if __name__ == "__main__":
    # Run tests
    success = test_local_issue_creation()
    test_direct_local_issue_creator()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)