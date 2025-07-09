#!/usr/bin/env python3
"""Test simple local repository creation without AI"""

import os
import sys

# Add the zip folder to Python path
sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai/zip')

from scripts.local_github_simulator import LocalGitHubSimulator

def test_local_repo_creation():
    """Test creating a local repository with template"""
    
    # Create simulator
    simulator = LocalGitHubSimulator()
    
    print("Creating test repository...")
    
    # Create repository with template
    repo = simulator.create_repository(
        name="test-template-copy",
        description="Testing template copying functionality",
        organization="CodeWebMobile-AI",
        template="react-starter-kit"
    )
    
    print(f"✅ Repository created at: {repo.path}")
    
    # Check if template files were copied
    from pathlib import Path
    repo_path = Path(repo.path)
    
    print("\nChecking template files:")
    essential_files = ['package.json', 'composer.json', 'vite.config.ts', 'artisan', 'README.md']
    
    for file in essential_files:
        file_path = repo_path / file
        if file_path.exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    # Test update_file method
    print("\nTesting file updates...")
    try:
        repo.update_file(
            "README.md",
            "Update README with custom content",
            "# Test Template Copy\n\nThis repository was created to test template copying in local mode."
        )
        print("✅ Successfully updated README.md")
    except Exception as e:
        print(f"❌ Failed to update README: {e}")
    
    # Test create_file method
    try:
        repo.create_file(
            "ARCHITECTURE.md",
            "Add architecture documentation",
            "# Architecture\n\nThis is a test architecture document."
        )
        print("✅ Successfully created ARCHITECTURE.md")
    except Exception as e:
        print(f"❌ Failed to create ARCHITECTURE: {e}")
    
    # Create an issue
    print("\nTesting issue creation...")
    try:
        issue = simulator.create_issue(
            repo.full_name,
            "Initial Setup",
            "Set up the development environment",
            ["setup", "documentation"]
        )
        print(f"✅ Created issue #{issue.number}: {issue.title}")
    except Exception as e:
        print(f"❌ Failed to create issue: {e}")

if __name__ == "__main__":
    test_local_repo_creation()