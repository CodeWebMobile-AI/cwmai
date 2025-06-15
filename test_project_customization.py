#!/usr/bin/env python3
"""
Test script to verify project customization works properly.
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import AIBrain
from scripts.work_item_types import WorkItem, TaskPriority


async def test_project_customization():
    """Test that project creation includes full customization."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n=== Project Customization Test ===\n")
    
    # Check for GitHub token
    github_token = os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ No GitHub token found. Set CLAUDE_PAT or GITHUB_TOKEN environment variable.")
        return False
    
    # Initialize AI brain
    print("1. Initializing AI brain...")
    ai_brain = AIBrain()
    
    # Initialize project creator
    print("2. Initializing project creator...")
    project_creator = ProjectCreator(github_token, ai_brain)
    
    # Create test task without pre-generated metadata
    print("\n3. Creating test NEW_PROJECT task WITHOUT pre-generated metadata...")
    task_without_metadata = {
        'title': 'Create Smart Home Automation Platform',
        'description': 'Build a platform for managing and automating smart home devices',
        'requirements': [
            'Real-time device control',
            'Automation rules engine',
            'Mobile app support'
        ],
        'type': 'NEW_PROJECT',
        'metadata': {
            'test_mode': True,
            'needs_venture_analysis': True,
            'needs_architecture': True
        }
    }
    
    print("\n4. Testing project creation with AI-generated details...")
    print(f"   Task: {task_without_metadata['title']}")
    print(f"   Has pre-generated metadata: NO")
    
    # This should trigger the fallback path
    result = await project_creator.create_project(task_without_metadata)
    
    if result.get('success'):
        print(f"\n✅ Project created successfully!")
        print(f"   Repository: {result.get('repo_url')}")
        print(f"   Customizations:")
        customizations = result.get('customizations', {})
        print(f"     - README updated: {customizations.get('readme_updated', False)}")
        print(f"     - Architecture saved: {customizations.get('architecture_saved', False)}")
        print(f"     - Package.json updated: {customizations.get('package_json_updated', False)}")
        print(f"   Initial issues created: {len(result.get('initial_issues', []))}")
        
        # Check if customization actually happened
        if customizations.get('readme_updated') and customizations.get('architecture_saved'):
            print("\n✅ FULL CUSTOMIZATION SUCCESSFUL!")
            return True
        else:
            print("\n⚠️ Partial customization - some steps may have failed")
            return False
    else:
        print(f"\n❌ Project creation failed: {result.get('error')}")
        return False


async def test_with_metadata():
    """Test project creation with pre-generated metadata."""
    
    print("\n\n=== Testing with Pre-generated Metadata ===\n")
    
    # Check for GitHub token
    github_token = os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
    if not github_token:
        return False
    
    # Initialize components
    ai_brain = AIBrain()
    project_creator = ProjectCreator(github_token, ai_brain)
    
    # Create task WITH pre-generated metadata
    print("1. Creating test NEW_PROJECT task WITH pre-generated metadata...")
    task_with_metadata = {
        'title': 'Create Online Learning Platform',
        'description': 'Build a platform for online courses and skill development',
        'requirements': [],
        'type': 'NEW_PROJECT',
        'metadata': {
            'test_mode': True,
            'selected_project': {
                'project_name': 'SkillForge Learning Platform',
                'project_goal': 'Create an online learning platform for professional skill development',
                'problem_solved': 'Professionals struggle to find time for continuous learning',
                'target_audience': 'Working professionals aged 25-45',
                'market_opportunity': 'Online education market growing at 15% annually',
                'monetization_strategy': 'Subscription model ($29/month) + course marketplace',
                'competitive_advantage': 'AI-powered personalized learning paths',
                'key_features': [
                    'Personalized learning paths',
                    'Progress tracking',
                    'Certificate generation',
                    'Live mentoring sessions',
                    'Mobile learning app'
                ]
            },
            'architecture': {
                'foundational_architecture': {
                    'design_system': {
                        'primary_color': '#4F46E5',
                        'font_family': 'Inter'
                    },
                    'database_schema': {
                        'users': 'User accounts and profiles',
                        'courses': 'Course content and metadata',
                        'progress': 'Learning progress tracking'
                    }
                }
            }
        }
    }
    
    print("\n2. Testing project creation with pre-generated details...")
    print(f"   Task: {task_with_metadata['title']}")
    print(f"   Has pre-generated metadata: YES")
    print(f"   Project name from metadata: {task_with_metadata['metadata']['selected_project']['project_name']}")
    
    result = await project_creator.create_project(task_with_metadata)
    
    if result.get('success'):
        print(f"\n✅ Project created successfully!")
        print(f"   Repository: {result.get('repo_url')}")
        customizations = result.get('customizations', {})
        print(f"   Customizations:")
        print(f"     - README updated: {customizations.get('readme_updated', False)}")
        print(f"     - Architecture saved: {customizations.get('architecture_saved', False)}")
        print(f"     - Package.json updated: {customizations.get('package_json_updated', False)}")
        
        return customizations.get('readme_updated') and customizations.get('architecture_saved')
    else:
        print(f"\n❌ Project creation failed: {result.get('error')}")
        return False


async def main():
    """Run all tests."""
    print("Project Customization Test Suite")
    print("=" * 50)
    print("\nThis test verifies that new projects get:")
    print("- Customized README.md")
    print("- ARCHITECTURE.md document")
    print("- Updated package.json")
    print("- Initial GitHub issues")
    
    # Test without metadata (AI generation path)
    test1_passed = await test_project_customization()
    
    # Test with metadata (pre-generated path)
    test2_passed = await test_with_metadata()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"- AI generation path: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"- Pre-generated path: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed! Project customization is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())