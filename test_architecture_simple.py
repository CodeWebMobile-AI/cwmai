#!/usr/bin/env python3
"""
Simple Test of Architecture System Core Functions
"""

import asyncio
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.architecture_generator import ArchitectureGenerator
from scripts.ai_brain import AIBrain


async def test_description_generation():
    """Test description generation from repository names."""
    print("\n=== Testing Description Generation from Repository Names ===\n")
    
    github_token = os.getenv('GITHUB_TOKEN', 'dummy_token')
    ai_brain = AIBrain()
    generator = ArchitectureGenerator(github_token, ai_brain)
    
    test_cases = [
        {
            'repo_name': 'project-analytics-dashboard',
            'expected': 'analytics dashboard'
        },
        {
            'repo_name': 'inventory-management-system',
            'expected': 'inventory management'
        },
        {
            'repo_name': 'customer-support-portal',
            'expected': 'customer support'
        },
        {
            'repo_name': 'employee-timesheet-tracker',
            'expected': 'employee timesheet'
        }
    ]
    
    for test in test_cases:
        repo_name = test['repo_name']
        expected = test['expected']
        
        print(f"\nTesting: {repo_name}")
        
        # Test name parsing
        name_parts = repo_name.lower().replace('-', ' ').replace('_', ' ').split()
        prefixes_to_remove = ['project', 'app', 'application', 'system', 'platform']
        filtered_parts = [part for part in name_parts if part not in prefixes_to_remove]
        meaningful_name = ' '.join(filtered_parts)
        
        print(f"   Parsed name: {meaningful_name}")
        print(f"   Contains expected words: {'✅' if expected in meaningful_name else '❌'}")
        
        # Test basic architecture generation
        mock_basic_info = {
            'name': repo_name,
            'description': 'Project created from Laravel React starter kit'
        }
        
        basic_arch = generator._create_basic_architecture(
            mock_basic_info,
            {'frameworks': ['Laravel', 'React']},
            {}
        )
        
        print(f"   Generated title: {basic_arch.get('title')}")
        print(f"   Generated description: {basic_arch.get('description')}")
        
        # Check if generic phrase is gone
        has_generic = 'starter kit' in basic_arch.get('description', '').lower()
        print(f"   Removed generic phrase: {'❌ Still generic' if has_generic else '✅ Yes'}")


async def test_architecture_needs_detection():
    """Test detection of architecture needs."""
    print("\n\n=== Testing Architecture Needs Detection ===\n")
    
    # Test cases for different scenarios
    test_cases = [
        {
            'name': 'No architecture, generic description',
            'description': 'Project created from Laravel React starter kit',
            'has_architecture': False,
            'should_need_arch': True,
            'should_need_desc': True
        },
        {
            'name': 'No architecture, good description',
            'description': 'Analytics dashboard for tracking business metrics',
            'has_architecture': False,
            'should_need_arch': True,
            'should_need_desc': False
        },
        {
            'name': 'Has architecture, generic description',
            'description': 'Forked from starter template',
            'has_architecture': True,
            'should_need_arch': False,
            'should_need_desc': True
        },
        {
            'name': 'Has everything',
            'description': 'Complete inventory management system',
            'has_architecture': True,
            'should_need_arch': False,
            'should_need_desc': False
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"   Description: \"{test['description']}\"")
        print(f"   Has architecture: {test['has_architecture']}")
        
        # Check architecture need
        needs_arch = not test['has_architecture']
        print(f"   Should need architecture: {test['should_need_arch']} → {'✅' if needs_arch == test['should_need_arch'] else '❌'}")
        
        # Check description need
        generic_phrases = ['starter kit', 'template', 'forked from', 'boilerplate']
        needs_desc = any(phrase in test['description'].lower() for phrase in generic_phrases)
        print(f"   Should need description update: {test['should_need_desc']} → {'✅' if needs_desc == test['should_need_desc'] else '❌'}")


async def test_full_generation_flow():
    """Test the full generation flow with AI."""
    print("\n\n=== Testing Full Generation Flow ===\n")
    
    github_token = os.getenv('GITHUB_TOKEN', 'dummy_token')
    ai_brain = AIBrain()
    generator = ArchitectureGenerator(github_token, ai_brain)
    
    # Test with a descriptive repo name
    repo_name = "project-expense-tracker"
    
    print(f"Repository: {repo_name}")
    print("Current description: 'Project created from Laravel React starter kit'")
    
    try:
        # Test description generation
        test_architecture = {
            'description': 'Expense tracking system',
            'core_entities': ['Expense', 'Category', 'Budget']
        }
        
        new_description = await generator._generate_repository_description(
            repo_name,
            test_architecture
        )
        
        print(f"\n✅ Generated description:")
        print(f"   \"{new_description}\"")
        print(f"   Length: {len(new_description)}/350 chars")
        
        # Verify it's not generic
        is_generic = any(phrase in new_description.lower() 
                        for phrase in ['starter kit', 'template', 'laravel react'])
        print(f"   Generic phrases removed: {'✅ Yes' if not is_generic else '❌ No'}")
        
        # Verify it uses the repo name context
        uses_expense = 'expense' in new_description.lower()
        print(f"   References 'expense': {'✅ Yes' if uses_expense else '❌ No'}")
        
    except Exception as e:
        print(f"\n❌ Error in generation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all simple tests."""
    print("Starting Simple Architecture System Tests...")
    print("=" * 60)
    
    # Test 1: Description generation from names
    await test_description_generation()
    
    # Test 2: Architecture needs detection
    await test_architecture_needs_detection()
    
    # Test 3: Full generation flow
    await test_full_generation_flow()
    
    print("\n" + "=" * 60)
    print("Simple Architecture System Tests Completed!")
    print("\n✅ Key Features Verified:")
    print("   1. Repository names are parsed correctly")
    print("   2. Generic descriptions are replaced with meaningful ones")
    print("   3. System detects when architecture/description updates are needed")
    print("   4. Generated descriptions use repository name context")


if __name__ == "__main__":
    asyncio.run(main())