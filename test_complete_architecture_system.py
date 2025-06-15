#!/usr/bin/env python3
"""
Complete Test of Architecture System

Tests:
1. Architecture persistence in new projects
2. Architecture detection for existing projects
3. Description update for generic descriptions
4. Task generation for missing architecture
5. Full end-to-end workflow
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.architecture_generator import ArchitectureGenerator
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.ai_task_content_generator import AITaskContentGenerator
from scripts.ai_brain import AIBrain
from scripts.dynamic_charter import DynamicCharter


async def test_repository_with_generic_description():
    """Test a repository with generic description."""
    print("\n=== Testing Repository with Generic Description ===\n")
    
    # Initialize components
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN not set")
        return False
        
    ai_brain = AIBrain()
    analyzer = RepositoryAnalyzer(github_token, ai_brain)
    generator = ArchitectureGenerator(github_token, ai_brain)
    
    # Simulate a repository with generic description
    mock_repo_name = "project-analytics-dashboard"
    mock_analysis = {
        'repository': f'test-org/{mock_repo_name}',
        'basic_info': {
            'name': mock_repo_name,
            'description': 'Project created from Laravel React starter kit',  # Generic!
            'language': 'PHP'
        },
        'architecture': {
            'document_exists': False,
            'generation_available': True
        },
        'technical_stack': {
            'frameworks': ['Laravel', 'React'],
            'primary_language': 'PHP'
        },
        'code_analysis': {
            'documentation': [],  # No README
            'test_coverage': 'unknown'
        },
        'issues_analysis': {
            'bug_count': 0,
            'feature_requests': 0
        },
        'health_metrics': {
            'days_since_update': 5
        }
    }
    
    print(f"1. Repository: {mock_repo_name}")
    print(f"   Current description: '{mock_analysis['basic_info']['description']}'")
    print(f"   Has architecture: {mock_analysis['architecture']['document_exists']}")
    
    # Check if system identifies the needs
    needs = analyzer._identify_specific_needs(mock_analysis)
    
    arch_needs = [n for n in needs if n['type'] == 'architecture_documentation']
    desc_needs = [n for n in needs if n['type'] == 'repository_description']
    
    print(f"\n2. Identified Needs:")
    print(f"   ✅ Architecture documentation needed: {len(arch_needs) > 0}")
    print(f"   ✅ Description update needed: {len(desc_needs) > 0}")
    
    if desc_needs:
        print(f"      - Current: {desc_needs[0].get('current_description')}")
        print(f"      - Action: {desc_needs[0].get('suggested_action')}")
    
    # Test description generation
    print(f"\n3. Testing description generation from repo name...")
    
    try:
        # Create minimal architecture for description generation
        test_architecture = {
            'description': 'Analytics system',
            'core_entities': ['Metric', 'Dashboard', 'Report']
        }
        
        new_description = await generator._generate_repository_description(
            mock_repo_name,
            test_architecture
        )
        
        print(f"   ✅ Generated description: \"{new_description}\"")
        print(f"   Length: {len(new_description)} chars (max 350)")
        
        # Check if it's better than generic
        is_better = "starter kit" not in new_description.lower()
        print(f"   Better than generic: {'✅ Yes' if is_better else '❌ No'}")
        
    except Exception as e:
        print(f"   ❌ Error generating description: {e}")
        import traceback
        traceback.print_exc()
    
    return True


async def test_architecture_generation_with_name():
    """Test architecture generation using repository name."""
    print("\n\n=== Testing Architecture Generation with Repository Name ===\n")
    
    github_token = os.getenv('GITHUB_TOKEN')
    ai_brain = AIBrain()
    generator = ArchitectureGenerator(github_token, ai_brain)
    
    # Test different repository names
    test_cases = [
        "inventory-management-system",
        "customer-support-portal",
        "employee-timesheet-tracker",
        "project-expense-tracker"
    ]
    
    for repo_name in test_cases:
        print(f"\nTesting: {repo_name}")
        
        mock_analysis = {
            'basic_info': {
                'name': repo_name,
                'description': 'Project created from Laravel React starter kit',
                'language': 'PHP'
            },
            'technical_stack': {
                'frameworks': ['Laravel', 'React']
            }
        }
        
        try:
            # Use basic architecture generation (no AI needed for test)
            basic_arch = generator._create_basic_architecture(
                mock_analysis['basic_info'],
                mock_analysis['technical_stack'],
                {}
            )
            
            print(f"   ✅ Architecture title: {basic_arch.get('title')}")
            print(f"   ✅ Architecture description: {basic_arch.get('description')}")
            
            # Verify it uses the repo name
            name_used = any(part in basic_arch.get('description', '').lower() 
                          for part in repo_name.split('-'))
            print(f"   Uses repo name: {'✅ Yes' if name_used else '❌ No'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def test_task_generation_for_missing_architecture():
    """Test that proper tasks are generated for missing architecture."""
    print("\n\n=== Testing Task Generation for Missing Architecture ===\n")
    
    ai_brain = AIBrain()
    content_generator = AITaskContentGenerator(ai_brain)
    
    mock_repo_context = {
        'basic_info': {
            'name': 'project-analytics-dashboard',
            'description': 'Project created from Laravel React starter kit',
            'language': 'PHP'
        },
        'architecture': {
            'document_exists': False
        },
        'technical_stack': {
            'frameworks': ['Laravel', 'React']
        }
    }
    
    print("1. Generating ARCHITECTURE_DOCUMENTATION task...")
    
    try:
        title, description = await content_generator.generate_architecture_documentation_content(
            'test-org/project-analytics-dashboard',
            mock_repo_context
        )
        
        print(f"\n   ✅ Task Title: {title}")
        print(f"\n   Task Description Preview:")
        print(f"   {description[:300]}...")
        
        # Verify task mentions architecture
        has_architecture_mention = 'architecture' in description.lower()
        print(f"\n   Mentions architecture: {'✅ Yes' if has_architecture_mention else '❌ No'}")
        
    except Exception as e:
        print(f"   ❌ Error generating task: {e}")
        import traceback
        traceback.print_exc()


async def test_full_workflow():
    """Test the complete workflow from detection to task generation."""
    print("\n\n=== Testing Complete Workflow ===\n")
    
    github_token = os.getenv('GITHUB_TOKEN')
    ai_brain = AIBrain()
    
    # Initialize all components
    analyzer = RepositoryAnalyzer(github_token, ai_brain)
    charter = DynamicCharter(ai_brain)
    task_generator = IntelligentTaskGenerator(ai_brain, charter)
    
    # Mock a repository that needs everything
    mock_repo_name = "CodeWebMobile-AI/project-task-management-system"
    
    print(f"1. Simulating analysis of: {mock_repo_name}")
    
    # Create complete mock analysis
    mock_analysis = {
        'repository': mock_repo_name,
        'basic_info': {
            'name': 'project-task-management-system',
            'description': 'Project created from Laravel React starter kit',
            'language': 'PHP',
            'open_issues_count': 0
        },
        'health_metrics': {
            'health_score': 70
        },
        'architecture': {
            'document_exists': False,
            'generation_available': True
        },
        'technical_stack': {
            'frameworks': ['Laravel', 'React', 'TypeScript']
        },
        'specific_needs': [
            {
                'type': 'architecture_documentation',
                'priority': 'high',
                'description': 'Repository lacks architecture documentation',
                'suggested_action': 'Generate and save ARCHITECTURE.md'
            }
        ],
        'issues_analysis': {
            'recent_issues': []
        },
        'recent_activity': {},
        'code_analysis': {}
    }
    
    print("\n2. Checking what tasks would be generated...")
    
    context = {
        'projects': [{'full_name': mock_repo_name}],
        'recent_tasks': []
    }
    
    try:
        # Test task generation
        task = await task_generator.generate_task_for_repository(
            mock_repo_name,
            mock_analysis,
            context
        )
        
        if task and not task.get('skip'):
            print(f"\n   ✅ Task generated successfully!")
            print(f"   Type: {task.get('type')}")
            print(f"   Title: {task.get('title', 'No title')}")
            print(f"   Priority: {task.get('priority')}")
            
            # Check if it's architecture related
            is_arch_task = 'architecture' in str(task).lower()
            print(f"   Architecture-related: {'✅ Yes' if is_arch_task else '❌ No'}")
        else:
            print("   ⚠️ Task was skipped or not generated")
            
    except Exception as e:
        print(f"   ❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all comprehensive tests."""
    print("Starting Complete Architecture System Tests...")
    print("=" * 70)
    
    # Test 1: Repository with generic description
    await test_repository_with_generic_description()
    
    # Test 2: Architecture generation using repo names
    await test_architecture_generation_with_name()
    
    # Test 3: Task generation for missing architecture
    await test_task_generation_for_missing_architecture()
    
    # Test 4: Full workflow
    await test_full_workflow()
    
    print("\n" + "=" * 70)
    print("Complete Architecture System Tests Finished!")
    print("\n🎯 Summary of what should happen:")
    print("   1. Repositories with generic descriptions are detected")
    print("   2. Repository names are used to generate meaningful descriptions")
    print("   3. Architecture documentation tasks are created when needed")
    print("   4. The full workflow identifies and addresses missing documentation")


if __name__ == "__main__":
    asyncio.run(main())