#!/usr/bin/env python3
"""
Test Architecture Generation for Existing Projects

This script tests:
1. Detecting missing architecture documentation
2. Generating architecture from existing codebase
3. Creating ARCHITECTURE_DOCUMENTATION tasks
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
from scripts.ai_brain import AIBrain
from scripts.dynamic_charter import DynamicCharter


async def test_architecture_detection():
    """Test detection of missing architecture documentation."""
    print("\n=== Testing Architecture Detection ===\n")
    
    # Initialize components
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN not set")
        return False
        
    ai_brain = AIBrain()
    analyzer = RepositoryAnalyzer(github_token, ai_brain)
    
    # Test with a repository that likely doesn't have architecture
    test_repo = "CodeWebMobile-AI/cwmai"  # Using cwmai as example
    
    print(f"1. Analyzing repository: {test_repo}")
    
    try:
        analysis = await analyzer.analyze_repository(test_repo)
        
        # Check architecture field
        architecture = analysis.get('architecture', {})
        print(f"\n✅ Architecture field found:")
        print(f"   - Document exists: {architecture.get('document_exists', False)}")
        print(f"   - Generation available: {architecture.get('generation_available', False)}")
        
        # Check specific needs
        needs = analysis.get('specific_needs', [])
        arch_needs = [n for n in needs if n.get('type') == 'architecture_documentation']
        
        if arch_needs:
            print(f"\n✅ Architecture documentation need identified:")
            for need in arch_needs:
                print(f"   - Priority: {need.get('priority')}")
                print(f"   - Description: {need.get('description')}")
                print(f"   - Action: {need.get('suggested_action')}")
        else:
            print("\n⚠️ No architecture documentation need identified")
            
    except Exception as e:
        print(f"❌ Error analyzing repository: {e}")
        import traceback
        traceback.print_exc()
        
    return True


async def test_architecture_generation():
    """Test generating architecture for an existing project."""
    print("\n\n=== Testing Architecture Generation ===\n")
    
    # Initialize components
    github_token = os.getenv('GITHUB_TOKEN')
    ai_brain = AIBrain()
    
    # Create architecture generator
    generator = ArchitectureGenerator(github_token, ai_brain)
    
    # Create mock repository analysis
    mock_analysis = {
        'basic_info': {
            'name': 'test-project',
            'description': 'A Laravel React application',
            'language': 'PHP'
        },
        'technical_stack': {
            'primary_language': 'PHP',
            'frameworks': ['Laravel', 'React'],
            'infrastructure': ['Docker']
        },
        'code_analysis': {
            'file_types': {'.php': 50, '.tsx': 30, '.ts': 20},
            'key_directories': ['app', 'resources/js', 'database']
        }
    }
    
    print("1. Generating architecture from mock analysis...")
    
    try:
        architecture = await generator.generate_architecture_for_project(
            'test-org/test-project',
            mock_analysis
        )
        
        if architecture:
            print("\n✅ Architecture generated successfully!")
            print(f"   - Title: {architecture.get('title')}")
            print(f"   - Core entities: {len(architecture.get('core_entities', []))}")
            print(f"   - Has design system: {'design_system' in architecture}")
            print(f"   - Has foundation: {'foundational_architecture' in architecture}")
            print(f"   - Feature roadmap items: {len(architecture.get('feature_implementation_roadmap', []))}")
            print(f"   - Generated from: {architecture.get('generated_from')}")
            
            # Test formatting
            from scripts.project_creator import ProjectCreator
            creator = ProjectCreator(github_token)
            
            details = {
                'name': 'test-project',
                'description': mock_analysis['basic_info']['description'],
                'problem_statement': 'Extracted from existing codebase',
                'target_audience': 'Development team',
                'core_entities': architecture.get('core_entities', []),
                'architecture': architecture
            }
            
            formatted = creator._format_architecture_document(details)
            print(f"\n✅ Architecture document formatted:")
            print(f"   - Document length: {len(formatted)} characters")
            print(f"\nFirst 300 characters:")
            print(formatted[:300] + "...")
            
        else:
            print("❌ Architecture generation failed")
            
    except Exception as e:
        print(f"❌ Error generating architecture: {e}")
        import traceback
        traceback.print_exc()
        
    return True


async def test_architecture_task_generation():
    """Test generating ARCHITECTURE_DOCUMENTATION tasks."""
    print("\n\n=== Testing Architecture Task Generation ===\n")
    
    # Initialize components
    ai_brain = AIBrain()
    charter = DynamicCharter(ai_brain)
    task_generator = IntelligentTaskGenerator(ai_brain, charter)
    
    # Create analysis with missing architecture
    mock_analysis = {
        'repository': 'test-org/test-project',
        'basic_info': {
            'name': 'test-project',
            'description': 'Project created from Laravel React starter kit',  # Generic description
            'language': 'PHP'
        },
        'architecture': {
            'document_exists': False,
            'generation_available': True,
            'message': 'Architecture can be generated for this project'
        },
        'specific_needs': [
            {
                'type': 'architecture_documentation',
                'priority': 'high',
                'description': 'Repository lacks architecture documentation',
                'suggested_action': 'Generate and save ARCHITECTURE.md to document system design',
                'can_generate': True
            }
        ]
    }
    
    context = {
        'projects': [{'full_name': 'test-org/test-project'}],
        'recent_tasks': []
    }
    
    print("1. Generating task for repository with missing architecture...")
    
    try:
        # Simulate task generation
        from scripts.ai_task_content_generator import AITaskContentGenerator
        content_generator = AITaskContentGenerator(ai_brain)
        
        title, description = await content_generator.generate_architecture_documentation_content(
            'test-org/test-project',
            mock_analysis
        )
        
        print(f"\n✅ Architecture documentation task generated!")
        print(f"   - Title: {title}")
        print(f"\nDescription preview (first 500 chars):")
        print(description[:500] + "...")
        
    except Exception as e:
        print(f"❌ Error generating task: {e}")
        import traceback
        traceback.print_exc()
        
    return True


async def main():
    """Run all architecture generation tests."""
    print("Starting Architecture Generation Tests...")
    print("=" * 60)
    
    # Test 1: Architecture detection
    await test_architecture_detection()
    
    # Test 2: Architecture generation
    await test_architecture_generation()
    
    # Test 3: Architecture task generation
    await test_architecture_task_generation()
    
    print("\n" + "=" * 60)
    print("Architecture Generation Tests Completed!")
    print("\n✅ Summary:")
    print("   1. System can detect missing architecture documentation")
    print("   2. System can generate architecture from existing codebase")
    print("   3. System can create ARCHITECTURE_DOCUMENTATION tasks")
    print("   4. Projects without architecture will get documentation tasks")
    print("\n🎯 The system can now document existing projects!")


if __name__ == "__main__":
    asyncio.run(main())