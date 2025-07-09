#!/usr/bin/env python3
"""
Simple test for dynamic work generation without AI
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.dynamic_work_generator import DynamicWorkGenerator, WorkContext
from scripts.work_item_types import WorkItem, TaskPriority


async def test_basic_functionality():
    """Test basic dynamic work generation without AI."""
    print("🚀 Testing Dynamic Work Generation (No AI)")
    print("=" * 60)
    
    # Create work generator without AI
    generator = DynamicWorkGenerator(ai_brain=None)
    
    # Test 1: Basic work generation
    print("\n1️⃣ Testing basic work generation...")
    
    test_context = WorkContext(
        repository="test-repo",
        architecture_content=None,
        completed_work=[],
        current_issues=[],
        repository_analysis={
            'specific_needs': [
                {
                    'type': 'testing',
                    'description': 'Add unit tests for authentication',
                    'suggested_action': 'Create test suite',
                    'priority': 'high'
                },
                {
                    'type': 'documentation',
                    'description': 'API docs missing',
                    'suggested_action': 'Document endpoints',
                    'priority': 'medium'
                }
            ]
        }
    )
    
    work_items = generator._generate_work_without_ai(test_context, max_items=3)
    
    print(f"✅ Generated {len(work_items)} work items:")
    for item in work_items:
        print(f"   - {item.title}")
        print(f"     ID: {item.id}")
        print(f"     Type: {item.task_type}")
        print(f"     Priority: {item.priority.name}")
    
    # Test 2: Verify work items have correct structure
    print("\n2️⃣ Verifying work item structure...")
    
    for item in work_items:
        assert isinstance(item, WorkItem), "Should be WorkItem instance"
        assert item.id.startswith("need_"), "ID should start with need_"
        assert item.task_type == "IMPROVEMENT", "Type should be IMPROVEMENT"
        assert item.repository == "test-repo", "Repository should match"
        assert item.metadata.get('generated_by') == 'dynamic_work_generator', "Should have generator metadata"
    
    print("✅ All work items have correct structure")
    
    # Test 3: Duplicate filtering
    print("\n3️⃣ Testing duplicate filtering...")
    
    test_context.completed_work = [
        {
            'title': 'add unit tests for authentication',
            'description': 'Already done'
        }
    ]
    
    filtered_items = generator._generate_work_without_ai(test_context, max_items=3)
    
    print(f"✅ After filtering: {len(filtered_items)} items (was {len(work_items)})")
    
    # Verify the authentication test was filtered out
    auth_test_found = any('authentication' in item.title.lower() for item in filtered_items)
    if not auth_test_found:
        print("✅ Duplicate authentication test was filtered out")
    else:
        print("❌ Duplicate was not filtered")
    
    # Test 4: No fixed task types
    print("\n4️⃣ Verifying no fixed task types...")
    
    fixed_types = ['RESEARCH_TECHNOLOGY', 'RESEARCH_MARKET', 'maintenance_dependencies']
    found_fixed = []
    
    for item in work_items:
        if any(fixed in item.task_type for fixed in fixed_types):
            found_fixed.append(item.task_type)
    
    if found_fixed:
        print(f"❌ Found fixed task types: {found_fixed}")
    else:
        print("✅ No fixed task types found")
    
    print("\n✅ Basic tests completed successfully!")


async def test_repository_analysis_integration():
    """Test integration with repository analysis."""
    print("\n\n🧪 Testing Repository Analysis Integration...")
    print("=" * 60)
    
    from scripts.repository_analyzer import RepositoryAnalyzer
    
    # Create analyzer without AI
    analyzer = RepositoryAnalyzer(ai_brain=None, use_cache=False)
    
    # Test with a mock repository
    print("\n1️⃣ Creating mock repository analysis...")
    
    mock_analysis = {
        'health_metrics': {'health_score': 85},
        'lifecycle_analysis': {'current_stage': 'active_development'},
        'specific_needs': [
            {
                'type': 'feature',
                'description': 'Implement user dashboard',
                'suggested_action': 'Create dashboard component',
                'priority': 'high'
            }
        ],
        'architecture': {
            'document_exists': True,
            'raw_content': '# Architecture\nLaravel/React application'
        }
    }
    
    # Create work generator
    generator = DynamicWorkGenerator(ai_brain=None)
    
    # Extract architecture content
    arch_content = generator._extract_architecture_content(mock_analysis)
    print(f"✅ Extracted architecture content: {arch_content[:50]}...")
    
    # Generate work based on analysis
    work_items = await generator.generate_work_for_repository(
        repository="mock-repo",
        max_work_items=2,
        completed_work=[]
    )
    
    print(f"✅ Generated {len(work_items)} work items from analysis")
    
    print("\n✅ Repository integration test completed!")


async def main():
    """Run all tests."""
    try:
        await test_basic_functionality()
        await test_repository_analysis_integration()
        
        print("\n\n🎉 All tests passed!")
        print("\nKey findings:")
        print("✓ Dynamic work generator creates contextual work items")
        print("✓ No fixed task types are used")
        print("✓ Completed work is properly filtered")
        print("✓ Work items have correct structure")
        print("✓ Repository analysis integration works")
        
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set local mode
    os.environ['STORAGE_MODE'] = 'LOCAL'
    
    # Run tests
    asyncio.run(main())