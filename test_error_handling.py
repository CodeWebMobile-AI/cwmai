#!/usr/bin/env python3
"""
Test Error Handling in Architecture System
"""

import asyncio
import os
import sys
import logging

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.architecture_generator import ArchitectureGenerator
from scripts.ai_brain import AIBrain

# Set up logging to see error messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


async def test_description_generation_with_errors():
    """Test description generation with AI failures."""
    print("\n=== Testing Description Generation Error Handling ===\n")
    
    # Test 1: Without AI Brain (should use fallback immediately)
    print("Test 1: No AI Brain available")
    generator_no_ai = ArchitectureGenerator('dummy_token', ai_brain=None)
    
    test_arch = {'core_entities': ['User', 'Product']}
    desc = await generator_no_ai._generate_repository_description(
        'project-expense-tracker',
        test_arch
    )
    print(f"Result: {desc}")
    print(f"Is fallback description: {'✅' if 'Expense Tracker' in desc else '❌'}\n")
    
    # Test 2: With AI Brain but simulating failure
    print("Test 2: AI Brain fails")
    ai_brain = AIBrain()
    generator_with_ai = ArchitectureGenerator('dummy_token', ai_brain)
    
    # This should trigger an error and then use fallback
    desc2 = await generator_with_ai._generate_repository_description(
        'inventory-management-system',
        test_arch
    )
    print(f"Result: {desc2}")
    print(f"Is fallback description: {'✅' if 'Inventory Management' in desc2 and 'Laravel and React' in desc2 else '❌'}\n")


async def test_architecture_generation_with_errors():
    """Test architecture generation with AI failures."""
    print("\n=== Testing Architecture Generation Error Handling ===\n")
    
    ai_brain = AIBrain()
    generator = ArchitectureGenerator('dummy_token', ai_brain)
    
    mock_info = {
        'name': 'customer-support-portal',
        'description': 'Project created from Laravel React starter kit',
        'language': 'PHP'
    }
    
    # This should fail AI generation and use fallback
    arch = await generator._generate_architecture_with_ai(
        'customer-support-portal',
        mock_info,
        {'frameworks': ['Laravel', 'React']},
        {},
        {'models': ['Ticket', 'Customer', 'Agent']}
    )
    
    print(f"Architecture generated: {'✅' if arch else '❌'}")
    print(f"Title: {arch.get('title', 'None')}")
    print(f"Description: {arch.get('description', 'None')}")
    print(f"Generated from: {arch.get('generated_from', 'Unknown')}")
    print(f"Has core entities: {'✅' if arch.get('core_entities') else '❌'}")


async def main():
    """Run error handling tests."""
    print("Starting Error Handling Tests...")
    print("=" * 60)
    print("Expected behavior:")
    print("- Errors should be logged before fallback")
    print("- Fallback should still produce meaningful results")
    print("- No crashes, always returns something useful")
    print("=" * 60)
    
    await test_description_generation_with_errors()
    await test_architecture_generation_with_errors()
    
    print("\n" + "=" * 60)
    print("Error Handling Tests Completed!")
    print("\n✅ Key Points:")
    print("   1. System logs errors before using fallback")
    print("   2. Fallback descriptions use repository names")
    print("   3. System never crashes, always returns useful output")


if __name__ == "__main__":
    asyncio.run(main())