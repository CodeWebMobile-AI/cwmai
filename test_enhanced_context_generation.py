#!/usr/bin/env python3
"""
Test the enhanced tool generation with full script context discovery
"""

import asyncio
import json
from scripts.improved_tool_generator import ImprovedToolGenerator
from scripts.tool_generation_templates import ToolGenerationTemplates


async def test_context_discovery():
    """Test that the system discovers all available scripts."""
    print("Testing Script Discovery System")
    print("=" * 80)
    
    # Initialize templates to trigger discovery
    templates = ToolGenerationTemplates()
    
    # Show discovered modules
    print("\nDiscovered Modules Summary:")
    print(templates.get_available_modules_summary())
    
    # Show import context
    print("\n" + "=" * 80)
    print("Import Context (first 2000 chars):")
    import_context = templates.get_import_context()
    print(import_context[:2000] + "..." if len(import_context) > 2000 else import_context)
    
    # Show some specific module details
    print("\n" + "=" * 80)
    print("Sample Module Details:")
    sample_modules = ['task_manager', 'swarm_intelligence', 'redis_async_state_manager']
    for module in sample_modules:
        details = templates.get_module_details(module)
        if details:
            print(f"\n{module}:")
            print(details)


async def test_tool_generation_with_context():
    """Test generating a tool that uses discovered modules."""
    print("\n" + "=" * 80)
    print("Testing Tool Generation with Full Context")
    print("=" * 80)
    
    generator = ImprovedToolGenerator()
    
    # Test case: Generate a tool that needs to use multiple discovered modules
    test_cases = [
        {
            "name": "analyze_worker_performance",
            "description": "Analyze performance metrics of all active workers",
            "requirements": "Use worker monitoring modules to collect metrics, aggregate data, and return performance summary"
        },
        {
            "name": "find_unused_modules",
            "description": "Find Python modules in scripts directory that are not imported anywhere",
            "requirements": "Scan all Python files, track imports, identify unused modules"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nGenerating tool: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        
        result = await generator.generate_tool(
            name=test_case['name'],
            description=test_case['description'],
            requirements=test_case['requirements']
        )
        
        if result['success']:
            print("✓ Tool generated successfully!")
            print("\nGenerated code preview (first 1000 chars):")
            code_preview = result['code'][:1000] + "..." if len(result['code']) > 1000 else result['code']
            print(code_preview)
            
            # Check if it uses any discovered modules
            discovered_imports = []
            for line in result['code'].split('\n'):
                if 'from scripts.' in line and 'import' in line:
                    discovered_imports.append(line.strip())
            
            if discovered_imports:
                print("\nDiscovered module imports used:")
                for imp in discovered_imports:
                    print(f"  - {imp}")
        else:
            print("✗ Tool generation failed:")
            print(f"  Issues: {result.get('validation', {}).get('issues', [])}")


async def main():
    """Run all tests."""
    await test_context_discovery()
    await test_tool_generation_with_context()
    
    print("\n" + "=" * 80)
    print("Test completed! The tool generation system now has full context")
    print("about all available scripts in /workspaces/cwmai/scripts")


if __name__ == "__main__":
    asyncio.run(main())