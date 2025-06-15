#!/usr/bin/env python3
"""
Test the enhanced smart tool generation templates
"""

import asyncio
from scripts.tool_generation_templates import ToolGenerationTemplates


async def test_smart_generation():
    """Test the smart tool generation capabilities."""
    print("Testing Smart Tool Generation Templates\n")
    
    # Initialize templates
    templates = ToolGenerationTemplates(use_ai=True)
    
    # Test cases
    test_cases = [
        {
            'name': 'file_monitor',
            'description': 'Monitor files for changes and trigger actions',
            'requirements': 'Watch directory recursively, detect file changes, filter by patterns, execute callbacks'
        },
        {
            'name': 'api_rate_limiter',
            'description': 'Implement intelligent API rate limiting',
            'requirements': 'Track API calls, enforce limits, queue requests, retry with backoff'
        },
        {
            'name': 'data_pipeline',
            'description': 'Process data through multiple transformation stages',
            'requirements': 'Stream large files, transform data, validate output, handle errors gracefully'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'='*60}")
        
        # Analyze requirements
        print("\n1. Requirement Analysis:")
        analysis = templates.analyze_tool_requirements(
            test_case['name'],
            test_case['description'],
            test_case['requirements']
        )
        
        print(f"   Category: {analysis['primary_category']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Operations: {', '.join(analysis['detected_operations'])}")
        print(f"   Complexity: {analysis['complexity_score']:.2f}")
        
        # Generate tool
        print("\n2. Generating Tool...")
        result = templates.generate_smart_tool(
            test_case['name'],
            test_case['description'],
            test_case['requirements']
        )
        
        if result['success']:
            print("   ✓ Generation successful!")
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence']:.2f}")
            
            # Get suggestions
            if 'code' in result:
                suggestions = templates.suggest_improvements(result['code'])
                if suggestions:
                    print("\n3. Improvement Suggestions:")
                    for s in suggestions[:3]:
                        print(f"   - [{s['priority']}] {s['message']}")
        else:
            print(f"   ✗ Generation failed: {result.get('error', 'Unknown')}")
    
    # Show overall insights
    print(f"\n{'='*60}")
    print("Overall Generation Insights:")
    print(f"{'='*60}")
    
    insights = templates.get_generation_insights()
    for key, value in insights.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_smart_generation())