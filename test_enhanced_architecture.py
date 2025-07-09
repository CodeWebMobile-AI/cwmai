#!/usr/bin/env python3
"""
Test the enhanced architecture documentation tool.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.custom_tools.documentation.enhance_architecture_simple import enhance_architecture_documentation


async def test_enhancement():
    """Test enhancing the micronutrient-optimizer architecture."""
    print("Testing architecture enhancement...")
    
    result = await enhance_architecture_documentation(
        project_path="/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/micronutrient-optimizer"
    )
    
    print(f"\nResult: {result}")
    
    if result['success'] and result['enhanced'] > 0:
        print("\n✅ Enhancement successful!")
        
        # Read and display a snippet of the enhanced file
        arch_path = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/micronutrient-optimizer/ARCHITECTURE.md"
        with open(arch_path, 'r') as f:
            content = f.read()
            
        print(f"\nEnhanced file size: {len(content)} characters")
        print(f"Number of lines: {len(content.splitlines())}")
        
        # Check for required sections
        required_sections = [
            "Admin Portal",
            "User Portal",
            "Admin API", 
            "User API",
            "Role-Based Access",
            "Admin Components",
            "User Components"
        ]
        
        found_sections = []
        missing_sections = []
        
        for section in required_sections:
            if section.lower() in content.lower():
                found_sections.append(section)
            else:
                missing_sections.append(section)
                
        print(f"\nFound sections: {', '.join(found_sections)}")
        if missing_sections:
            print(f"Missing sections: {', '.join(missing_sections)}")
            
        # Show first 100 lines
        print("\n--- First 100 lines of enhanced architecture ---")
        lines = content.splitlines()
        for i, line in enumerate(lines[:100], 1):
            print(f"{i:3d}: {line}")
            
    else:
        print("\n❌ Enhancement failed!")
        if result.get('details'):
            for project, details in result['details'].items():
                if details.get('error'):
                    print(f"Error in {project}: {details['error']}")


if __name__ == "__main__":
    # Set environment variables
    os.environ['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Run test
    asyncio.run(test_enhancement())