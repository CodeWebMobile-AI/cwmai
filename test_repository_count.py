#!/usr/bin/env python3
"""
Test script to verify the repository count is correctly reported by the tool calling system.
"""

import asyncio
import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from tool_calling_system import ToolCallingSystem


async def main():
    """Test the count_repositories tool."""
    print("Testing repository count reporting...")
    print("-" * 50)
    
    # Initialize tool calling system
    tool_system = ToolCallingSystem()
    
    # Call the count_repositories tool
    print("\nCalling count_repositories tool...")
    result = await tool_system.call_tool('count_repositories')
    
    if result.get('success'):
        repo_data = result['result']
        print(f"\n✅ Repository count successfully retrieved!")
        print(f"\nTotal repositories: {repo_data['total']}")
        print(f"\nBreakdown by status:")
        for status, count in repo_data['breakdown']['by_status'].items():
            print(f"  - {status}: {count}")
        
        print(f"\nBreakdown by language:")
        for language, count in repo_data['breakdown']['by_language'].items():
            print(f"  - {language}: {count}")
        
        print(f"\nMetrics:")
        print(f"  - Total stars: {repo_data['metrics']['total_stars']}")
        print(f"  - Total open issues: {repo_data['metrics']['total_open_issues']}")
        print(f"  - Average stars per repo: {repo_data['metrics']['avg_stars_per_repo']:.2f}")
        
        print(f"\nSummary: {repo_data['summary']}")
        
        # Verify the count is correct
        if repo_data['total'] == 12:
            print("\n✅ SUCCESS: The conversational AI is now correctly reporting 12 repositories!")
        else:
            print(f"\n⚠️  WARNING: Expected 12 repositories, but got {repo_data['total']}")
    else:
        print(f"\n❌ ERROR calling tool: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())