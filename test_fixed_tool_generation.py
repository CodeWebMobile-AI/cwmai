#!/usr/bin/env python3
"""
Test Fixed Tool Generation
Verify the autonomous tool creation now works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant


async def test_fixed_tool_generation():
    """Test that the fixed tool generation works correctly."""
    print("🧪 Testing Fixed Tool Generation\n")
    
    # First, remove the existing count_workers tool to force recreation
    count_workers_file = Path("scripts/custom_tools/count_workers.py")
    if count_workers_file.exists():
        count_workers_file.unlink()
        print("✓ Removed existing count_workers.py to test fresh generation\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test queries that should trigger tool creation
    test_queries = [
        "how many workers do we have?",
        "count the workers",
        "show worker count"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: '{query}'")
        print('='*60)
        
        try:
            response = await assistant.handle_conversation(query)
            print(f"✅ Response: {response}")
            
            # Check if tool was created
            if count_workers_file.exists():
                print(f"\n📋 Tool file created: {count_workers_file}")
                
                # Verify the generated code doesn't have main() function
                content = count_workers_file.read_text()
                if "def main(" in content or "async def main(" in content:
                    print("⚠️  WARNING: Generated tool contains main() function")
                else:
                    print("✅ Generated tool is clean (no main function)")
                    
                if "(self" in content and "def count_workers" in content:
                    print("❌ ERROR: Generated tool expects self parameter")
                else:
                    print("✅ Generated tool has correct signature")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎯 Test Summary")
    print("="*60)
    
    if count_workers_file.exists():
        print("✅ Tool generation works - count_workers.py was created")
        
        # Show a snippet of the generated function
        content = count_workers_file.read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'async def count_workers' in line:
                print(f"\n📝 Generated function signature:")
                print(f"   {line}")
                if i+1 < len(lines):
                    print(f"   {lines[i+1]}")
                break
    else:
        print("⚠️  No tool file was generated")


async def main():
    """Run the test."""
    await test_fixed_tool_generation()


if __name__ == "__main__":
    asyncio.run(main())