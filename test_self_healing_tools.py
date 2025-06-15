#!/usr/bin/env python3
"""
Test the self-healing tool functionality of CWMAI
"""

import asyncio
from scripts.conversational_ai_assistant import ConversationalAIAssistant


async def test_self_healing():
    """Test that CWMAI can automatically fix broken tools."""
    print("Testing CWMAI Self-Healing Tool System")
    print("=" * 80)
    
    # Initialize the conversational AI
    assistant = ConversationalAIAssistant()
    
    # Test 1: Try to use the broken orchestrate_workflows tool
    print("\nTest 1: Using a broken tool (orchestrate_workflows)")
    print("-" * 40)
    
    response = await assistant.process_input("use orchestrate_workflows to list all workflows")
    print(f"Assistant Response:\n{response}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("\nThe system should have:")
    print("1. Detected the import/attribute error")
    print("2. Automatically deleted and regenerated the tool")
    print("3. Fixed the imports to use correct modules")
    print("4. Successfully executed the fixed tool")


if __name__ == "__main__":
    asyncio.run(test_self_healing())