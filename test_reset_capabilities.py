#!/usr/bin/env python3
"""
Test the smart reset capabilities of the conversational AI assistant.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def test_reset_capabilities():
    """Test various reset scenarios."""
    print("=== Testing Smart Reset Capabilities ===\n")
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Initialize (suppress warnings)
    try:
        await assistant.initialize()
    except:
        pass
    
    print("1. Testing Reset Analysis")
    print("-" * 40)
    analysis = await assistant.analyze_reset_need()
    print(f"Urgency: {analysis['urgency']}")
    print(f"Reasons: {', '.join(analysis['reasons'][:2]) if analysis['reasons'] else 'None'}")
    print(f"Recommended: {analysis['recommended_type']}")
    print()
    
    print("2. Testing Reset Recommendations")
    print("-" * 40)
    
    test_inputs = [
        "Reset the system",
        "Clean up everything but keep the knowledge",
        "The system seems broken",
        "Clear all the logs"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        recommendation = await assistant.recommend_reset_type(user_input)
        print(f"Reset Type: {recommendation['reset_type']}")
        print(f"What will happen: {recommendation['explanation'][:100]}...")
        if recommendation['warnings']:
            print(f"Warnings: {recommendation['warnings'][0]}")
    
    print("\n3. Testing Dry Run")
    print("-" * 40)
    
    # Do a dry run of logs reset
    from scripts.conversational_ai_assistant import ResetType
    result = await assistant.execute_system_reset(
        reset_type=ResetType.LOGS_ONLY,
        dry_run=True
    )
    
    if result['success']:
        print(f"Dry run successful!")
        print(f"Would delete {result['files_deleted']} files")
        print(f"Would free {result['space_freed'] / (1024*1024):.1f} MB")
        if result['files_to_delete']:
            print(f"Sample files: {', '.join(result['files_to_delete'][:3])}")
    
    print("\n4. Testing Natural Language Conversation")
    print("-" * 40)
    
    # Test a full conversation flow
    conversation = [
        "The system seems really slow and there are lots of old logs",
        "What kind of reset do you recommend?"
    ]
    
    for user_input in conversation:
        print(f"\nUser: {user_input}")
        response = await assistant.handle_conversation(user_input)
        # Truncate long responses
        if len(response) > 300:
            response = response[:300] + "..."
        print(f"Assistant: {response}")

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    asyncio.run(test_reset_capabilities())