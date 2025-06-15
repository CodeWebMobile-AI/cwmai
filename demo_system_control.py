#!/usr/bin/env python3
"""
Demo: Controlling the Continuous AI System through Conversation

Shows how the conversational AI can monitor and control the main system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def demo_system_control():
    """Demonstrate system control capabilities."""
    print("=== CWMAI System Control Demo ===\n")
    print("This demo shows how the conversational AI can control the continuous AI system.\n")
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Initialize
    try:
        await assistant.initialize()
    except:
        print("⚠️  Warning during initialization (this is normal)\n")
    
    # Demo conversations for system control
    conversations = [
        "Is the continuous AI system running?",
        "Can you check the health of the continuous AI?",
        "Start the continuous AI with 3 workers",
        "What's the status of the continuous AI now?",
        "Stop the continuous AI system please",
    ]
    
    for user_input in conversations:
        print(f"\n{'='*60}")
        print(f"👤 User: {user_input}")
        print(f"{'='*60}")
        
        response = await assistant.handle_conversation(user_input)
        
        print(f"🤖 Assistant: {response}")
        
        # Small delay between commands
        await asyncio.sleep(1)
    
    print("\n=== Demo Complete ===")
    print("\nThe assistant can now:")
    print("✅ Check if the continuous AI is running")
    print("✅ Start the system with custom parameters")
    print("✅ Monitor system health and performance")
    print("✅ Stop the system gracefully")
    print("\nAll through natural conversation!")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")
    
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    asyncio.run(demo_system_control())