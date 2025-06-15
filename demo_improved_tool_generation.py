#!/usr/bin/env python3
"""
Demo: Improved Tool Generation System
Shows how the conversational AI handles abbreviations and tool creation intelligently
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant


async def demo_improved_system():
    """Demonstrate the improved tool generation capabilities."""
    print("🚀 CWMAI Improved Tool Generation Demo\n")
    print("This demo shows how the system now intelligently handles:")
    print("  ✓ Common abbreviations (reps, repos, cmds)")
    print("  ✓ Avoiding duplicate tool creation")
    print("  ✓ Smart tool generation only when needed\n")
    
    # Initialize assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Demo queries
    demos = [
        ("📊 Repository Count with Abbreviation", "how many reps are we managing?"),
        ("📋 Task Status", "what tasks do we have?"),
        ("💻 System Health", "check system health"),
        ("🔧 Worker Status", "how many workers are running?"),
        ("📈 Performance Check", "show me system performance metrics")
    ]
    
    for title, query in demos:
        print(f"\n{title}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        try:
            response = await assistant.handle_conversation(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n✨ Demo Complete!")
    print("\nKey Improvements:")
    print("1. 'reps' → automatically uses count_repositories")
    print("2. 'workers' → uses system_status instead of creating new tool")
    print("3. Natural language queries map to appropriate existing tools")
    print("4. New tools only created for genuinely missing functionality")


async def main():
    """Run the demo."""
    await demo_improved_system()


if __name__ == "__main__":
    asyncio.run(main())