#!/usr/bin/env python3
"""
Example: Natural conversation for controlling the continuous AI system
"""

import asyncio
import sys
import os

# Setup
sys.path.insert(0, '.')
os.environ.setdefault('ANTHROPIC_API_KEY', 'dummy_key_for_testing')

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def example_conversation():
    """Show how system control works in conversation."""
    print("=== CWMAI System Control Conversation Example ===\n")
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Check if continuous AI is running
    status = await assistant.check_continuous_ai_status()
    
    print("Current Status:")
    print(f"- Continuous AI Running: {status['running']}")
    if status['running']:
        print(f"- Process ID: {status.get('pid', 'Unknown')}")
        print(f"- Uptime: {status.get('uptime', 'Unknown')}")
    print("\n" + "="*50 + "\n")
    
    # Simulate conversation
    print("CONVERSATION EXAMPLE:\n")
    
    # User asks about status
    user_input = "Is the continuous AI system running?"
    print(f"👤 User: {user_input}")
    
    # Build response based on actual status
    if status['running']:
        response = f"""Let me check the continuous AI system status...

✅ Yes, the continuous AI system is currently running!
• Process ID: {status.get('pid', 'Unknown')}
• Uptime: {status.get('uptime', 'Unknown')}
• Status: Active and processing tasks

Would you like me to show more details about its performance?"""
    else:
        response = """Let me check the continuous AI system status...

❌ The continuous AI system is not currently running.

Would you like me to start it for you? I can launch it with default settings or you can specify the number of workers."""
    
    print(f"🤖 Assistant: {response}")
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Health check
    if status['running']:
        user_input = "How is the continuous AI performing?"
        print(f"👤 User: {user_input}")
        
        health = await assistant.monitor_system_health()
        
        response = f"""Let me analyze the continuous AI system performance...

📊 Continuous AI Health Report:
• Health Score: {health['health_score']}/100
• CPU Usage: {health.get('cpu_percent', 'N/A')}%
• Memory: {health.get('memory_mb', 'N/A')} MB
• Threads: {health.get('num_threads', 'N/A')}
"""
        
        if health['issues']:
            response += f"\n⚠️ Issues detected: {', '.join(health['issues'])}"
        
        if health['recommendations']:
            response += f"\n💡 Recommendations:\n"
            for rec in health['recommendations']:
                response += f"   • {rec}\n"
        
        print(f"🤖 Assistant: {response}")
    
    print("\n=== End of Example ===")
    print("\nThe assistant can handle various natural language requests:")
    print("- 'Start the continuous AI with 5 workers'")
    print("- 'Stop the continuous AI system'")
    print("- 'Restart the continuous AI'")
    print("- 'Check if the main system is healthy'")
    print("- And many more variations!")

if __name__ == "__main__":
    asyncio.run(example_conversation())