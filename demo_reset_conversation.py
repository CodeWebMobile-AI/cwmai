#!/usr/bin/env python3
"""
Demo: Reset functionality in natural conversation
"""

import asyncio
import sys
import os
from pathlib import Path

# Setup
sys.path.insert(0, '.')
os.environ.setdefault('ANTHROPIC_API_KEY', 'dummy_key_for_testing')

from scripts.conversational_ai_assistant import ConversationalAIAssistant, ResetType

async def demo_reset_conversation():
    """Show how reset works in conversation."""
    print("=== CWMAI Reset Conversation Demo ===\n")
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    
    # Test direct reset methods
    print("1. Testing System Analysis")
    print("-" * 40)
    analysis = await assistant.analyze_reset_need()
    print(f"Current system urgency: {analysis['urgency']}")
    if analysis['reasons']:
        print(f"Reasons: {', '.join(analysis['reasons'][:2])}")
    print()
    
    print("2. Reset Type Recommendations")
    print("-" * 40)
    
    # Test different user inputs
    test_cases = [
        ("Full reset", ResetType.FULL),
        ("Clear logs only", ResetType.LOGS_ONLY),
        ("Keep the knowledge base", ResetType.SELECTIVE)
    ]
    
    for desc, expected_type in test_cases:
        recommendation = await assistant.recommend_reset_type(desc)
        print(f"Input: '{desc}'")
        print(f"Recommended: {recommendation['type'].value}")
        print(f"Details: {recommendation['explanation'][:80]}...")
        print()
    
    print("3. Conversation Examples")
    print("-" * 40)
    
    # Simulate conversations
    conversations = [
        {
            "scenario": "User wants to clean up logs",
            "input": "The system has too many log files, can you clean them up?",
            "expected": "offer to clear log files"
        },
        {
            "scenario": "User reports system issues",
            "input": "The system seems broken and nothing is working",
            "expected": "analyze and recommend appropriate reset"
        },
        {
            "scenario": "User wants selective reset",
            "input": "Reset everything but keep the AI cache to save money",
            "expected": "recommend selective reset preserving cache"
        }
    ]
    
    for conv in conversations:
        print(f"\nScenario: {conv['scenario']}")
        print(f"User: {conv['input']}")
        print(f"Expected: {conv['expected']}")
        
        # Get recommendation
        rec = await assistant.recommend_reset_type(conv['input'])
        print(f"Assistant recommends: {rec['type'].value} reset")
        
        # Show what would happen (dry run)
        result = await assistant.execute_system_reset(rec['type'], dry_run=True)
        if result['success']:
            print(f"Would delete: {result['files_deleted']} files")
            print(f"Would free: {result['space_freed'] / (1024*1024):.1f} MB")
    
    print("\n4. Emergency Reset Example")
    print("-" * 40)
    
    print("User: 'Everything is stuck! Do an emergency reset!'")
    rec = await assistant.recommend_reset_type("Everything is stuck! Do an emergency reset!")
    print(f"Assistant: Preparing {rec['type'].value} reset...")
    
    if rec['warnings']:
        print(f"⚠️  Warning: {rec['warnings'][0]}")
    
    print("\n=== Demo Complete ===")
    print("\nThe assistant can handle various reset scenarios:")
    print("✅ Analyze system health and recommend resets")
    print("✅ Understand natural language reset requests")
    print("✅ Execute different types of resets safely")
    print("✅ Preserve important data when requested")

if __name__ == "__main__":
    asyncio.run(demo_reset_conversation())