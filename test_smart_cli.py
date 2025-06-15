#!/usr/bin/env python3
"""
Test Smart CLI functionality
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.smart_natural_language_interface import SmartNaturalLanguageInterface, CommandConfidence
from scripts.ai_brain import IntelligentAIBrain


async def test_smart_interface():
    """Test the smart natural language interface."""
    print("🧪 Testing Smart Natural Language Interface\n")
    
    # Create interface
    ai_brain = IntelligentAIBrain(enable_round_robin=True)
    interface = SmartNaturalLanguageInterface(
        ai_brain=ai_brain,
        enable_learning=True,
        enable_multi_model=False  # Single model for testing
    )
    
    # Initialize
    await interface.initialize()
    
    # Test cases
    test_cases = [
        # Basic commands
        "show me the system status",
        "create an issue for auth-api about users reporting slow login times",
        "search for AI code review tools",
        "generate architecture for a real-time collaboration platform",
        
        # Context-aware commands
        "create another issue about performance",
        
        # Complex commands
        "find repositories about machine learning and then create a comparison task",
        
        # Natural variations
        "what's trending in developer tools?",
        "make a bug report for the payment service saying transactions are failing",
        "I need an architecture for something like Netflix but for educational content",
        
        # Unclear commands (should trigger clarification)
        "do the thing with the stuff",
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_input}")
        print('='*60)
        
        try:
            result = await interface.process_input(test_input)
            
            # Display results
            print(f"\nAction: {result.get('action', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            
            if result.get('explanation'):
                print(f"Explanation: {result['explanation']}")
            
            if result.get('suggestions'):
                print(f"Suggestions: {result['suggestions']}")
            
            if result.get('clarification_needed'):
                print("Clarification needed:")
                for q in result.get('questions', []):
                    print(f"  - {q}")
            
            # Show confidence if available
            if 'intent' in result:
                print(f"Confidence: {result['intent'].confidence.value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All tests completed!")
    
    # Test pattern matching
    print("\n🧪 Testing Pattern Matching\n")
    
    pattern_tests = [
        ("create issue for myrepo about bug", "create_issue"),
        ("search for python libraries", "search"),
        ("generate architecture for chat app", "architecture"),
        ("unknown gibberish command", None)
    ]
    
    for input_text, expected_action in pattern_tests:
        intent = interface._match_patterns(input_text)
        if intent:
            print(f"✅ '{input_text}' → {intent.action} (confidence: {intent.confidence.value})")
            assert expected_action is None or intent.action == expected_action
        else:
            print(f"❓ '{input_text}' → No pattern match")
            assert expected_action is None
    
    # Test learning
    if interface.enable_learning:
        print("\n🧪 Testing Learning System\n")
        
        # Simulate repeated commands
        for _ in range(3):
            await interface.process_input("create issue for test-repo about testing")
        
        # Check if pattern was learned
        patterns = interface.context.command_patterns
        print(f"Learned patterns: {patterns}")
        assert len(patterns) > 0, "Learning system should track patterns"
    
    print("\n✅ All tests passed!")


async def test_multi_model_consensus():
    """Test multi-model consensus (if keys available)."""
    print("\n🧪 Testing Multi-Model Consensus\n")
    
    interface = SmartNaturalLanguageInterface(
        enable_learning=False,
        enable_multi_model=True
    )
    
    await interface.initialize()
    
    if len(interface.ai_models) > 1:
        print(f"✅ Multi-model available: {list(interface.ai_models.keys())}")
        
        # Test consensus
        result = await interface.process_input("create a sophisticated AI-powered debugging tool")
        print(f"Consensus result: {result.get('action')}")
    else:
        print("⚠️ Multi-model not available (missing API keys)")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════╗
║     Smart CWMAI CLI Test Suite             ║
╚════════════════════════════════════════════╝
""")
    
    # Run tests
    asyncio.run(test_smart_interface())
    
    # Optional: test multi-model if keys available
    if os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY'):
        asyncio.run(test_multi_model_consensus())
    
    print("\n🎉 All tests completed successfully!")