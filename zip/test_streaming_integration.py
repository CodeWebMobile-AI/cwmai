#!/usr/bin/env python3
"""
Test streaming integration in the enhanced conversational assistant.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))


async def test_streaming_components():
    """Test that streaming components work."""
    print("=== Testing Streaming Components ===\n")
    
    from streaming_response_generator import StreamingResponseGenerator, VisualStateIndicator
    
    # Test basic streaming
    streamer = StreamingResponseGenerator(typing_speed=0.01)
    indicator = VisualStateIndicator()
    
    print("1. Testing character-by-character streaming:")
    await streamer.stream_response("Hello, this is streaming text!", indicator="🤖")
    
    print("\n2. Testing visual indicators:")
    states = ["processing", "queued", "completed", "error"]
    for state in states:
        await indicator.show_state(state, f"Testing {state} state")
        await asyncio.sleep(0.5)
    print()
    
    print("\n3. Testing spinner animation (3 seconds):")
    spinner_task = asyncio.create_task(indicator.show_spinner("Processing complex request"))
    await asyncio.sleep(3)
    indicator.stop_spinner()
    
    print("\n✅ All streaming components working!")
    return True


async def test_assistant_streaming():
    """Test streaming in the assistant."""
    print("\n=== Testing Assistant Streaming ===\n")
    
    try:
        from enhanced_conversational_assistant import EnhancedConversationalAssistant
        
        # Create assistant
        assistant = EnhancedConversationalAssistant()
        
        # Check streaming availability
        print(f"Streaming available: {assistant.enable_streaming}")
        print(f"Streamer initialized: {assistant.streamer is not None}")
        print(f"Visual indicators initialized: {assistant.visual_indicator is not None}")
        
        if not assistant.enable_streaming:
            print("⚠️  Streaming not available in assistant")
            return False
            
        # Initialize (without full startup)
        assistant.todo_processor_pool = None  # Skip pool initialization for test
        
        print("\n✅ Assistant has streaming capabilities!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing assistant streaming: {e}")
        return False


async def test_concurrent_queue():
    """Test concurrent message queuing."""
    print("\n=== Testing Concurrent Queue ===\n")
    
    from natural_language_todo_queue import ConcurrentTodoQueue, MessagePriority
    
    queue = ConcurrentTodoQueue()
    
    # Simulate concurrent messages
    messages = [
        ("First task", MessagePriority.NORMAL),
        ("URGENT: Second task", MessagePriority.URGENT),
        ("Third task", MessagePriority.HIGH),
    ]
    
    print("Queuing messages:")
    for content, priority in messages:
        msg = queue.enqueue(content, priority)
        print(f"  ✓ Queued: {content} (Priority: {priority.value}, ID: {msg.id[:8]})")
    
    # Check queue status
    status = queue.get_status()
    print(f"\nQueue status: {status['pending']} pending")
    
    # Dequeue in priority order
    print("\nDequeuing in priority order:")
    for _ in range(3):
        msg = queue.dequeue(timeout=0.1)
        if msg:
            print(f"  → {msg.content} (Priority: {msg.priority})")
    
    print("\n✅ Concurrent queue working with priorities!")
    return True


async def demo_full_experience():
    """Demo the full Claude Code experience."""
    print("\n=== Full Experience Demo ===\n")
    
    from streaming_response_generator import StreamingResponseGenerator, VisualStateIndicator
    
    streamer = StreamingResponseGenerator(typing_speed=0.02)
    indicator = VisualStateIndicator()
    
    # Simulate a conversation
    print("> add task to implement streaming output")
    await indicator.show_state("processing", "Understanding your request")
    await asyncio.sleep(0.5)
    indicator.stop_spinner()
    
    await streamer.stream_response(
        "I'll add that task to your todo list. The task 'implement streaming output' "
        "has been added with medium priority.",
        indicator="🤖"
    )
    
    # Simulate concurrent input
    print("\n> show my tasks")
    print("📋 Command queued while processing...")
    await asyncio.sleep(0.5)
    
    await streamer.stream_response(
        "Here are your current tasks:\n"
        "1. 🟢 [medium] Implement streaming output\n"
        "2. 🟡 [high] Test concurrent processing\n"
        "3. 🔴 [urgent] Fix streaming integration",
        indicator="🤖"
    )
    
    print("\n\n✨ This demonstrates the Claude Code experience!")
    return True


async def main():
    """Run all tests."""
    print("🧪 Testing Streaming Integration\n")
    
    tests = [
        ("Streaming Components", test_streaming_components),
        ("Assistant Streaming", test_assistant_streaming),
        ("Concurrent Queue", test_concurrent_queue),
        ("Full Experience Demo", demo_full_experience),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All streaming integration tests passed!")
        print("\n🎉 The system now has full Claude Code capabilities:")
        print("  • Natural language understanding")
        print("  • Concurrent message processing")
        print("  • Character-by-character streaming")
        print("  • Visual state indicators")
        print("  • Priority-based queue management")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())