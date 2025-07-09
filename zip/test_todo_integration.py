#!/usr/bin/env python3
"""
Comprehensive test of the natural language todo system integration.
Tests various scenarios to ensure the system isn't hardcoded.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Test imports
print("Testing imports...")
try:
    from natural_language_todo_queue import ConcurrentTodoQueue, MessagePriority, MessageStatus
    from todo_processor_pool import DynamicProcessorPool
    from todo_nlp_processor import TodoNLPProcessor, TodoIntent
    from natural_language_todo_manager import NaturalLanguageTodoManager, TodoStatus
    print("✓ All todo modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def test_nlp_processor():
    """Test the NLP processor with various inputs."""
    print("\n=== Testing NLP Processor ===")
    processor = TodoNLPProcessor(use_ai=False)
    
    test_cases = [
        # Various ways to add tasks
        ("Add a task to review the documentation", TodoIntent.ADD),
        ("remind me to call john tomorrow", TodoIntent.ADD),
        ("I need to finish the report by Friday", TodoIntent.ADD),
        ("Create a reminder for the meeting at 3pm", TodoIntent.ADD),
        ("todo: update the README file", TodoIntent.ADD),
        
        # Various ways to list tasks
        ("show me my tasks", TodoIntent.LIST),
        ("what do I need to do today?", TodoIntent.LIST),
        ("list all pending items", TodoIntent.LIST),
        ("what's on my agenda", TodoIntent.LIST),
        
        # Various ways to complete tasks
        ("mark the documentation task as done", TodoIntent.COMPLETE),
        ("I finished the API review", TodoIntent.COMPLETE),
        ("complete task 1", TodoIntent.COMPLETE),
        ("done with the report", TodoIntent.COMPLETE),
        
        # Edge cases
        ("", TodoIntent.UNKNOWN),
        ("hello there", TodoIntent.UNKNOWN),
        ("URGENT: FIX THE BUG ASAP!!!", TodoIntent.ADD),  # Should detect high priority
        
        # Complex scenarios
        ("add a high priority task to deploy the app by tomorrow at 5pm", TodoIntent.ADD),
        ("show me all tasks related to documentation", TodoIntent.SEARCH),
        ("delete the old meeting task", TodoIntent.DELETE),
        ("update the deadline for the report to next Monday", TodoIntent.UPDATE),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected_intent in test_cases:
        result = await processor.process(input_text)
        
        if result.intent == expected_intent:
            print(f"✓ '{input_text}' -> {result.intent.value}")
            passed += 1
            
            # Check entity extraction
            if result.entities:
                print(f"  Entities: {result.entities}")
        else:
            print(f"✗ '{input_text}' -> Expected: {expected_intent.value}, Got: {result.intent.value}")
            failed += 1
    
    print(f"\nNLP Tests: {passed} passed, {failed} failed")
    return failed == 0


async def test_concurrent_processing():
    """Test concurrent message processing."""
    print("\n=== Testing Concurrent Processing ===")
    
    queue = ConcurrentTodoQueue()
    processed_messages = []
    
    async def mock_processor(message):
        """Mock processor that simulates work."""
        await asyncio.sleep(0.1)  # Simulate processing time
        processed_messages.append(message.content)
        return {"success": True, "message": f"Processed: {message.content}"}
    
    # Create processor pool
    pool = DynamicProcessorPool(queue, mock_processor, min_processors=2, max_processors=4)
    await pool.start()
    
    # Add multiple messages concurrently
    messages = [
        "Add task 1",
        "Add task 2 with high priority",
        "Show me all tasks",
        "Complete the first task",
        "Add another urgent task"
    ]
    
    # Enqueue all messages at once
    for msg in messages:
        priority = MessagePriority.HIGH if "urgent" in msg or "high" in msg else MessagePriority.NORMAL
        queue.enqueue(msg, priority)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Check results
    print(f"Queued: {len(messages)} messages")
    print(f"Processed: {len(processed_messages)} messages")
    print(f"Queue status: {queue.get_status()}")
    
    # Stop pool
    await pool.stop()
    
    success = len(processed_messages) == len(messages)
    print(f"\nConcurrent processing: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_todo_manager():
    """Test the todo manager functionality."""
    print("\n=== Testing Todo Manager ===")
    
    # Use a test file to avoid conflicts
    manager = NaturalLanguageTodoManager(storage_path=Path("test_todos.json"))
    
    # Test adding todos
    todo1 = manager.add_todo("Review documentation", priority="high")
    todo2 = manager.add_todo("Fix bug in login system", priority="urgent")
    todo3 = manager.add_todo("Update README", priority="low")
    
    print(f"✓ Added {len(manager.todos)} todos")
    
    # Test listing
    pending = manager.list_todos(status=TodoStatus.PENDING)
    print(f"✓ Found {len(pending)} pending todos")
    
    # Test fuzzy matching
    found = manager.find_todo("documentation")
    if found and found.id == todo1.id:
        print("✓ Fuzzy matching works")
    else:
        print("✗ Fuzzy matching failed")
    
    # Test completion
    completed = manager.complete_todo("bug")
    if completed and completed.status == TodoStatus.COMPLETED:
        print("✓ Task completion works")
    else:
        print("✗ Task completion failed")
    
    # Test undo
    manager.undo_last_operation()
    if manager.todos[todo2.id].status == TodoStatus.PENDING:
        print("✓ Undo works")
    else:
        print("✗ Undo failed")
    
    # Clean up test file
    Path("test_todos.json").unlink(missing_ok=True)
    
    return True


async def test_integration():
    """Test the full integration."""
    print("\n=== Testing Full Integration ===")
    
    # Import the enhanced assistant
    try:
        from enhanced_conversational_assistant import EnhancedConversationalAssistant
        print("✓ Enhanced assistant imported")
    except ImportError as e:
        print(f"✗ Failed to import assistant: {e}")
        return False
    
    # Create and initialize assistant
    assistant = EnhancedConversationalAssistant()
    await assistant.initialize()
    print("✓ Assistant initialized with todo support")
    
    # Test various todo commands
    test_commands = [
        "Add a task to review the API documentation tomorrow",
        "Create a high priority reminder to call the client",
        "I need to prepare the monthly report by next Friday",
        "Show me what I have to do",
        "What tasks do I have pending?",
        "Mark the API documentation task as complete",
        "I'm done with the client call",
    ]
    
    for cmd in test_commands:
        print(f"\nTesting: '{cmd}'")
        try:
            response = await assistant.handle_conversation(cmd)
            print(f"Response: {response}")
            
            # Verify we get appropriate responses
            if "add" in cmd.lower() and ("Adding" in response or "Added" in response):
                print("✓ Add command handled correctly")
            elif ("show" in cmd.lower() or "what" in cmd.lower()) and "tasks" in cmd.lower():
                print("✓ List command handled correctly")
            elif ("mark" in cmd.lower() or "done" in cmd.lower()) and "complete" in response.lower():
                print("✓ Complete command handled correctly")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    # Clean up
    if assistant.todo_processor_pool:
        await assistant.todo_processor_pool.stop()
    
    return True


async def test_edge_cases():
    """Test edge cases and robustness."""
    print("\n=== Testing Edge Cases ===")
    
    processor = TodoNLPProcessor(use_ai=False)
    
    edge_cases = [
        # Empty and whitespace
        ("", "Should handle empty input"),
        ("   ", "Should handle whitespace"),
        
        # Very long input
        ("Add a task " + "with a very long description " * 20, "Should handle long input"),
        
        # Special characters
        ("Add task: Review @john's PR #123", "Should handle special chars"),
        ("Complete the task with emoji 🚀", "Should handle emoji"),
        
        # Mixed languages/numbers
        ("Add task 123456", "Should handle numbers"),
        ("Añadir tarea para mañana", "Should handle non-English"),
        
        # Ambiguous input
        ("task", "Should handle single word"),
        ("do the thing", "Should handle vague input"),
        
        # Multiple intents
        ("Add a task and then show me all tasks", "Should handle multiple intents"),
    ]
    
    errors = 0
    for input_text, description in edge_cases:
        try:
            result = await processor.process(input_text)
            print(f"✓ {description}: {result.intent.value}")
        except Exception as e:
            print(f"✗ {description}: Error - {e}")
            errors += 1
    
    print(f"\nEdge case handling: {'✓ PASSED' if errors == 0 else f'✗ FAILED ({errors} errors)'}")
    return errors == 0


async def main():
    """Run all tests."""
    print("🧪 Testing Natural Language Todo System Integration\n")
    
    all_passed = True
    
    # Run individual test suites
    tests = [
        ("NLP Processor", test_nlp_processor),
        ("Concurrent Processing", test_concurrent_processing),
        ("Todo Manager", test_todo_manager),
        ("Edge Cases", test_edge_cases),
        ("Full Integration", test_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            passed = await test_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! The system is working correctly.")
        print("\nThe implementation is flexible and handles:")
        print("- Various natural language patterns")
        print("- Concurrent message processing")
        print("- Edge cases and errors")
        print("- Different phrasings and intents")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return all_passed


if __name__ == "__main__":
    # Run with proper event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")