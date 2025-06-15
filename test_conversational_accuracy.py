#!/usr/bin/env python3
"""
Test Conversational AI Accuracy
Verifies that the conversational interface returns accurate information
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager


async def test_conversational_accuracy():
    """Test that conversational AI returns accurate data."""
    print("🧪 Testing Conversational AI Accuracy\n")
    
    # Initialize components
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    state_manager = StateManager()
    state = state_manager.load_state()
    
    # Test queries and expected results
    test_cases = [
        {
            "query": "how many repositories are we managing?",
            "expected_keywords": ["12", "repositories", "active"],
            "check_type": "repository_count"
        },
        {
            "query": "what is the system status?",
            "expected_keywords": ["healthy", "true", "system"],
            "check_type": "system_status"
        },
        {
            "query": "show me available commands",
            "expected_keywords": ["Available Commands", "22", "repository", "task", "ai"],
            "check_type": "command_list"
        },
        {
            "query": "count tasks",
            "expected_keywords": ["0 tasks", "Total 0 tasks"],
            "check_type": "task_count"
        },
        {
            "query": "check repository health",
            "expected_keywords": ["12 repositories", "healthy", "Health Check"],
            "check_type": "health_check"
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"📝 Testing: {test['query']}")
        
        # Get AI response
        response = await assistant.handle_conversation(test['query'])
        
        # Check if response contains expected information
        response_lower = response.lower()
        found_keywords = []
        missing_keywords = []
        
        for keyword in test['expected_keywords']:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Verify accuracy based on check type
        if test['check_type'] == 'repository_count':
            # Extract actual count from state
            actual_count = len(state.get('projects', {}))
            if str(actual_count) in response:
                print(f"  ✅ Correctly reported {actual_count} repositories")
                results.append(True)
            else:
                print(f"  ❌ Failed to report correct count ({actual_count})")
                results.append(False)
                
        elif missing_keywords:
            print(f"  ❌ Missing keywords: {missing_keywords}")
            results.append(False)
        else:
            print(f"  ✅ Response contains all expected information")
            results.append(True)
            
        print(f"  Response: {response[:200]}...")
        print()
    
    # Summary
    print("="*60)
    print("📊 CONVERSATIONAL AI ACCURACY SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(results)
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Accuracy: {(passed/total*100):.1f}%")
    
    return passed == total


async def main():
    """Run the tests."""
    success = await test_conversational_accuracy()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())