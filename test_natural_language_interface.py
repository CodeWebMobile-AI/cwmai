#!/usr/bin/env python3
"""
Test script for the Natural Language Interface

This script demonstrates various natural language commands that can be used
with the CWMAI system.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.natural_language_interface import NaturalLanguageInterface


async def test_commands():
    """Test various natural language commands."""
    nli = NaturalLanguageInterface()
    
    print("Testing CWMAI Natural Language Interface\n")
    
    # Initialize
    print("Initializing...")
    await nli.initialize()
    
    # Test commands
    test_inputs = [
        "show me the system status",
        "what tasks are active?",
        "search repositories for python machine learning",
        "create an issue for test-repo about implementing user authentication",
        "analyze performance",
        "create architecture for an e-commerce platform",
        "help",
        "help issue",
        "create task to review and update documentation",
        "how are we doing?"  # Should map to status
    ]
    
    for i, command in enumerate(test_inputs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {command}")
        print('='*60)
        
        # Parse command
        cmd_type, params = nli.parse_command(command)
        print(f"Parsed: command={cmd_type}, params={params}")
        
        # If not parsed, try AI interpretation
        if not cmd_type:
            print("No pattern match, would use AI interpretation...")
            # In real usage, this would call interpret_with_ai
            
        # Process command
        try:
            result = await nli.process_natural_language(command)
            print(f"Result: {result['status']} - {result['message']}")
            if result.get('data'):
                print(f"Data preview: {str(result['data'])[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
            
    # Cleanup
    await nli.close()
    print("\nTest completed!")


def test_pattern_matching():
    """Test the regex pattern matching."""
    nli = NaturalLanguageInterface()
    
    print("\nTesting Pattern Matching\n")
    
    test_cases = [
        # Create issue patterns
        ("create an issue for myrepo about adding dark mode", "create_issue"),
        ("make a new issue in test-repo for bug fixes", "create_issue"),
        ("open issue for project-x about memory leak", "create_issue"),
        
        # Search patterns
        ("search repositories for python AI", "search_repos"),
        ("find me some repos about machine learning", "search_repos"),
        ("look for repositories containing react", "search_repos"),
        
        # Architecture patterns
        ("create architecture for social media app", "create_architecture"),
        ("design system architecture for payment gateway", "create_architecture"),
        ("generate an architecture for microservices", "create_architecture"),
        
        # Status patterns
        ("show status", "show_status"),
        ("what's the system status?", "show_status"),
        ("how are things doing?", "show_status"),
        
        # Task patterns
        ("list active tasks", "list_tasks"),
        ("show me all tasks", "list_tasks"),
        ("what tasks are pending", "list_tasks"),
        
        # Performance patterns
        ("analyze performance", "analyze_performance"),
        ("show performance metrics", "analyze_performance"),
        ("how is the system performing?", "analyze_performance"),
        
        # Help patterns
        ("help", "help"),
        ("help issue", "help"),
        ("what commands do you support", "help"),
    ]
    
    passed = 0
    failed = 0
    
    for input_text, expected_cmd in test_cases:
        cmd_type, params = nli.parse_command(input_text)
        if cmd_type == expected_cmd:
            print(f"✓ PASS: '{input_text}' -> {cmd_type}")
            passed += 1
        else:
            print(f"✗ FAIL: '{input_text}' -> Expected: {expected_cmd}, Got: {cmd_type}")
            failed += 1
            
    print(f"\nPattern Matching Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    print("Running Natural Language Interface Tests\n")
    
    # Test pattern matching
    test_pattern_matching()
    
    # Test full command processing
    print("\n" + "="*60 + "\n")
    asyncio.run(test_commands())