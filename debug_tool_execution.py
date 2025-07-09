#!/usr/bin/env python3
"""Debug script to test tool execution and regex pattern matching"""

import re
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.ai_brain import AIBrain

def test_regex_patterns():
    """Test various regex patterns for tool matching"""
    print("=== Testing Regex Patterns ===\n")
    
    # Define test cases with expected tool calls
    test_cases = [
        # Standard XML format
        {
            "name": "Standard XML format",
            "text": """I'll create a test tool for you.

<tool_call>
<tool_name>create_test_tool</tool_name>
<parameters>
{
    "name": "test_tool",
    "description": "A test tool"
}
</parameters>
</tool_call>

Tool created successfully!""",
            "expected_tools": ["create_test_tool"]
        },
        
        # Multiple tools
        {
            "name": "Multiple tools",
            "text": """Let me check the system status first, then list tasks.

<tool_call>
<tool_name>check_system_status</tool_name>
<parameters>{}</parameters>
</tool_call>

Now listing tasks:

<tool_call>
<tool_name>list_tasks</tool_name>
<parameters>{"status": "pending"}</parameters>
</tool_call>""",
            "expected_tools": ["check_system_status", "list_tasks"]
        },
        
        # Markdown code block format
        {
            "name": "Markdown code block",
            "text": """I'll analyze the repository for you.

```tool_call
tool_name: analyze_repository
parameters:
  path: /home/user/project
  recursive: true
```

Analysis complete.""",
            "expected_tools": ["analyze_repository"]
        },
        
        # JSON format
        {
            "name": "JSON format", 
            "text": """Executing tool now:

{
    "tool_call": {
        "tool_name": "count_files",
        "parameters": {
            "directory": "/tmp",
            "pattern": "*.txt"
        }
    }
}

Found 5 files.""",
            "expected_tools": ["count_files"]
        },
        
        # No tools
        {
            "name": "No tools (plain text)",
            "text": "This is just a regular response without any tool calls.",
            "expected_tools": []
        }
    ]
    
    # Test regex patterns from the assistant
    patterns = [
        # XML pattern
        (r'<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>', "XML"),
        # Markdown pattern
        (r'```tool_call\s*tool_name:\s*(\w+)\s*parameters:\s*(.*?)```', "Markdown"),
        # JSON pattern
        (r'\{\s*"tool_call":\s*\{\s*"tool_name":\s*"(\w+)",\s*"parameters":\s*(\{.*?\})\s*\}\s*\}', "JSON"),
    ]
    
    for test_case in test_cases:
        print(f"Test Case: {test_case['name']}")
        print(f"Expected tools: {test_case['expected_tools']}")
        print(f"Text sample: {test_case['text'][:100]}...")
        print()
        
        found_tools = []
        for pattern, pattern_name in patterns:
            matches = re.findall(pattern, test_case['text'], re.DOTALL | re.IGNORECASE)
            if matches:
                print(f"  ✓ {pattern_name} pattern matched!")
                for match in matches:
                    tool_name = match[0] if isinstance(match, tuple) else match
                    found_tools.append(tool_name.strip())
                    print(f"    - Found tool: {tool_name.strip()}")
        
        if not found_tools and test_case['expected_tools']:
            print(f"  ✗ No patterns matched!")
        
        print(f"  Found tools: {found_tools}")
        print(f"  Match: {'✓' if set(found_tools) == set(test_case['expected_tools']) else '✗'}")
        print("-" * 50)
        print()

def test_process_ai_response():
    """Test the _process_ai_response method directly"""
    print("\n=== Testing _process_ai_response Method ===\n")
    
    # Create assistant instance
    assistant = ConversationalAIAssistant()
    
    # Test responses
    test_responses = [
        {
            "name": "Simple tool call",
            "response": """I'll check the system status for you.

<tool_call>
<tool_name>check_system_status</tool_name>
<parameters>{}</parameters>
</tool_call>

The system is running normally."""
        },
        {
            "name": "Tool with parameters",
            "response": """Let me count the Python files in the project.

<tool_call>
<tool_name>count_python_files</tool_name>
<parameters>{"directory": "/Users/fortressarigo/PhpstormProjects/cwmai"}</parameters>
</tool_call>

I found the files."""
        },
        {
            "name": "Invalid tool format",
            "response": """I'll try to call a tool but with wrong format:

<tool_call>
check_system_status
</tool_call>"""
        }
    ]
    
    for test in test_responses:
        print(f"Test: {test['name']}")
        print(f"Response: {test['response'][:100]}...")
        
        try:
            # Test the _process_ai_response method
            result = assistant._process_ai_response(test['response'])
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        
        print("-" * 50)
        print()

def test_ai_brain_tool_extraction():
    """Test how AIBrain extracts tool calls"""
    print("\n=== Testing AIBrain Tool Extraction ===\n")
    
    # Create AIBrain instance
    brain = AIBrain()
    
    # Test the tool call extraction regex
    tool_call_pattern = re.compile(
        r'<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>',
        re.DOTALL | re.IGNORECASE
    )
    
    test_text = """I'll analyze the code for you.

<tool_call>
<tool_name>analyze_code_complexity</tool_name>
<parameters>
{
    "file_path": "/Users/fortressarigo/PhpstormProjects/cwmai/scripts/ai_brain.py",
    "include_metrics": true
}
</parameters>
</tool_call>

The analysis is complete."""
    
    print("Test text:")
    print(test_text)
    print()
    
    matches = tool_call_pattern.findall(test_text)
    print(f"Found {len(matches)} tool call(s)")
    
    for i, (tool_name, params) in enumerate(matches):
        print(f"\nMatch {i+1}:")
        print(f"  Tool name: '{tool_name.strip()}'")
        print(f"  Parameters: '{params.strip()}'")
        
        # Try to parse parameters as JSON
        try:
            parsed_params = json.loads(params.strip())
            print(f"  Parsed params: {parsed_params}")
        except json.JSONDecodeError as e:
            print(f"  Failed to parse params as JSON: {e}")

def test_actual_tool_execution():
    """Test actual tool execution with a real assistant instance"""
    print("\n=== Testing Actual Tool Execution ===\n")
    
    # Export environment variables
    os.system('export $(cat .env.local | grep -v "^#" | xargs)')
    
    try:
        # Create assistant
        assistant = ConversationalAIAssistant()
        
        # Test with a simple tool call
        test_query = "Can you check the system status?"
        
        print(f"Query: {test_query}")
        print("Processing...")
        
        response = assistant.process_query(test_query)
        print(f"\nResponse: {response}")
        
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debug tests"""
    print("Tool Execution Debug Script")
    print("=" * 60)
    
    # Run tests
    test_regex_patterns()
    test_process_ai_response()
    test_ai_brain_tool_extraction()
    test_actual_tool_execution()
    
    print("\n" + "=" * 60)
    print("Debug testing complete!")

if __name__ == "__main__":
    main()