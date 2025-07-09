#!/usr/bin/env python3
"""
Fix for Tool Call Parsing in Conversational AI Assistant

This script fixes the issue where multi-line string parameters in tool calls
are incorrectly parsed, causing parameter misparsing errors.

The solution improves the _execute_tool_call method to properly handle:
1. Multi-line strings with newlines
2. Escaped quotes within strings
3. Complex nested data structures
"""

import os
import re
import ast
from pathlib import Path

def create_improved_parser():
    """Create the improved parameter parsing code."""
    return r'''
    async def _execute_tool_call(self, tool_call_str: str) -> str:
        """Execute a tool call string like 'tool_name(param1="value1")'."""
        print(f"[DEBUG] _execute_tool_call called with: {tool_call_str}")
        try:
            # Parse tool name and parameters more robustly
            # Use DOTALL flag to handle multi-line strings
            match = re.match(r'(\w+)\((.*)\)$', tool_call_str, re.DOTALL)
            if not match:
                return f"Invalid tool call format: {tool_call_str}"

            tool_name = match.group(1)
            params_str = match.group(2).strip()

            # SECURITY: Parse parameters safely
            params = {}
            if params_str:
                try:
                    # Method 1: Try to parse as a Python function call using AST
                    # This handles complex cases including multi-line strings
                    try:
                        # Create a temporary function call and parse it
                        tree = ast.parse(f"f({params_str})")
                        call = tree.body[0].value
                        
                        # Extract keyword arguments
                        for keyword in call.keywords:
                            if keyword.arg:  # Skip **kwargs
                                # Safely evaluate the value
                                try:
                                    # For string literals, get the actual string value
                                    if isinstance(keyword.value, ast.Str):
                                        params[keyword.arg] = keyword.value.s
                                    elif isinstance(keyword.value, ast.Constant):
                                        params[keyword.arg] = keyword.value.value
                                    else:
                                        # For other types, compile and eval safely
                                        params[keyword.arg] = ast.literal_eval(
                                            compile(ast.Expression(keyword.value), '<string>', 'eval')
                                        )
                                except:
                                    # If we can't evaluate, convert to string
                                    params[keyword.arg] = ast.unparse(keyword.value)
                        
                        print(f"[DEBUG] Successfully parsed parameters using AST: {list(params.keys())}")
                        
                    except (SyntaxError, ValueError) as e:
                        print(f"[DEBUG] AST parsing failed: {e}, trying alternative method")
                        
                        # Method 2: Try to evaluate as dict literal
                        # First, try to convert function-style params to dict
                        dict_str = params_str
                        
                        # Handle key=value format by converting to dict format
                        # This regex handles quoted strings with proper nesting
                        def replace_param(match):
                            key = match.group(1)
                            value = match.group(2)
                            return f'"{key}": {value}'
                        
                        # More sophisticated regex that handles nested quotes and multi-line
                        param_pattern = r'(\w+)\s*=\s*("""[^"]*"""|\'\'\'[^\']*\'\'\'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\[[^\]]*\]|\{[^\}]*\}|[^,]+?)(?=\s*(?:,|$))'
                        dict_str = re.sub(param_pattern, replace_param, params_str, flags=re.DOTALL)
                        dict_str = "{" + dict_str + "}"
                        
                        # Clean up any trailing commas
                        dict_str = re.sub(r',\s*}', '}', dict_str)
                        
                        # Use ast.literal_eval for safe parsing
                        params = ast.literal_eval(dict_str)
                        print(f"[DEBUG] Successfully parsed parameters using dict conversion")

                except (ValueError, SyntaxError) as e:
                    # Method 3: Fallback to simple parsing for basic cases
                    print(f"[DEBUG] Parameter parsing with ast failed: {e}, using simple fallback")
                    log_system_message(f"Parameter parsing with ast.literal_eval failed: {e}, using fallback", "WARNING")
                    
                    # Simple regex for basic key=value pairs
                    param_matches = re.findall(r'(\w+)=["\']?([^,"\']+)["\']?', params_str)
                    for key, value in param_matches:
                        # Try to convert to appropriate type
                        if value.isdigit():
                            params[key] = int(value)
                        elif value.lower() in ['true', 'false']:
                            params[key] = value.lower() == 'true'
                        else:
                            params[key] = value.strip()
                    
                    print(f"[DEBUG] Fallback parsing found params: {params}")

            # Log the final parsed parameters
            print(f"[DEBUG] Final parsed parameters: {params}")
            
            # Enhance parameters with context from conversation
            # TEMPORARILY DISABLED to debug issue
            # enhanced_params = self.context_enhancer.enhance_tool_parameters(
            #     tool_name,
            #     params,
            #     self._get_recent_user_input()
            # )
            enhanced_params = params  # Use params directly

            # Execute the tool
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()

            # Debug logging
            print(f"[DEBUG] About to call tool '{tool_name}' with params: {enhanced_params}")
            log_system_message(f"Calling tool {tool_name} with params: {enhanced_params}")

            try:
                result = await tool_system.call_tool(tool_name, **enhanced_params)
                print(f"[DEBUG] Tool call completed, result type: {type(result)}")
            except Exception as e:
                print(f"[DEBUG] Tool call exception: {type(e).__name__}: {e}")
                log_system_message(f"Tool call exception: {e}", "ERROR")
                # Re-raise with more context
                raise Exception(f"Tool {tool_name} failed with: {str(e)}")

            # Format the result with type checking
            if isinstance(result, dict):
                if result.get('success'):
                    return self._format_tool_result(result.get('result'), tool_name)
                elif result.get('error') and 'not found' in result.get('error'):
                    # Tool doesn't exist, offer to create it
                    return await self._handle_missing_tool(tool_name, params)
                else:
                    error_msg = result.get('error', '')
                    return f"Error executing {tool_name}: {error_msg}"
            else:
                # Result is not a dict - convert to string for error handling
                result_str = str(result)
                print(f"[DEBUG] Tool result is not a dict: {type(result)} - {result_str[:100]}")
                if 'not found' in result_str:
                    return await self._handle_missing_tool(tool_name, params)
                else:
                    return f"Unexpected tool result format: {result_str}"

        except Exception as e:
            self.logger.error(f"Error executing tool call: {e}")
            return f"I encountered an error executing the tool: {str(e)}"
'''

def apply_fix():
    """Apply the fix to the conversational AI assistant."""
    assistant_path = Path("scripts/conversational_ai_assistant.py")
    
    # Read the current file
    with open(assistant_path, 'r') as f:
        original_content = f.read()
    
    # Backup the original first
    backup_path = assistant_path.with_suffix('.py.backup_tool_parsing')
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    # Find the _execute_tool_call method and replace it
    pattern = r'(    async def _execute_tool_call\(self, tool_call_str: str\) -> str:.*?)(\n    async def \w+|\n    def \w+|\Z)'
    
    # Get improved code
    improved_code = create_improved_parser()
    
    # Use a replacement function to avoid escaping issues
    def replacer(match):
        return improved_code.strip() + match.group(2)
    
    # Replace with improved version
    content = re.sub(pattern, replacer, original_content, flags=re.DOTALL)
    
    # Write the fixed version
    with open(assistant_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Applied tool call parsing fix to {assistant_path}")
    print(f"📁 Backup saved to {backup_path}")
    
    # Also create a test script
    create_test_script()

def create_test_script():
    """Create a test script to verify the fix works."""
    test_content = '''#!/usr/bin/env python3
"""Test script to verify tool call parsing fix."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.conversational_ai_assistant import ConversationalAIAssistant

async def test_multi_line_parsing():
    """Test parsing of multi-line string parameters."""
    assistant = ConversationalAIAssistant()
    
    # Test case 1: Simple multi-line string
    test_call1 = 'create_new_tool(name="test_tool", description="A test tool", requirements="line1\\nline2\\nline3")'
    print("Test 1 - Simple multi-line:")
    print(f"Input: {test_call1}")
    result = await assistant._execute_tool_call(test_call1)
    print(f"Result: {result[:200]}...")
    print()
    
    # Test case 2: Complex multi-line with quotes
    test_call2 = """create_new_tool(name='delete_tool', description='Deletes a tool', requirements='
import os
import json

def delete_tool(tool_name):
    """
    Deletes a tool.
    """
    return {"success": True}
')"""
    print("Test 2 - Complex multi-line with docstring:")
    print(f"Input: {test_call2[:100]}...")
    result = await assistant._execute_tool_call(test_call2)
    print(f"Result: {result[:200]}...")
    print()
    
    # Test case 3: Escaped quotes
    test_call3 = 'test_tool(message="This has \\"escaped quotes\\" inside")'
    print("Test 3 - Escaped quotes:")
    print(f"Input: {test_call3}")
    result = await assistant._execute_tool_call(test_call3)
    print(f"Result: {result[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_multi_line_parsing())
'''
    
    test_path = Path("test_tool_parsing_fix.py")
    with open(test_path, 'w') as f:
        f.write(test_content)
    os.chmod(test_path, 0o755)
    
    print(f"📝 Created test script: {test_path}")

if __name__ == "__main__":
    print("🔧 Applying tool call parsing fix...")
    apply_fix()
    print("\n✨ Fix applied successfully!")
    print("\nYou can test the fix with: python test_tool_parsing_fix.py")