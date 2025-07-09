#!/usr/bin/env python3
"""Verify the parsing fix works correctly."""

import ast
import re

def test_ast_parsing():
    """Test if our AST parsing approach works."""
    
    # Simulate the parameters string that would be extracted
    # This is what comes after 'create_new_tool('
    params_str = '''name='delete_tool', description='Deletes a tool from the system', requirements='import os
import json
from pathlib import Path

def delete_tool(tool_name):
    """
    Deletes a tool from the custom_tools directory.
    
    Args:
        tool_name: Name of the tool to delete
        
    Returns:
        dict: Success status and message
    """
    try:
        # Path to custom tools directory
        tools_dir = Path("scripts/custom_tools")
        tool_file = tools_dir / f"{tool_name}.py"
        
        if not tool_file.exists():
            return {"success": False, "error": f"Tool {tool_name} not found"}
            
        # Delete the tool file
        tool_file.unlink()
        
        return {"success": True, "message": f"Tool {tool_name} deleted successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}\'\'\'
    
    print("Testing AST parsing approach...")
    
    # Try Method 1: AST parsing
    try:
        # Create a function call and parse it
        tree = ast.parse(f"f({params_str})")
        print("✅ AST parsing succeeded!")
        
        # Extract parameters
        call = tree.body[0].value
        params = {}
        
        for keyword in call.keywords:
            if keyword.arg:
                if isinstance(keyword.value, ast.Str):
                    params[keyword.arg] = keyword.value.s
                elif isinstance(keyword.value, ast.Constant):
                    params[keyword.arg] = keyword.value.value
                    
        print("\nExtracted parameters:")
        for key, value in params.items():
            if key == 'requirements':
                print(f"  {key}: <code with {value.count(chr(10))} lines>")
                # Verify the code is complete
                if 'def delete_tool' in value and 'return {' in value:
                    print("  ✅ Requirements code looks complete")
            else:
                print(f"  {key}: {value}")
                
    except SyntaxError as e:
        print(f"❌ AST parsing failed: {e}")
        
        # Try Method 2: Regex-based extraction
        print("\nTrying regex-based extraction...")
        
        # Extract name
        name_match = re.search(r"name='([^']*)'", params_str)
        if name_match:
            print(f"  name: {name_match.group(1)}")
            
        # Extract description  
        desc_match = re.search(r"description='([^']*)'", params_str)
        if desc_match:
            print(f"  description: {desc_match.group(1)}")
            
        # Extract requirements (everything after requirements=' until the last ')
        req_match = re.search(r"requirements='(.*)'$", params_str, re.DOTALL)
        if req_match:
            requirements = req_match.group(1)
            print(f"  requirements: <code with {requirements.count(chr(10))} lines>")

if __name__ == "__main__":
    test_ast_parsing()