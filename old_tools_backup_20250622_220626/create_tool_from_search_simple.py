"""
Create Tool From Search (Simple)
================================

Creates new tools by searching the codebase for related functionality
and combining existing code patterns, similar to Claude Code's ability.

This is a simpler version that works with the tool calling system.
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from scripts.smart_tool_generator import SmartToolGenerator
from scripts.ai_brain import IntelligentAIBrain

__description__ = "Create a new tool by searching codebase for similar functionality and combining patterns"
__parameters__ = {
    'tool_name': {'type': 'string', 'required': True, 'description': 'Name for the new tool'},
    'description': {'type': 'string', 'required': True, 'description': 'What the tool should do'},
    'search_results': {'type': 'string', 'required': False, 'description': 'Previous search results to use as context'},
    'file_contents': {'type': 'string', 'required': False, 'description': 'File contents to use as patterns'}
}
__examples__ = [
    {
        'tool_name': 'check_architecture_quality',
        'description': 'Check if all repositories have detailed ARCHITECTURE.md files'
    }
]

async def create_tool_from_search_simple(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new tool based on search results and existing patterns.
    """
    tool_name = kwargs.get('tool_name')
    description = kwargs.get('description')
    search_results = kwargs.get('search_results', '')
    file_contents = kwargs.get('file_contents', '')
    
    project_root = Path(__file__).parent.parent.parent
    
    try:
        # If no context provided, search for it
        if not search_results and not file_contents:
            # Extract keywords from description
            keywords = _extract_keywords(description)
            
            # Search for related code
            search_results = await _search_for_patterns(keywords[0] if keywords else 'tool', project_root)
        
        # Use AI to generate the tool
        ai_brain = IntelligentAIBrain()
        
        # Build context from search results and file contents
        context = ""
        if search_results:
            context += f"Search results:\n{search_results}\n\n"
        if file_contents:
            context += f"Reference code:\n{file_contents}\n\n"
        
        prompt = f"""Create a Python tool with this exact structure:

Tool name: {tool_name}
Description: {description}

{context}

Generate a complete tool following this EXACT format:

```python
\"\"\"
{tool_name.replace('_', ' ').title()} Tool
{'=' * (len(tool_name) + 5)}

{description}
\"\"\"

import necessary_modules
from typing import Dict, Any

__description__ = "{description}"
__parameters__ = {{
    'param1': {{'type': 'string', 'required': True, 'description': 'Description'}}
}}
__examples__ = [
    {{'param1': 'example_value'}}
]

async def {tool_name}(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    {description}
    \"\"\"
    try:
        # Implementation based on patterns found
        
        return {{
            "success": True,
            "result": "result_data"
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}

# Export for tool system
{tool_name}_tool = {tool_name}
```

Base the implementation on the patterns found in the search results and reference code.
Make it practical and working, not just a placeholder."""

        response = await ai_brain.generate_enhanced_response(prompt)
        
        # Extract code from response
        code_match = re.search(r'```python\n(.*?)\n```', str(response), re.DOTALL)
        if not code_match:
            # Try without backticks
            code_match = re.search(r'(""".*?{}_tool = {})'.format(tool_name, tool_name), str(response), re.DOTALL)
        
        if not code_match:
            return {
                "success": False,
                "error": "Failed to generate valid tool code from AI response"
            }
        
        tool_code = code_match.group(1) if '```' in str(response) else code_match.group(0)
        
        # Save the tool
        tool_path = project_root / 'scripts' / 'custom_tools' / f'{tool_name}.py'
        
        with open(tool_path, 'w') as f:
            f.write(tool_code)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "tool_path": str(tool_path),
            "description": description,
            "message": f"Tool '{tool_name}' created successfully at {tool_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating tool: {str(e)}"
        }

async def _search_for_patterns(keyword: str, project_root: Path) -> str:
    """Search for code patterns related to keyword."""
    results = []
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'logs'}
    
    pattern = re.compile(keyword, re.IGNORECASE)
    
    for root, dirs, files in os.walk(project_root / 'scripts'):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files[:20]:  # Limit to prevent too much context
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if pattern.search(content):
                        # Get first match with context
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            if pattern.search(line):
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                context = '\n'.join(lines[start:end])
                                results.append(f"File: {file_path.name}\n{context}")
                                break
                except:
                    continue
        
        if len(results) > 5:  # Limit results
            break
    
    return '\n\n'.join(results)

def _extract_keywords(text: str) -> List[str]:
    """Extract keywords from description."""
    words = re.findall(r'\w+', text.lower())
    
    # Filter out common words
    skip_words = {'a', 'an', 'the', 'to', 'for', 'that', 'this', 'with', 'from', 
                  'should', 'would', 'could', 'check', 'create', 'make', 'if', 'all'}
    
    keywords = [w for w in words if w not in skip_words and len(w) > 3]
    
    return keywords[:3] if keywords else ['tool']

# Export for tool system
create_tool_from_search_simple_tool = create_tool_from_search_simple

if __name__ == "__main__":
    # Test the tool creation
    async def test():
        result = await create_tool_from_search_simple(
            tool_name="test_architecture_checker",
            description="Check if repositories have well-detailed ARCHITECTURE.md files"
        )
        print(f"Result: {result}")
    
    asyncio.run(test())