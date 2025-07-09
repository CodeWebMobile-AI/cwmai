"""
Create Tool Like Claude
=======================

Creates tools with Claude Code's intelligence - searches, reads, analyzes,
and generates working tools on the first try.
"""

from typing import Dict, Any, List, Optional
from scripts.claude_code_tool_creator import ClaudeCodeToolCreator

__description__ = "Create a new tool with Claude Code level intelligence by searching codebase and reusing patterns"
__parameters__ = {
    'name': {'type': 'string', 'required': True, 'description': 'Name for the new tool (snake_case)'},
    'description': {'type': 'string', 'required': True, 'description': 'What the tool should do'},
    'search_results': {'type': 'array', 'required': False, 'description': 'Previous grep_files results'},
    'file_contents': {'type': 'object', 'required': False, 'description': 'Previously read file contents'}
}
__examples__ = [
    {
        'name': 'check_architecture_quality',
        'description': 'Check if repositories have well-detailed ARCHITECTURE.md files with all required sections'
    },
    {
        'name': 'analyze_code_health',
        'description': 'Analyze code health metrics across all repositories'
    }
]

async def create_tool_like_claude(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a tool with Claude Code's intelligence.
    
    This tool:
    1. Searches the codebase for relevant patterns
    2. Intelligently selects which files to read
    3. Analyzes code relationships and dependencies
    4. Reuses existing code instead of reinventing
    5. Generates working tools on the first try
    """
    tool_name = kwargs.get('name')  # Changed from 'tool_name' to 'name'
    description = kwargs.get('description')
    search_results = kwargs.get('search_results')
    file_contents = kwargs.get('file_contents')
    
    if not tool_name or not description:
        return {
            "success": False,
            "error": "Both name and description are required"
        }
    
    try:
        # Initialize the Claude Code level creator
        creator = ClaudeCodeToolCreator()
        
        # Convert file contents dict to list of files if provided
        context_files = None
        if file_contents and isinstance(file_contents, dict):
            context_files = list(file_contents.keys())
        
        # Create the tool with full intelligence
        result = await creator.create_tool_from_request(
            tool_name=tool_name,
            description=description,
            grep_results=search_results,
            context_files=context_files
        )
        
        if result['success']:
            # Format the response to show what was done
            response = {
                "success": True,
                "tool_name": tool_name,
                "tool_path": result['tool_path'],
                "message": f"Successfully created '{tool_name}' with Claude Code intelligence",
                "intelligence_used": {
                    "files_analyzed": result.get('based_on_files', []),
                    "components_reused": result.get('reused_components', {}),
                    "validation": result.get('test_result', {})
                },
                "next_steps": [
                    f"Tool is ready to use: EXECUTE_TOOL: {tool_name}()",
                    "The tool reuses existing code patterns for reliability",
                    "All imports have been validated and fixed"
                ]
            }
            
            # If the tool is importable and working, that's a huge success
            test_result = result.get('test_result', {})
            if test_result.get('importable') and test_result.get('has_function'):
                response['quality'] = "PRODUCTION_READY"
                response['confidence'] = "HIGH"
            else:
                response['quality'] = "NEEDS_REVIEW"
                response['confidence'] = "MEDIUM"
            
            return response
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create tool: {str(e)}",
            "suggestion": "Try searching for more specific patterns first"
        }

# Export for tool system
create_tool_like_claude_tool = create_tool_like_claude