"""
List Tools - Safe Static Analysis Version
=========================================

Lists available tools without executing module code or triggering system initialization.
Uses AST (Abstract Syntax Tree) analysis to extract tool information statically.

This prevents the unintended side effects of importing modules that may:
- Initialize database connections
- Start background processes
- Load heavy AI models
- Trigger system-wide initialization

__description__ = "Safely list all available tools without executing their code"
__parameters__ = {
    'directories': {
        'type': 'list',
        'required': False,
        'description': 'Directories to search for tools (default: ["scripts/custom_tools"])'
    },
    'include_details': {
        'type': 'boolean',
        'required': False,
        'description': 'Include detailed information about each tool (default: False)'
    },
    'format': {
        'type': 'string',
        'required': False,
        'description': 'Output format: "simple", "detailed", or "grouped" (default: "simple")'
    }
}
"""

from typing import Dict, Any, List, Optional
import os
import ast
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SafeToolAnalyzer(ast.NodeVisitor):
    """Analyzes Python files to extract tool metadata without code execution."""
    
    def __init__(self):
        self.description = None
        self.parameters = {}
        self.examples = []
        self.tool_function = None
        self.is_tool = False
        
    def visit_Module(self, node):
        """Extract module docstring if present."""
        docstring = ast.get_docstring(node)
        if docstring:
            # First line is often the best description
            self.description = docstring.strip().split('\n')[0]
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Extract __description__, __parameters__, and __examples__ assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == '__description__' and isinstance(node.value, ast.Str):
                    self.description = node.value.s
                    self.is_tool = True
                elif target.id == '__parameters__':
                    self.parameters = self._extract_dict_safely(node.value)
                    self.is_tool = True
                elif target.id == '__examples__':
                    self.examples = self._extract_list_safely(node.value)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        """Identify main tool function."""
        # Common patterns for tool functions
        if node.name in ['execute', 'run', 'process', 'main'] or node.name.endswith('_tool'):
            self.tool_function = node.name
            self.is_tool = True
        # Also check if function name matches file name (common pattern)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Identify main tool function (non-async)."""
        if node.name in ['execute', 'run', 'process', 'main'] or node.name.endswith('_tool'):
            self.tool_function = node.name
            self.is_tool = True
        self.generic_visit(node)
        
    def _extract_dict_safely(self, node):
        """Safely extract dictionary literal from AST."""
        if not isinstance(node, ast.Dict):
            return {}
            
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Str):
                key_str = key.s
                if isinstance(value, ast.Str):
                    result[key_str] = value.s
                elif isinstance(value, ast.Dict):
                    result[key_str] = self._extract_dict_safely(value)
                else:
                    # Try to safely evaluate simple literals
                    try:
                        result[key_str] = ast.literal_eval(value)
                    except:
                        result[key_str] = "<complex value>"
        return result
        
    def _extract_list_safely(self, node):
        """Safely extract list literal from AST."""
        if not isinstance(node, ast.List):
            return []
            
        result = []
        for item in node.elts:
            try:
                result.append(ast.literal_eval(item))
            except:
                result.append("<complex value>")
        return result


def analyze_tool_safely(filepath: str) -> Optional[Dict[str, Any]]:
    """Analyze a single tool file without executing it."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Skip if file is too large (likely not a tool)
        if len(content) > 100000:  # 100KB limit
            return None
            
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.debug(f"Syntax error in {filepath}, skipping")
            return None
            
        # Analyze
        analyzer = SafeToolAnalyzer()
        analyzer.visit(tree)
        
        # Skip if doesn't look like a tool
        if not analyzer.is_tool and not analyzer.tool_function:
            return None
            
        # Build tool info
        tool_name = Path(filepath).stem
        return {
            'name': tool_name,
            'description': analyzer.description or f"Tool: {tool_name}",
            'parameters': analyzer.parameters,
            'examples': analyzer.examples,
            'function': analyzer.tool_function,
            'path': filepath
        }
        
    except Exception as e:
        logger.debug(f"Error analyzing {filepath}: {e}")
        return None


async def list_tools(
    directories: List[str] = None,
    include_details: bool = False,
    format: str = 'simple',
    **kwargs
) -> Dict[str, Any]:
    """
    List available tools using safe static analysis.
    
    This function analyzes tool files without importing or executing them,
    preventing unintended system initialization or side effects.
    
    Args:
        directories: List of directories to search (default: ["scripts/custom_tools"])
        include_details: Include full parameter and example information
        format: Output format - "simple", "detailed", or "grouped"
        
    Returns:
        Dictionary with tool information
    """
    # Default directories
    if directories is None:
        directories = ["scripts/custom_tools"]
        
    # Validate format
    if format not in ['simple', 'detailed', 'grouped']:
        format = 'simple'
        
    tools = []
    errors = []
    
    # Search each directory
    for directory in directories:
        if not os.path.exists(directory):
            errors.append(f"Directory not found: {directory}")
            continue
            
        # Find Python files
        try:
            for file in os.listdir(directory):
                if file.endswith('.py') and not file.startswith('_'):
                    filepath = os.path.join(directory, file)
                    tool_info = analyze_tool_safely(filepath)
                    if tool_info:
                        # Adjust detail level
                        if not include_details:
                            # Remove detailed info for simple format
                            tool_info.pop('parameters', None)
                            tool_info.pop('examples', None)
                            tool_info.pop('function', None)
                            tool_info.pop('path', None)
                        tools.append(tool_info)
        except Exception as e:
            errors.append(f"Error reading directory {directory}: {str(e)}")
            
    # Sort tools by name
    tools.sort(key=lambda x: x['name'])
    
    # Format output based on requested format
    if format == 'simple':
        # Just tool names and descriptions
        result = {
            'success': True,
            'count': len(tools),
            'tools': [{'name': t['name'], 'description': t['description']} for t in tools]
        }
    elif format == 'grouped':
        # Group by directory
        grouped = {}
        for tool in tools:
            if 'path' in tool:
                dir_name = os.path.dirname(tool['path'])
                if dir_name not in grouped:
                    grouped[dir_name] = []
                grouped[dir_name].append(tool)
        result = {
            'success': True,
            'count': len(tools),
            'tools_by_directory': grouped
        }
    else:  # detailed
        result = {
            'success': True,
            'count': len(tools),
            'tools': tools
        }
        
    # Add any errors
    if errors:
        result['errors'] = errors
        
    # Add metadata
    result['method'] = 'static_analysis'
    result['note'] = 'Tools analyzed without code execution'
    
    return result


# For backward compatibility
def list_available_tools(**kwargs):
    """Alias for list_tools."""
    import asyncio
    return asyncio.run(list_tools(**kwargs))


if __name__ == '__main__':
    # Test the tool
    import asyncio
    
    async def test():
        print("Testing safe tool listing...")
        result = await list_tools(include_details=True, format='detailed')
        print(json.dumps(result, indent=2))
        
    asyncio.run(test())