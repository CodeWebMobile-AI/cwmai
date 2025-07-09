"""
Tool Context Enhancer

Enhances context extraction and parameter parsing for tool calls.
Ensures tools receive proper parameters from natural language input.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import inspect
import ast

class ToolContextEnhancer:
    """Enhances context extraction for tool calls from natural language."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Common patterns for extracting context
        self.extraction_patterns = {
            'repository': [
                r'(?:repository|repo)\s+(?:name\s+)?["\']?([a-zA-Z0-9-_]+)["\']?',
                r'["\']?([a-zA-Z0-9-_]+)["\']?\s+(?:repository|repo)',
                r'(?:for|in|on|about)\s+["\']?([a-zA-Z0-9-_]+)["\']?',
                r'^["\']?([a-zA-Z0-9-_]+)["\']?\s+["\']?\w+["\']?$',  # "repo-name 'tool_name'"
                r'^([a-zA-Z0-9-_]+)\s+',  # First word if it looks like a repo
            ],
            'tool_name': [
                r'["\'](\w+)["\']',  # Quoted tool name
                r'(?:tool|command|function)\s+["\']?(\w+)["\']?',
                r'(?:run|execute|call)\s+["\']?(\w+)["\']?',
            ],
            'file_path': [
                r'(?:file|path)\s+["\']?([/\w.-]+)["\']?',
                r'["\']([/\w.-]+\.\w+)["\']',  # Quoted file path with extension
            ],
            'number': [
                r'(\d+)\s*(?:lines?|items?|results?)?',
                r'(?:top|first|last)\s+(\d+)',
            ]
        }
        
        # Parameter name mappings for common variations
        self.param_mappings = {
            'repo': 'repo_name',
            'repository': 'repo_name',
            'project': 'repo_name',
            'name': 'repo_name',
            'path': 'file_path',
            'file': 'file_path',
            'dir': 'directory',
            'folder': 'directory',
            'num': 'number',
            'count': 'number',
            'limit': 'number',
        }
        
    def extract_context_from_input(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract context and parameters from user input for a specific tool.
        
        Args:
            user_input: The raw user input
            tool_name: The name of the tool being called
            
        Returns:
            Dictionary of extracted parameters
        """
        context = {}
        
        # Clean input
        clean_input = user_input.strip()
        
        # Extract repository name if the tool likely needs it
        if self._tool_needs_repository(tool_name):
            repo_name = self._extract_repository(clean_input, tool_name)
            if repo_name:
                context['repo_name'] = repo_name
                context['repo'] = repo_name  # Add both variations
                
        # Extract file path if the tool likely needs it
        if self._tool_needs_file_path(tool_name):
            file_path = self._extract_file_path(clean_input)
            if file_path:
                context['file_path'] = file_path
                
        # Extract numbers if the tool likely needs them
        if self._tool_needs_number(tool_name):
            number = self._extract_number(clean_input)
            if number:
                context['number'] = number
                context['limit'] = number  # Add common variations
                context['count'] = number
                
        # Add the full user input as context
        context['context'] = {'user_input': clean_input}
        
        return context
        
    def enhance_tool_parameters(self, tool_name: str, provided_params: Dict[str, Any], 
                              user_input: str) -> Dict[str, Any]:
        """
        Enhance tool parameters by extracting missing ones from user input.
        
        Args:
            tool_name: Name of the tool
            provided_params: Parameters already provided
            user_input: The original user input
            
        Returns:
            Enhanced parameters dictionary
        """
        # Start with provided parameters
        enhanced_params = provided_params.copy()
        
        # Extract additional context from user input
        extracted_context = self.extract_context_from_input(user_input, tool_name)
        
        # Merge extracted context, but don't override existing params
        for key, value in extracted_context.items():
            if key not in enhanced_params:
                enhanced_params[key] = value
                
        # Map common parameter variations
        enhanced_params = self._map_parameter_names(enhanced_params)
        
        # Read tool file to understand expected parameters
        expected_params = self._get_tool_expected_parameters(tool_name)
        
        # Try to fill in missing required parameters
        for param_name, param_info in expected_params.items():
            if param_name not in enhanced_params and param_info.get('required', False):
                # Try to extract this specific parameter
                value = self._extract_specific_parameter(param_name, user_input)
                if value:
                    enhanced_params[param_name] = value
                    
        return enhanced_params
        
    def _tool_needs_repository(self, tool_name: str) -> bool:
        """Check if a tool likely needs a repository parameter."""
        repo_keywords = ['repository', 'repo', 'project', 'github', 'code', 'analyze']
        return any(keyword in tool_name.lower() for keyword in repo_keywords)
        
    def _tool_needs_file_path(self, tool_name: str) -> bool:
        """Check if a tool likely needs a file path parameter."""
        file_keywords = ['file', 'read', 'write', 'analyze', 'parse', 'load']
        return any(keyword in tool_name.lower() for keyword in file_keywords)
        
    def _tool_needs_number(self, tool_name: str) -> bool:
        """Check if a tool likely needs a number parameter."""
        number_keywords = ['count', 'limit', 'top', 'analyze', 'list', 'show']
        return any(keyword in tool_name.lower() for keyword in number_keywords)
        
    def _extract_repository(self, user_input: str, tool_name: str) -> Optional[str]:
        """Extract repository name from user input."""
        # First, check if the tool name is quoted and preceded by a repo name
        # Pattern: repo-name 'tool_name'
        tool_quoted_pattern = rf'^([a-zA-Z0-9-_]+)\s+["\']?{re.escape(tool_name)}["\']?'
        match = re.match(tool_quoted_pattern, user_input)
        if match:
            return match.group(1)
            
        # Try all repository extraction patterns
        for pattern in self.extraction_patterns['repository']:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
                
        # Special case: if input is "word 'other_stuff'", the word might be the repo
        first_word_match = re.match(r'^([a-zA-Z0-9-_]+)\s+["\']', user_input)
        if first_word_match:
            potential_repo = first_word_match.group(1)
            # Check if it looks like a repository name
            if '-' in potential_repo or potential_repo.lower().endswith('ai'):
                return potential_repo
                
        return None
        
    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """Extract file path from user input."""
        for pattern in self.extraction_patterns['file_path']:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
        
    def _extract_number(self, user_input: str) -> Optional[int]:
        """Extract number from user input."""
        for pattern in self.extraction_patterns['number']:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        return None
        
    def _extract_specific_parameter(self, param_name: str, user_input: str) -> Optional[Any]:
        """Try to extract a specific parameter from user input."""
        # Use parameter name to determine extraction strategy
        if param_name in ['repo_name', 'repo', 'repository']:
            return self._extract_repository(user_input, '')
        elif param_name in ['file_path', 'path', 'file']:
            return self._extract_file_path(user_input)
        elif param_name in ['number', 'count', 'limit']:
            return self._extract_number(user_input)
            
        # Try generic extraction
        # Pattern: param_name=value or param_name: value
        patterns = [
            rf'{param_name}\s*=\s*["\']?([^"\'\\s]+)["\']?',
            rf'{param_name}\s*:\s*["\']?([^"\'\\s]+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
        
    def _map_parameter_names(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map common parameter name variations to expected names."""
        mapped_params = {}
        
        for key, value in params.items():
            # Check if this key has a mapping
            if key in self.param_mappings:
                mapped_key = self.param_mappings[key]
                mapped_params[mapped_key] = value
            else:
                mapped_params[key] = value
                
        return mapped_params
        
    def _get_tool_expected_parameters(self, tool_name: str) -> Dict[str, Dict[str, Any]]:
        """Get expected parameters for a tool by reading its file."""
        expected_params = {}
        
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        if not tool_path.exists():
            return expected_params
            
        try:
            content = tool_path.read_text()
            
            # Look for __parameters__ definition
            param_match = re.search(r'__parameters__\s*=\s*({[^}]+})', content, re.DOTALL)
            if param_match:
                try:
                    # Safely evaluate the parameters dict
                    params_str = param_match.group(1)
                    expected_params = ast.literal_eval(params_str)
                except:
                    pass
                    
            # Also look for function signature
            func_match = re.search(rf'async\s+def\s+{tool_name}\s*\((.*?)\)', content, re.DOTALL)
            if func_match:
                sig_str = func_match.group(1)
                # Parse parameters from signature
                if 'kwargs' not in sig_str:
                    # Extract individual parameters
                    param_pattern = r'(\w+)(?:\s*:\s*[^,=]+)?(?:\s*=\s*[^,]+)?'
                    for match in re.finditer(param_pattern, sig_str):
                        param_name = match.group(1)
                        if param_name not in ['self', 'cls']:
                            expected_params[param_name] = {'required': '=' not in match.group(0)}
                            
        except Exception as e:
            self.logger.error(f"Error reading tool parameters: {e}")
            
        return expected_params
        
    def create_enhanced_tool_call(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse user input to create an enhanced tool call.
        
        Args:
            user_input: Raw user input like "repo-name 'tool_name'"
            
        Returns:
            Tuple of (tool_name, parameters)
        """
        # Extract tool name
        tool_name = None
        remaining_input = user_input
        
        # Check for quoted tool name
        quoted_match = re.search(r'["\'](\w+)["\']', user_input)
        if quoted_match:
            tool_name = quoted_match.group(1)
            # Remove the quoted tool name from input
            remaining_input = user_input[:quoted_match.start()] + user_input[quoted_match.end():]
        else:
            # Try to extract tool name from patterns
            for pattern in self.extraction_patterns['tool_name']:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    tool_name = match.group(1)
                    break
                    
        if not tool_name:
            # Last resort: last word might be the tool
            words = user_input.split()
            if words:
                tool_name = words[-1].strip('"\'')
                remaining_input = ' '.join(words[:-1])
                
        # Extract parameters from remaining input
        if tool_name:
            params = self.extract_context_from_input(remaining_input.strip(), tool_name)
            return tool_name, params
        else:
            return None, {}


# Singleton instance
_enhancer_instance = None

def get_tool_context_enhancer() -> ToolContextEnhancer:
    """Get the singleton ToolContextEnhancer instance."""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = ToolContextEnhancer()
    return _enhancer_instance