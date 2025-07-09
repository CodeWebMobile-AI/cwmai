"""
Tool Error Monitor

Proactive monitoring system that detects and automatically fixes tool loading errors.
Specifically handles import errors and other initialization issues.
"""

import asyncio
import logging
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Ensure environment variables are loaded
env_files = ['.env.local', '.env']
for env_file in env_files:
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        break

from scripts.module_discovery import get_module_discovery


class ToolErrorMonitor:
    """Monitors tool loading errors and automatically fixes them."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_patterns = {
            'import_error': re.compile(r"No module named ['\"](.+?)['\"]"),
            'attribute_error': re.compile(r"module '(.+?)' has no attribute '(.+?)'"),
            'type_error': re.compile(r"(.+?)\.__init__\(\) got an unexpected keyword argument '(.+?)'"),
            'name_error': re.compile(r"name '(.+?)' is not defined"),
            'argument_error': re.compile(r"(.+?)\(\) takes (\d+) positional argument[s]? but (\d+) (?:was|were) given"),
        }
        self.fix_attempts = {}  # Track fix attempts to prevent infinite loops
        self.max_fix_attempts = 3
        
        # Get module discovery system
        self.module_discovery = get_module_discovery()
        
        # Module mapping for common import errors (will be enhanced by discovery)
        self.module_mappings = {
            'tool_executor': 'tool_calling_system',
            'workflow_manager': 'workflow_orchestrator',
            'task_executor': 'task_manager',
            'ai_executor': 'ai_brain',
            'state_storage': 'state_manager',
        }
        
    async def monitor_and_fix_tool_error(self, tool_name: str, error_msg: str, tool_path: Path) -> bool:
        """
        Monitor a tool error and attempt to fix it automatically.
        
        Args:
            tool_name: Name of the tool with the error
            error_msg: The error message
            tool_path: Path to the tool file
            
        Returns:
            True if the error was fixed, False otherwise
        """
        print(f"Monitoring error for tool '{tool_name}': {error_msg}")
        
        # Check if we've exceeded max attempts
        attempt_key = f"{tool_name}:{error_msg[:50]}"
        attempts = self.fix_attempts.get(attempt_key, 0)
        if attempts >= self.max_fix_attempts:
            self.logger.warning(f"Max fix attempts ({self.max_fix_attempts}) reached for {tool_name}")
            return False
            
        self.fix_attempts[attempt_key] = attempts + 1
        
        # Identify error type and get fix strategy
        error_type, error_details = self._identify_error_type(error_msg)
        
        if error_type == 'import_error':
            return await self._fix_import_error(tool_name, tool_path, error_details)
        elif error_type == 'attribute_error':
            return await self._fix_attribute_error(tool_name, tool_path, error_details)
        elif error_type == 'type_error':
            return await self._fix_type_error(tool_name, tool_path, error_details)
        elif error_type == 'name_error':
            return await self._fix_name_error(tool_name, tool_path, error_details)
        elif error_type == 'argument_error':
            return await self._fix_argument_error(tool_name, tool_path, error_details)
        else:
            self.logger.warning(f"Unknown error type for tool '{tool_name}': {error_msg}")
            return await self._fix_generic_error(tool_name, tool_path, error_msg)
    
    def _identify_error_type(self, error_msg: str) -> Tuple[str, Dict[str, Any]]:
        """Identify the type of error and extract relevant details."""
        for error_type, pattern in self.error_patterns.items():
            match = pattern.search(error_msg)
            if match:
                return error_type, {'groups': match.groups(), 'full_match': match.group(0)}
        return 'unknown', {}
    
    async def _fix_import_error(self, tool_name: str, tool_path: Path, error_details: Dict) -> bool:
        """Fix import errors by updating imports to correct modules."""
        missing_module = error_details['groups'][0] if error_details.get('groups') else None
        if not missing_module:
            return False
            
        print(f"Fixing import error for '{tool_name}': missing module '{missing_module}'")
        
        # Check if it's a scripts module
        if missing_module.startswith('scripts.'):
            module_part = missing_module.replace('scripts.', '')
            
            # First check module mappings
            if module_part in self.module_mappings:
                new_module = self.module_mappings[module_part]
                return await self._update_tool_imports(tool_path, module_part, new_module)
            
            # Use module discovery to find alternative
            suggested_module = self.module_discovery.suggest_module_for_import(module_part)
            if suggested_module:
                print(f"Module discovery suggests '{suggested_module}' instead of '{module_part}'")
                return await self._update_tool_imports(tool_path, module_part, suggested_module)
                
        # Check if it's a class import that needs to be found
        elif '.' not in missing_module:
            # Try to find the class in available modules
            import_suggestion = self.module_discovery.suggest_class_import(missing_module)
            if import_suggestion:
                import_statement, module = import_suggestion
                print(f"Found class '{missing_module}' in module '{module}'")
                return await self._add_missing_import(tool_path, import_statement)
        
        # Use AI to regenerate the tool with correct imports
        return await self._regenerate_tool_with_ai(tool_name, tool_path, f"Import error: {missing_module}")
    
    async def _fix_attribute_error(self, tool_name: str, tool_path: Path, error_details: Dict) -> bool:
        """Fix attribute errors by updating method/attribute names."""
        error_msg = error_details.get('full_match', '')
        
        # Check for NoneType attribute errors
        if "'NoneType' object has no attribute" in error_msg:
            print(f"Fixing NoneType attribute error for '{tool_name}'")
            # Extract the attribute being accessed
            import re
            match = re.search(r"has no attribute '(\w+)'", error_msg)
            if match:
                attribute = match.group(1)
                return await self._fix_nonetype_error(tool_path, attribute)
                
        if error_details.get('groups') and len(error_details['groups']) >= 2:
            module_name = error_details['groups'][0]
            attribute_name = error_details['groups'][1]
            print(f"Fixing attribute error for '{tool_name}': {module_name}.{attribute_name}")
            
            # Common attribute fixes
            attribute_fixes = {
                'generate': 'generate_enhanced_response',
                'create_task': 'add_task',
                'get_state': 'load_state',
                'set_state': 'save_state',
            }
            
            # Special case for HTTPAIClient.generate
            if module_name == 'HTTPAIClient' and attribute_name == 'generate':
                return await self._update_method_calls(tool_path, 'generate', 'generate_enhanced_response')
            elif attribute_name in attribute_fixes:
                return await self._update_tool_attributes(tool_path, attribute_name, attribute_fixes[attribute_name])
                
        return await self._regenerate_tool_with_ai(tool_name, tool_path, f"Attribute error: {error_details}")
    
    async def _fix_type_error(self, tool_name: str, tool_path: Path, error_details: Dict) -> bool:
        """Fix type errors by updating function calls with correct parameters."""
        if error_details.get('groups') and len(error_details['groups']) >= 2:
            class_name = error_details['groups'][0]
            wrong_param = error_details['groups'][1]
            print(f"Fixing type error for '{tool_name}': {class_name} wrong param '{wrong_param}'")
            
            # Common parameter fixes
            param_fixes = {
                'tool_calling_system': None,  # Remove this parameter
                'state_manager': None,
                'logger': None,
            }
            
            if wrong_param in param_fixes:
                return await self._fix_function_parameters(tool_path, class_name, wrong_param, param_fixes[wrong_param])
                
        return await self._regenerate_tool_with_ai(tool_name, tool_path, f"Type error: {error_details}")
    
    async def _fix_name_error(self, tool_name: str, tool_path: Path, error_details: Dict) -> bool:
        """Fix name errors by adding missing imports or definitions."""
        undefined_name = error_details['groups'][0] if error_details.get('groups') else None
        if undefined_name:
            print(f"Fixing name error for '{tool_name}': undefined name '{undefined_name}'")
            
            # Common missing imports
            common_imports = {
                'json': 'import json',
                'asyncio': 'import asyncio',
                'logging': 'import logging',
                'Path': 'from pathlib import Path',
                'datetime': 'from datetime import datetime',
                'Dict': 'from typing import Dict',
                'Any': 'from typing import Any',
                'List': 'from typing import List',
            }
            
            if undefined_name in common_imports:
                return await self._add_missing_import(tool_path, common_imports[undefined_name])
                
        return await self._regenerate_tool_with_ai(tool_name, tool_path, f"Name error: {undefined_name}")
    
    async def _fix_generic_error(self, tool_name: str, tool_path: Path, error_msg: str) -> bool:
        """Fix generic errors by regenerating the tool."""
        print(f"Attempting generic fix for '{tool_name}'")
        return await self._regenerate_tool_with_ai(tool_name, tool_path, error_msg)
    
    async def _update_tool_imports(self, tool_path: Path, old_module: str, new_module: str) -> bool:
        """Update import statements in a tool file."""
        try:
            content = tool_path.read_text()
            
            # Update various import patterns
            patterns = [
                (f'from scripts.{old_module} import', f'from scripts.{new_module} import'),
                (f'import scripts.{old_module}', f'import scripts.{new_module}'),
                (f'scripts.{old_module}.', f'scripts.{new_module}.'),
            ]
            
            modified = False
            for old_pattern, new_pattern in patterns:
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    modified = True
                    
            if modified:
                tool_path.write_text(content)
                print(f"Updated imports in {tool_path.name}: {old_module} -> {new_module}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating imports: {e}")
            
        return False
    
    async def _update_tool_attributes(self, tool_path: Path, old_attr: str, new_attr: str) -> bool:
        """Update attribute names in a tool file."""
        try:
            content = tool_path.read_text()
            
            # Update attribute usage
            pattern = rf'\.{old_attr}\b'
            if re.search(pattern, content):
                content = re.sub(pattern, f'.{new_attr}', content)
                tool_path.write_text(content)
                print(f"Updated attributes in {tool_path.name}: {old_attr} -> {new_attr}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating attributes: {e}")
            
        return False
    
    async def _fix_function_parameters(self, tool_path: Path, class_name: str, param_to_remove: str, new_param: Optional[str]) -> bool:
        """Fix function parameters by removing or replacing them."""
        try:
            content = tool_path.read_text()
            
            # Pattern to find the class instantiation
            pattern = rf'{class_name}\s*\([^)]*{param_to_remove}[^)]*\)'
            matches = list(re.finditer(pattern, content))
            
            if matches:
                # Work backwards to not mess up positions
                for match in reversed(matches):
                    old_call = match.group(0)
                    # Remove the parameter
                    new_call = re.sub(rf',?\s*{param_to_remove}\s*=\s*[^,)]+', '', old_call)
                    # Clean up any double commas or leading commas
                    new_call = re.sub(r',\s*,', ',', new_call)
                    new_call = re.sub(r'\(\s*,', '(', new_call)
                    
                    content = content[:match.start()] + new_call + content[match.end():]
                
                tool_path.write_text(content)
                print(f"Fixed parameters in {tool_path.name}: removed '{param_to_remove}'")
                return True
                
        except Exception as e:
            self.logger.error(f"Error fixing parameters: {e}")
            
        return False
    
    async def _fix_nonetype_error(self, tool_path: Path, attribute: str) -> bool:
        """Fix NoneType errors by adding null checks."""
        try:
            content = tool_path.read_text()
            
            # Pattern to find where attribute is accessed
            patterns = [
                (rf'(\w+)\.{attribute}\s*\(', rf'(\1 and \1.{attribute})'),  # obj.lower() -> (obj and obj.lower())
                (rf'(\w+)\[["\']{attribute}["\']\]\.', rf'\1.get("{attribute}", "").'),  # obj["name"].lower() -> obj.get("name", "").lower()
                (rf'if\s+(\w+)\.get\("(\w+)"\)\s+and\s+\1\["(\2)"\]\.', rf'if \1.get("\2") and \1.get("\2", "").')  # Special case for get() followed by direct access
            ]
            
            modified = False
            for pattern, replacement in patterns:
                new_content, count = re.subn(pattern, replacement, content)
                if count > 0:
                    content = new_content
                    modified = True
                    print(f"Applied NoneType fix pattern for attribute '{attribute}': {count} replacements")
                    
            if modified:
                tool_path.write_text(content)
                print(f"Fixed NoneType errors in {tool_path.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error fixing NoneType: {e}")
            
        return False
    
    async def _update_method_calls(self, tool_path: Path, old_method: str, new_method: str) -> bool:
        """Update method calls from old name to new name."""
        try:
            content = tool_path.read_text()
            
            # Replace method calls
            pattern = rf'\.{old_method}\s*\('
            replacement = f'.{new_method}('
            
            new_content, count = re.subn(pattern, replacement, content)
            
            if count > 0:
                tool_path.write_text(new_content)
                print(f"Updated {count} method calls from {old_method} to {new_method} in {tool_path.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating method calls: {e}")
            
        return False
    
    async def _add_missing_import(self, tool_path: Path, import_statement: str) -> bool:
        """Add a missing import to a tool file."""
        try:
            content = tool_path.read_text()
            
            # Check if import already exists
            if import_statement in content:
                return False
                
            # Find the right place to add the import (after other imports)
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip() == '':
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
                    
            # Insert the import
            lines.insert(import_section_end, import_statement)
            content = '\n'.join(lines)
            
            tool_path.write_text(content)
            print(f"Added import to {tool_path.name}: {import_statement}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding import: {e}")
            
        return False
    
    async def _fix_argument_error(self, tool_name: str, tool_path: Path, error_details: Dict) -> bool:
        """Fix argument errors like save_state() taking wrong number of arguments."""
        if error_details.get('groups') and len(error_details['groups']) >= 3:
            function_name = error_details['groups'][0]
            expected_args = error_details['groups'][1]
            given_args = error_details['groups'][2]
            
            print(f"Fixing argument error: {function_name} expects {expected_args} args but got {given_args}")
            
            # Special case for StateManager.save_state()
            if 'save_state' in function_name and expected_args == '1' and given_args == '2':
                return await self._fix_save_state_calls(tool_path)
                
        return await self._regenerate_tool_with_ai(tool_name, tool_path, f"Argument error: {error_details}")
    
    async def _fix_save_state_calls(self, tool_path: Path) -> bool:
        """Fix incorrect save_state() calls that pass arguments."""
        try:
            content = tool_path.read_text()
            
            # Pattern to find save_state calls with arguments
            # Match patterns like: save_state(state), save_state(data), etc.
            pattern = r'(\w+)\.save_state\s*\(\s*([^)]+)\s*\)'
            
            def replacement(match):
                obj_name = match.group(1)
                arg_name = match.group(2).strip()
                # Replace with proper pattern: obj.state = arg; obj.save_state()
                return f"{obj_name}.state = {arg_name}\n        {obj_name}.save_state()"
            
            new_content, count = re.subn(pattern, replacement, content)
            
            if count > 0:
                tool_path.write_text(new_content)
                print(f"Fixed {count} save_state() calls in {tool_path.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error fixing save_state calls: {e}")
            
        return False
    
    async def _regenerate_tool_with_ai(self, tool_name: str, tool_path: Path, error_context: str) -> bool:
        """Use AI to regenerate a tool with fixes for the identified errors."""
        try:
            # Get original content before regenerating
            original_content = tool_path.read_text() if tool_path.exists() else ""
            
            # Use ConversationalAIAssistant to regenerate the tool
            print(f"Using conversational AI to regenerate tool: {tool_name}")
            
            # Import here to avoid circular imports
            from scripts.conversational_ai_assistant import ConversationalAIAssistant
            
            # Initialize assistant
            assistant = ConversationalAIAssistant()
            await assistant.initialize()
            
            # Get original tool content and metadata
            original_description = ""
            original_requirements = ""
            
            if original_content:
                # Extract description
                desc_match = re.search(r'__description__\s*=\s*["\'](.+?)["\']', original_content)
                if desc_match:
                    original_description = desc_match.group(1)
                    
                # Extract original requirements from docstring
                req_match = re.search(r'Requirements:\s*(.+?)(?:\n\n|\*/)', original_content, re.DOTALL)
                if req_match:
                    original_requirements = req_match.group(1).strip()
            
            # Delete the broken tool
            if tool_path.exists():
                tool_path.unlink()
                print(f"Deleted broken tool file: {tool_path}")
            
            # Use the assistant's _handle_broken_tool method
            fix_result = await assistant._handle_broken_tool(
                tool_name,
                {},  # Empty params
                f"{error_context}. This is an automatic fix attempt by the error monitor."
            )
            
            # Check if the fix was successful
            if tool_path.exists():
                # Verify the tool can be loaded
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(tool_name, tool_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    print(f"Successfully regenerated and verified tool: {tool_name}")
                    return True
                except Exception as verify_error:
                    self.logger.error(f"Regenerated tool still has errors: {verify_error}")
                    return False
            else:
                self.logger.error(f"Tool regeneration failed - file not created")
                return False
                
        except Exception as e:
            self.logger.error(f"Error regenerating tool with AI: {e}")
            return False
    
    def reset_fix_attempts(self, tool_name: Optional[str] = None):
        """Reset fix attempt counters."""
        if tool_name:
            # Reset specific tool
            keys_to_remove = [k for k in self.fix_attempts.keys() if k.startswith(f"{tool_name}:")]
            for key in keys_to_remove:
                del self.fix_attempts[key]
        else:
            # Reset all
            self.fix_attempts.clear()
            
    async def monitor_directory(self, directory: Path, fix_errors: bool = True) -> Dict[str, Any]:
        """
        Monitor a directory for tool errors and optionally fix them.
        
        Args:
            directory: Directory to monitor
            fix_errors: Whether to attempt to fix errors
            
        Returns:
            Summary of monitoring results
        """
        results = {
            'total_tools': 0,
            'healthy_tools': 0,
            'broken_tools': 0,
            'fixed_tools': 0,
            'failed_fixes': 0,
            'errors': []
        }
        
        # Find all Python files in directory
        tool_files = list(directory.glob('*.py'))
        results['total_tools'] = len(tool_files)
        
        for tool_file in tool_files:
            if tool_file.name.startswith('_'):
                continue
                
            tool_name = tool_file.stem
            
            try:
                # Try to import the tool
                import importlib.util
                spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                results['healthy_tools'] += 1
                
            except Exception as e:
                results['broken_tools'] += 1
                error_msg = str(e)
                results['errors'].append({
                    'tool': tool_name,
                    'error': error_msg,
                    'fixed': False
                })
                
                if fix_errors:
                    print(f"Attempting to fix error in {tool_name}: {error_msg}")
                    fixed = await self.monitor_and_fix_tool_error(tool_name, error_msg, tool_file)
                    
                    if fixed:
                        results['fixed_tools'] += 1
                        results['errors'][-1]['fixed'] = True
                    else:
                        results['failed_fixes'] += 1
                        
        return results


# Singleton instance
_monitor_instance = None

def get_tool_error_monitor() -> ToolErrorMonitor:
    """Get the singleton ToolErrorMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ToolErrorMonitor()
    return _monitor_instance