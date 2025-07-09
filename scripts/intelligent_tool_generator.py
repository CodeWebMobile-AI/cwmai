#!/usr/bin/env python3
"""
Intelligent Tool Generator
=========================

This module ensures tools are created with proper integration into the existing codebase.
It enforces exploration, pattern matching, and proper implementation.
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from scripts.http_ai_client import HTTPAIClient
from scripts.code_analyzer import CodeAnalyzer
from scripts.module_discovery import ModuleDiscovery
from scripts.tool_categorizer import ToolCategorizer
from scripts.conversational_ai_assistant import ConversationalAIAssistant


class IntelligentToolGenerator:
    """Generates tools that properly integrate with the existing codebase."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.code_analyzer = CodeAnalyzer()
        self.module_discovery = ModuleDiscovery()
        self.assistant = None  # Will be initialized when needed
        self.tool_categorizer = ToolCategorizer()
        
    async def generate_tool(self, name: str, description: str, requirements: str) -> Dict[str, Any]:
        """Generate a tool with proper codebase integration."""
        
        print(f"🔍 Analyzing codebase for tool: {name}")
        
        # Step 1: Analyze requirements to understand what the tool needs
        parsed_requirements = await self._analyze_requirements(requirements)
        
        # Step 2: Search for similar existing tools
        similar_tools = await self._find_similar_tools(name, description, parsed_requirements)
        
        # Step 3: Discover relevant modules and imports
        relevant_modules = await self._discover_relevant_modules(parsed_requirements)
        
        # Step 4: Analyze patterns from similar tools
        patterns = await self._analyze_tool_patterns(similar_tools)
        
        # Step 5: Generate the tool with proper integration
        tool_code = await self._generate_integrated_tool(
            name, description, parsed_requirements, 
            relevant_modules, patterns
        )
        
        # Step 6: Validate the generated tool
        validation_result = await self._validate_tool(name, tool_code)
        
        if validation_result['valid']:
            # Categorize the tool
            category, confidence, analysis = self.tool_categorizer.categorize_tool(
                name, description, tool_code
            )
            
            # Ensure category directory exists
            category_path = self.tool_categorizer.ensure_category_exists(category)
            
            # Save the tool in the appropriate category
            tool_path = category_path / f"{name}.py"
            tool_path.write_text(tool_code)
            
            print(f"Tool categorized as '{category}' (confidence: {confidence:.2f})")
            if analysis['reasons']:
                print(f"Reasoning: {'; '.join(analysis['reasons'])}")
            
            return {
                "success": True,
                "path": str(tool_path),
                "category": category,
                "category_confidence": confidence,
                "integration_points": relevant_modules,
                "based_on_patterns": [t['name'] for t in similar_tools[:3]]
            }
        else:
            # Save for debugging
            debug_path = Path(f"scripts/custom_tools/{name}_debug.py")
            debug_path.write_text(tool_code)
            
            return {
                "success": False,
                "error": "Tool validation failed",
                "issues": validation_result['issues'],
                "debug_path": str(debug_path),
                "generated_code": tool_code[:500] + "..." if len(tool_code) > 500 else tool_code
            }
    
    async def _analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Parse requirements to understand what the tool needs to do."""
        prompt = f"""Analyze these tool requirements and extract:
1. Core functionality needed
2. External systems to integrate with (GitHub, filesystem, etc)
3. Data sources required
4. Expected inputs and outputs
5. Similar existing functionality that could be reused

Requirements:
{requirements}

Return as JSON with keys: functionality, integrations, data_sources, io_spec, similar_features"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        try:
            return json.loads(response.get('content', '{}'))
        except:
            # Fallback parsing
            return {
                "functionality": requirements,
                "integrations": [],
                "data_sources": [],
                "io_spec": {},
                "similar_features": []
            }
    
    async def _find_similar_tools(self, name: str, description: str, 
                                  parsed_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find existing tools that do similar things."""
        similar_tools = []
        
        # Search by name similarity
        tools_dir = Path("scripts/custom_tools")
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*.py"):
                if any(word in tool_file.stem for word in name.split('_')):
                    tool_info = await self._analyze_tool_file(tool_file)
                    if tool_info:
                        similar_tools.append(tool_info)
        
        # Search by functionality
        for feature in parsed_requirements.get('similar_features', []):
            # Search for tools that might implement this feature
            search_results = await self._search_codebase(feature)
            for result in search_results[:3]:
                tool_info = await self._analyze_tool_file(Path(result))
                if tool_info and tool_info not in similar_tools:
                    similar_tools.append(tool_info)
        
        return similar_tools[:5]  # Return top 5 most similar
    
    async def _discover_relevant_modules(self, parsed_requirements: Dict[str, Any]) -> Dict[str, List[str]]:
        """Discover which existing modules to import and use."""
        modules = {
            'imports': [],
            'integrations': [],
            'utilities': []
        }
        
        # Map requirements to known modules
        integration_mapping = {
            'github': ['scripts.github_issue_creator', 'scripts.mcp_github_issue_creator'],
            'repository': ['scripts.repository_analyzer', 'scripts.state_manager'],
            'workflow': ['scripts.workflow_orchestrator', 'scripts.workflow_executor'],
            'ai': ['scripts.http_ai_client', 'scripts.ai_brain'],
            'file': ['pathlib.Path', 'os', 'glob'],
            'async': ['asyncio', 'scripts.async_state_manager'],
            'state': ['scripts.state_manager', 'scripts.redis_state_adapter'],
            'task': ['scripts.task_manager', 'scripts.task_persistence']
        }
        
        # Check which integrations are needed
        for integration in parsed_requirements.get('integrations', []):
            integration_lower = integration.lower()
            for key, modules_list in integration_mapping.items():
                if key in integration_lower:
                    modules['integrations'].extend(modules_list)
        
        # Discover actual module paths
        for module in modules['integrations']:
            if '.' in module:
                # It's already a module path
                modules['imports'].append(module)
            else:
                # Try to find the module
                discovered = self.module_discovery.find_module(module)
                if discovered:
                    modules['imports'].append(discovered)
        
        # Remove duplicates
        modules['imports'] = list(set(modules['imports']))
        
        return modules
    
    async def _analyze_tool_patterns(self, similar_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from similar tools."""
        patterns = {
            'async_pattern': False,
            'error_handling': [],
            'return_format': None,
            'parameter_validation': [],
            'common_imports': []
        }
        
        if not similar_tools:
            # Default patterns from best practices
            patterns['async_pattern'] = True
            patterns['error_handling'] = ['try/except', 'logging']
            patterns['return_format'] = {'success': bool, 'result': Any, 'error': Optional[str]}
            return patterns
        
        # Analyze each similar tool
        for tool in similar_tools:
            code = tool.get('code', '')
            
            # Check if async
            if 'async def' in code:
                patterns['async_pattern'] = True
            
            # Extract error handling patterns
            if 'try:' in code:
                patterns['error_handling'].append('try/except')
            if 'logging' in code or 'logger' in code:
                patterns['error_handling'].append('logging')
            
            # Extract imports
            import_matches = re.findall(r'^(?:from|import)\s+(\S+)', code, re.MULTILINE)
            patterns['common_imports'].extend(import_matches)
        
        # Deduplicate
        patterns['error_handling'] = list(set(patterns['error_handling']))
        patterns['common_imports'] = list(set(patterns['common_imports']))
        
        return patterns
    
    async def _generate_integrated_tool(self, name: str, description: str,
                                      parsed_requirements: Dict[str, Any],
                                      relevant_modules: Dict[str, List[str]],
                                      patterns: Dict[str, Any]) -> str:
        """Generate the actual tool code with proper integration using conversational assistant."""
        
        # Initialize conversational assistant if not already done
        if self.assistant is None:
            self.assistant = ConversationalAIAssistant()
            await self.assistant.initialize()
        
        # First, run discovery ourselves to gather actual methods
        discovered_apis = await self._run_discovery_phase(parsed_requirements)
        
        # Build the request for the assistant
        request = f"""I need you to create a Python tool with proper codebase integration.
        
=== DISCOVERED APIs (from actual codebase) ===

Discovered modules and their methods:
{self._format_discovered_modules(discovered_apis.get('modules', {}))}

Available tools (call via tool_system):
{json.dumps(discovered_apis.get('tools', []), indent=2)}

Usage examples found:
{json.dumps(discovered_apis.get('examples', {}), indent=2)}

YOU MUST ONLY USE THE METHODS LISTED ABOVE!
IMPORTANT: If a method is marked as async, you MUST call it with await!

=== SAFE DATA ACCESS ===
When accessing nested data structures, ALWAYS include this helper function in your tool:

```python
def safe_get(data, *keys, default=None):
    \"\"\"Safely navigate nested dictionaries.\"\"\"
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
```

IMPORTANT: You MUST include this function definition in your generated tool code!

Example: Instead of `result['health_metrics']['health_score']`, use:
`health_score = safe_get(result, 'health_metrics', 'health_score', default=0)`

This prevents KeyError and AttributeError when data structure is different than expected.

Tool Name: {name}
Description: {description}
Requirements: {json.dumps(parsed_requirements, indent=2)}

=== CRITICAL: UNDERSTANDING TOOLS VS MODULES ===

This system has TWO different patterns you MUST understand:

1. MODULES/CLASSES (import and use directly):
   - StateManager, RepositoryAnalyzer, GitHubIssueCreator → These are Python classes
   - Import them: from scripts.module_name import ClassName
   - Use them: instance = ClassName(); result = instance.method()

2. TOOLS (call via tool_calling_system):
   - analyze_repository, get_repositories, create_issue → These are registered tools
   - DO NOT import them! They're not modules!
   - Call them: 
     ```python
     from scripts.tool_calling_system import ToolCallingSystem
     tool_system = ToolCallingSystem()
     result = await tool_system.call_tool('tool_name', param1=value1)
     ```

=== DISCOVERY PROCESS - FOLLOW IN ORDER ===

PHASE 1: Identify what you need
- List the functionality required
- For each function/class mentioned in requirements

PHASE 2: Classify each as MODULE or TOOL
- Is it a class? execute_command(command='grep -l "class ClassName" scripts/*.py')
  → If found in scripts/*.py (NOT custom_tools) → It's a MODULE, import it
- Is it a tool? execute_command(command='grep -l "def function_name" scripts/custom_tools/*.py') 
  → If found in custom_tools → It's a TOOL, call via tool_system
- Check registered tools: execute_command(command='grep -l "register_tool.*function_name" scripts/*.py')

PHASE 3: For MODULES - discover exact API
- Find methods: execute_command(command='grep "def " scripts/module_name.py | grep -v "__"')
- Check signatures: execute_command(command='grep -A 5 "def method_name" scripts/module_name.py')
- Find usage: execute_command(command='grep -B 2 -A 2 "ClassName(" scripts/')

PHASE 4: For TOOLS - understand parameters
- Check tool definition: execute_command(command='grep -A 20 "def tool_name" scripts/custom_tools/tool_name.py')
- Find examples: execute_command(command='grep -B 2 -A 2 "call_tool.*tool_name" scripts/')

=== IMPLEMENTATION RULES ===
1. Function MUST be named: {name}
2. Include: __description__ = "{description}"
3. Return: {{"success": bool, "result": data, "error": str if failed}}
4. Use logging: import logging; logger = logging.getLogger(__name__)
5. NO placeholders, NO assumptions - only use what you discovered

=== MANDATORY DISCOVERY OUTPUT FORMAT ===

You MUST show each discovery step and its results like this:

DISCOVERY STEP 1: [What you're looking for]
COMMAND: execute_command(command='...')
RESULT: [Show the actual output]
CONCLUSION: [What you learned]

DISCOVERY STEP 2: [Next thing to discover]
COMMAND: execute_command(command='...')
RESULT: [Show the actual output]
CONCLUSION: [What you learned]

Continue until you have discovered ALL actual method names and signatures.

=== METHOD INVENTORY REQUIREMENT ===
Before writing ANY code, create an inventory of discovered methods:

{self._generate_inventory_template(discovered_apis)}

DISCOVERED TOOLS (to call via tool_system):
- [List ONLY tools you confirmed exist in custom_tools]

=== CODE GENERATION RULES ===
1. You can ONLY use methods that appear in your DISCOVERED METHODS inventory
2. If a method is NOT in your inventory, you CANNOT use it - find an alternative
3. NO ASSUMPTIONS - if save_state() isn't in inventory, don't use it
4. NO INVENTED METHODS - only use what's in your inventory
5. ASYNC RULES:
   - If a discovered method is marked as async=true, you MUST use 'await' when calling it
   - If your tool calls ANY async method, the tool function itself must be 'async def'
   - Example: If analyze_repository is async, use: result = await analyzer.analyze_repository(repo)

=== FINAL OUTPUT ===
After discovery and inventory, provide the tool code:

<<<TOOL_CODE>>>
# Your complete Python code using ONLY discovered methods
<<<END_TOOL_CODE>>>

Start discovery now:"""
        
        # Use the conversational assistant to generate the tool with actual discovery
        response = await self.assistant.handle_conversation(request)
        
        # Check if discovery actually happened
        if not self._validate_discovery_happened(response):
            # Try again with more emphasis
            retry_request = request + "\n\nIMPORTANT: You MUST run execute_command to discover actual methods. DO NOT skip discovery! Show me the EXACT output of your grep commands!"
            response = await self.assistant.handle_conversation(retry_request)
            
        # Double-check for method inventory
        if "DISCOVERED METHODS" not in response:
            inventory_request = "Please show your METHOD INVENTORY listing all discovered methods before generating code."
            response = await self.assistant.handle_conversation(inventory_request)
        
        # Extract the tool code from the response
        # The assistant might return the code in various formats
        tool_code = self._extract_tool_code(response)
        
        # Ensure proper structure
        if '__description__' not in tool_code:
            tool_code = f'__description__ = "{description}"\n\n' + tool_code
        
        return tool_code
    
    async def _validate_tool(self, name: str, tool_code: str) -> Dict[str, Any]:
        """Validate the generated tool."""
        issues = []
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r'print\s*\(["\']Calling .+ tool',
            r'#\s*Placeholder',
            r'pass\s*#.*TODO',
            r'return\s+["\']placeholder',
            r'mock|dummy|fake|stub'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, tool_code, re.IGNORECASE):
                issues.append(f"Found placeholder pattern: {pattern}")
        
        # Check for required imports
        if 'import' not in tool_code:
            issues.append("No imports found - tool likely won't work")
        
        # Check for proper function pattern - can be named after the tool
        if f'async def {name}' not in tool_code and f'def {name}' not in tool_code:
            # Also check for execute function
            if 'async def execute' not in tool_code and 'def execute' not in tool_code:
                issues.append(f"No function found named '{name}' or 'execute'")
        
        # Check for common mistakes
        
        # 1. Direct tool calls without tool_system
        tool_names = ['analyze_repository', 'get_repositories', 'create_issue', 'get_project_info', 
                      'get_repository_issues', 'count_repositories', 'list_tools']
        for tool_name in tool_names:
            # Check if tool is called directly like a function
            direct_call_pattern = rf'(?<!tool_system\.call_tool\(["\']){tool_name}\s*\('
            if re.search(direct_call_pattern, tool_code):
                # Make sure it's not imported or defined locally
                if not re.search(rf'from .* import .*{tool_name}', tool_code) and \
                   not re.search(rf'def {tool_name}', tool_code):
                    issues.append(f"Tool '{tool_name}' called directly without tool_system - should use tool_system.call_tool('{tool_name}', ...)")
        
        # 2. Missing tool_system import when calling tools
        if 'tool_system.call_tool' in tool_code or 'ToolCallingSystem' in tool_code:
            if 'from scripts.tool_calling_system import ToolCallingSystem' not in tool_code:
                issues.append("Uses tool_system but missing import: from scripts.tool_calling_system import ToolCallingSystem")
        
        # 3. Using classes without importing them
        classes_to_check = {
            'StateManager': 'scripts.state_manager',
            'RepositoryAnalyzer': 'scripts.repository_analyzer',
            'GitHubIssueCreator': 'scripts.github_issue_creator',
            'TaskManager': 'scripts.task_manager'
        }
        
        for class_name, module in classes_to_check.items():
            if class_name + '(' in tool_code:  # Class is instantiated
                if f'from {module} import {class_name}' not in tool_code:
                    issues.append(f"Uses {class_name} but missing import: from {module} import {class_name}")
        
        # 4. Check for proper return format
        if 'return {' in tool_code:
            # Check if it returns the expected format
            if not ('"success":' in tool_code or "'success':" in tool_code):
                issues.append("Return value missing 'success' key - should return {'success': bool, 'result': data, 'error': str}")
        
        # 5. Check for unsafe data access patterns
        unsafe_patterns = [
            (r"result\['[^']+'\]\['[^']+'\]", "Use safe_get() for nested access: result['a']['b'] -> safe_get(result, 'a', 'b')"),
            (r'result\["[^"]+"\]\["[^"]+"\]', 'Use safe_get() for nested access: result["a"]["b"] -> safe_get(result, "a", "b")'),
            (r'\.get\([^)]+\)\[', "Chain of .get()[key] can fail - use safe_get() instead"),
        ]
        
        for pattern, message in unsafe_patterns:
            if re.search(pattern, tool_code):
                issues.append(f"Unsafe data access detected: {message}")
        
        # Try to parse the code
        try:
            ast.parse(tool_code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    async def _analyze_tool_file(self, tool_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze an existing tool file."""
        if not tool_path.exists():
            return None
        
        try:
            code = tool_path.read_text()
            return {
                'name': tool_path.stem,
                'path': str(tool_path),
                'code': code,
                'imports': re.findall(r'^(?:from|import)\s+(\S+)', code, re.MULTILINE)
            }
        except:
            return None
    
    async def _search_codebase(self, query: str) -> List[str]:
        """Search codebase for relevant files."""
        # Simple search implementation - could be enhanced
        results = []
        search_dirs = [Path("scripts"), Path("scripts/custom_tools")]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for file_path in search_dir.rglob("*.py"):
                    try:
                        content = file_path.read_text()
                        if query.lower() in content.lower():
                            results.append(str(file_path))
                    except:
                        pass
        
        return results[:10]
    
    def _extract_tool_code(self, response: str) -> str:
        """Extract Python code from the assistant's response."""
        # First, check for our specific markers
        if '<<<TOOL_CODE>>>' in response and '<<<END_TOOL_CODE>>>' in response:
            start_marker = '<<<TOOL_CODE>>>'
            end_marker = '<<<END_TOOL_CODE>>>'
            start_idx = response.find(start_marker) + len(start_marker)
            end_idx = response.find(end_marker)
            if start_idx < end_idx:
                code = response[start_idx:end_idx].strip()
                # Validate it looks like proper Python code
                if ('def ' in code or 'async def ' in code):
                    return code
        
        # Fallback: Check for ```python code blocks
        if '```python' in response:
            # Extract the last Python code block (which should be the complete tool)
            blocks = response.split('```python')
            if len(blocks) > 1:
                # Get the last code block
                for block in reversed(blocks[1:]):
                    # Remove the closing ``` if present
                    code = block.split('```')[0].strip()
                    # Check if this looks like a complete tool (has a function definition)
                    if 'def ' in code:
                        return code
        
        # Fallback: Try to extract code between certain patterns
        if 'def ' in response or 'async def ' in response:
            # Find the start of the code (usually starts with imports or __description__)
            lines = response.split('\n')
            code_lines = []
            in_code = False
            found_def = False
            
            for line in lines:
                # Start capturing when we see Python code patterns
                if not in_code and (line.strip().startswith('import ') or 
                                   line.strip().startswith('from ') or
                                   line.strip().startswith('__description__') or
                                   line.strip().startswith('#') and len(code_lines) == 0):
                    in_code = True
                
                if in_code:
                    # Stop if we hit non-code content after finding a function
                    if found_def and line.strip() and not line[0].isspace() and not any([
                        line.strip().startswith('#'),
                        line.strip().startswith('"""'),
                        line.strip().startswith("'''"),
                        line.strip().startswith('def '),
                        line.strip().startswith('async def '),
                        line.strip().startswith('class '),
                        line.strip().startswith('if __name__'),
                        line.strip() == ''
                    ]):
                        # Looks like we've hit explanatory text, stop here
                        break
                    
                    code_lines.append(line)
                    
                    if 'def ' in line or 'async def ' in line:
                        found_def = True
            
            if code_lines and found_def:
                return '\n'.join(code_lines).rstrip()
        
        # Last resort - return the entire response
        return response
    
    def _validate_discovery_happened(self, response: str) -> bool:
        """Check if the AI actually ran discovery commands."""
        discovery_indicators = [
            'COMMAND: execute_command',
            'execute_command(command=',
            'RESULT:',
            'DISCOVERY STEP',
            'grep',
            'def ',
            'class '
        ]
        
        # Count how many indicators are present
        indicators_found = sum(1 for indicator in discovery_indicators if indicator in response)
        
        # Need at least 3 indicators to consider discovery happened
        return indicators_found >= 3
    
    async def _run_discovery_phase(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Run discovery phase to find actual APIs and their return types."""
        import subprocess
        
        discovered = {
            'modules': {},
            'tools': [],
            'examples': {}
        }
        
        # Extract modules to discover from requirements
        modules_to_discover = self._extract_modules_from_requirements(requirements)
        
        try:
            # Discover methods for each requested module
            for module_name, module_path in modules_to_discover.items():
                if module_path and Path(module_path).exists():
                    result = subprocess.run(
                        ["grep", "-E", "(async def |def )", module_path],
                        capture_output=True, text=True
                    )
                    
                    if result.stdout:
                        discovered['modules'][module_name] = {}
                        for line in result.stdout.split('\n'):
                            if 'def ' in line and '__' not in line and '(' in line:
                                # Check if it's async
                                is_async = 'async def' in line
                                method_match = re.search(r'(?:async\s+)?def\s+(\w+)\s*\((.*?)\)', line)
                                if method_match:
                                    method_name = method_match.group(1)
                                    params = method_match.group(2)
                                    
                                    # Try to get return type hint
                                    return_type = None
                                    return_match = re.search(r'->\s*([^:]+):', line)
                                    if return_match:
                                        return_type = return_match.group(1).strip()
                                    
                                    discovered['modules'][module_name][method_name] = {
                                        'params': params,
                                        'async': is_async,
                                        'return_type': return_type
                                    }
                    
                    # Also discover example usage
                    discovered['examples'][module_name] = await self._discover_usage_examples(module_name, module_path)
            
            # Discover available tools based on requirements
            tool_keywords = self._extract_tool_keywords(requirements)
            if tool_keywords:
                # Build grep pattern from keywords
                pattern = '|'.join(tool_keywords)
                tools_cmd = f"ls scripts/custom_tools/*.py | xargs -I {{}} basename {{}} .py | grep -E '({pattern})' | head -20"
            else:
                # Fallback: list all tools
                tools_cmd = "ls scripts/custom_tools/*.py | xargs -I {} basename {} .py | head -20"
            
            tools_result = subprocess.run(
                tools_cmd,
                shell=True, capture_output=True, text=True
            )
            
            if tools_result.stdout:
                discovered['tools'] = [tool.strip() for tool in tools_result.stdout.split('\n') if tool.strip()]
                
        except Exception as e:
            print(f"Discovery error: {e}")
        
        return discovered
    
    def _extract_modules_from_requirements(self, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Extract module names and paths from requirements."""
        modules = {}
        
        # Check for explicitly mentioned modules in requirements
        if 'integrations' in requirements:
            for integration in requirements.get('integrations', []):
                # Try to find the module for this integration
                if isinstance(integration, str):
                    module_path = self.module_discovery.find_module(integration)
                    if module_path:
                        modules[integration] = module_path
        
        # Check functionality for common patterns
        functionality = requirements.get('functionality', '')
        if isinstance(functionality, str):
            functionality_lower = functionality.lower()
            
            # Common module patterns
            module_patterns = {
                'state': ('StateManager', 'scripts/state_manager.py'),
                'repository': ('RepositoryAnalyzer', 'scripts/repository_analyzer.py'),
                'github': ('GitHubIssueCreator', 'scripts/github_issue_creator.py'),
                'task': ('TaskManager', 'scripts/task_manager.py'),
                'workflow': ('WorkflowOrchestrator', 'scripts/workflow_orchestrator.py'),
                'redis': ('RedisStateAdapter', 'scripts/redis_state_adapter.py'),
                'ai': ('AIBrain', 'scripts/ai_brain.py')
            }
            
            for keyword, (module_name, module_path) in module_patterns.items():
                if keyword in functionality_lower and Path(module_path).exists():
                    modules[module_name] = module_path
        
        # If no specific modules found, use defaults based on common needs
        if not modules:
            # These are commonly needed modules
            default_modules = {
                'StateManager': 'scripts/state_manager.py',
                'ToolCallingSystem': 'scripts/tool_calling_system.py'
            }
            for name, path in default_modules.items():
                if Path(path).exists():
                    modules[name] = path
        
        return modules
    
    def _extract_tool_keywords(self, requirements: Dict[str, Any]) -> List[str]:
        """Extract keywords to search for relevant tools."""
        keywords = []
        
        # Extract from functionality description
        functionality = requirements.get('functionality', '')
        if isinstance(functionality, str):
            # Extract meaningful words (3+ chars, not common words)
            words = re.findall(r'\b[a-z]{3,}\b', functionality.lower())
            common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'will', 'can', 'should'}
            keywords.extend([w for w in words if w not in common_words][:5])
        
        # Add keywords from similar features
        similar = requirements.get('similar_features', [])
        if isinstance(similar, list):
            for feature in similar[:3]:
                if isinstance(feature, str):
                    feature_words = re.findall(r'\b[a-z]{3,}\b', feature.lower())
                    keywords.extend(feature_words[:2])
        
        # Remove duplicates and limit
        return list(set(keywords))[:10]
    
    def _generate_inventory_template(self, discovered_apis: Dict[str, Any]) -> str:
        """Generate dynamic inventory template based on discovered modules."""
        template_lines = []
        
        modules = discovered_apis.get('modules', {})
        for module_name in modules:
            template_lines.append(f"DISCOVERED METHODS FOR {module_name}:")
            template_lines.append(f"- [List ONLY methods you found via grep, with their exact signatures]")
            template_lines.append("")
        
        return '\n'.join(template_lines)
    
    def _format_discovered_modules(self, modules: Dict[str, Dict[str, Any]]) -> str:
        """Format discovered modules with async information."""
        formatted = {}
        for module_name, methods in modules.items():
            formatted[module_name] = {}
            for method_name, info in methods.items():
                if isinstance(info, dict):
                    formatted[module_name][method_name] = {
                        'params': info['params'],
                        'async': info['async'],
                        'signature': f"{'async ' if info['async'] else ''}def {method_name}({info['params']})"
                    }
                else:
                    # Old format compatibility
                    formatted[module_name][method_name] = info
        
        return json.dumps(formatted, indent=2)
    
    async def _discover_usage_examples(self, module_name: str, module_path: str) -> Dict[str, Any]:
        """Discover how a module is used by finding example calls in the codebase."""
        import subprocess
        examples = {}
        
        try:
            # Find files that import this module
            import_pattern = f"from .* import {module_name}|import .*{module_name}"
            result = subprocess.run(
                ["grep", "-r", "-l", import_pattern, "scripts/"],
                capture_output=True, text=True
            )
            
            if result.stdout:
                # Take first few files that use this module
                using_files = result.stdout.strip().split('\n')[:3]
                
                for file_path in using_files:
                    if file_path and Path(file_path).exists():
                        # Find actual usage patterns
                        usage_result = subprocess.run(
                            ["grep", "-A", "5", "-B", "2", f"{module_name}(", file_path],
                            capture_output=True, text=True
                        )
                        
                        if usage_result.stdout:
                            # Extract patterns like "result = analyzer.analyze_repository(...)"
                            patterns = re.findall(
                                rf'(\w+)\s*=\s*\w*\.?{module_name}\(\)\.(\w+)\((.*?)\)',
                                usage_result.stdout
                            )
                            for var_name, method_name, params in patterns:
                                if method_name not in examples:
                                    examples[method_name] = {
                                        'variable_name': var_name,
                                        'example_params': params,
                                        'source_file': file_path
                                    }
        except Exception as e:
            print(f"Error discovering examples: {e}")
        
        return examples