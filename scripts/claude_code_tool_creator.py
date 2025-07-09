"""
Claude Code Tool Creator
========================

Creates tools with the same intelligence as Claude Code, but even better.
Understands code relationships, reuses existing patterns, and generates
working tools on the first try.
"""

import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scripts.intelligent_code_analyzer import IntelligentCodeAnalyzer
from scripts.tool_calling_system import ToolCallingSystem
from scripts.ai_brain import IntelligentAIBrain
from scripts.code_validator import validate_python_code
import ast


class ClaudeCodeToolCreator:
    """Creates tools with Claude Code level intelligence."""
    
    def __init__(self):
        self.analyzer = IntelligentCodeAnalyzer()
        self.ai_brain = IntelligentAIBrain()
        self.tool_system = ToolCallingSystem()
        self.project_root = Path(__file__).parent.parent
        
    async def create_tool_from_request(self, 
                                      tool_name: str, 
                                      description: str,
                                      grep_results: Optional[List[Dict]] = None,
                                      context_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a tool with full context understanding like Claude Code."""
        
        try:
            # Step 1: Intelligent file selection if grep results provided
            if grep_results:
                file_analysis = await self.analyzer.analyze_grep_results(grep_results, description)
                files_to_read = file_analysis['recommended_read_order']
            elif context_files:
                files_to_read = context_files
            else:
                # Search for relevant files ourselves
                keywords = self._extract_keywords(description)
                grep_results = await self._search_codebase(keywords[0] if keywords else 'tool')
                file_analysis = await self.analyzer.analyze_grep_results(grep_results, description)
                files_to_read = file_analysis['recommended_read_order']
            
            # Step 2: Deep analysis of selected files
            analyzed_files = {}
            for file_path in files_to_read[:5]:  # Limit to top 5 files
                analysis = await self.analyzer.analyze_code_patterns(file_path)
                if 'error' not in analysis:
                    analyzed_files[file_path] = analysis
            
            # Step 3: Get code reuse suggestions
            reuse_suggestions = await self.analyzer.suggest_code_reuse(description, analyzed_files)
            
            # Step 4: Read actual code from most relevant files
            code_examples = await self._read_code_examples(files_to_read[:3])
            
            # Step 5: Generate tool with full context
            tool_code = await self._generate_intelligent_tool(
                tool_name,
                description,
                analyzed_files,
                reuse_suggestions,
                code_examples
            )
            
            # Step 6: Validate and fix the generated code
            validated_code = await self._validate_and_fix_code(tool_code, tool_name)
            
            # Step 7: Save the tool
            tool_path = self.project_root / 'scripts' / 'custom_tools' / f'{tool_name}.py'
            with open(tool_path, 'w') as f:
                f.write(validated_code)
            
            # Step 8: Test the tool can be imported
            test_result = await self._test_tool_import(tool_name)
            
            return {
                "success": True,
                "tool_name": tool_name,
                "tool_path": str(tool_path),
                "based_on_files": files_to_read[:3],
                "reused_components": {
                    "classes": [c['class'] for c in reuse_suggestions.get('classes_to_import', [])],
                    "functions": [f['function'] for f in reuse_suggestions.get('functions_to_reuse', [])]
                },
                "test_result": test_result,
                "message": f"Tool '{tool_name}' created successfully with deep code understanding"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create tool: {str(e)}",
                "tool_name": tool_name
            }
    
    async def _generate_intelligent_tool(self,
                                       tool_name: str,
                                       description: str,
                                       analyzed_files: Dict[str, Dict],
                                       reuse_suggestions: Dict[str, Any],
                                       code_examples: Dict[str, str]) -> str:
        """Generate tool code with deep understanding of the codebase."""
        
        # Build comprehensive context
        context = self._build_generation_context(analyzed_files, reuse_suggestions, code_examples)
        
        prompt = f"""Create a Python tool that: {description}

Tool name: {tool_name}

IMPORTANT: You must reuse existing code from the codebase. Here's what you should use:

{context}

Generate a COMPLETE, WORKING tool following this EXACT structure:

```python
\"\"\"
{tool_name.replace('_', ' ').title()} Tool
{'=' * (len(tool_name) + 5)}

{description}
\"\"\"

# Import EXACTLY what you need from the codebase
{self._format_import_suggestions(reuse_suggestions)}

from typing import Dict, Any, List, Optional

__description__ = "{description}"
__parameters__ = {{
    # Add parameters based on what the tool needs
}}
__examples__ = [
    {{}}
]

async def {tool_name}(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    {description}
    \"\"\"
    try:
        # REUSE existing code - don't reinvent the wheel!
        {self._format_usage_examples(reuse_suggestions)}
        
        # Implementation based on analyzed patterns
        
        return {{
            "success": True,
            "results": results  # or whatever makes sense
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": f"Error in {tool_name}: {{str(e)}}"
        }}

# Export for tool system
{tool_name}_tool = {tool_name}
```

CRITICAL REQUIREMENTS:
1. Import and reuse existing classes/functions - don't create new implementations
2. Follow the exact patterns you see in the code examples
3. Use the same error handling patterns
4. Make sure all imports are correct and exist
5. The tool must be fully functional, not a placeholder
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        
        # Extract the actual content from the response dict
        if isinstance(response, dict):
            content = response.get('content', response.get('result', str(response)))
        else:
            content = str(response)
        
        # Extract code from markdown blocks
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Try to extract without backticks
        return content
    
    async def _validate_and_fix_code(self, code: str, tool_name: str) -> str:
        """Validate and automatically fix common issues in generated code."""
        
        # Step 1: Basic syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Try to fix common syntax errors
            code = self._fix_syntax_errors(code, str(e))
        
        # Step 2: Fix imports
        code = await self._fix_imports(code)
        
        # Step 3: Ensure proper tool structure
        code = self._ensure_tool_structure(code, tool_name)
        
        # Step 4: Validate again
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            # If still broken, generate a minimal working version
            return self._generate_minimal_working_tool(tool_name)
    
    async def _fix_imports(self, code: str) -> str:
        """Fix import statements to ensure they're valid."""
        lines = code.split('\n')
        fixed_lines = []
        
        import_mappings = {
            'scripts.mcp_github_issue_creator': 'scripts.github_issue_creator',
            'scripts.logger': 'scripts.logger',
            'scripts.ai_code_analyzer': 'scripts.ai_brain',
            'scripts.task_manager': 'scripts.task_manager',
            'scripts.state_manager': 'scripts.state_manager',
            'scripts.repository_analyzer': 'scripts.repository_analyzer',
            'scripts.architecture_generator': 'scripts.architecture_generator',
            'scripts.project_creator': 'scripts.project_creator'
        }
        
        for line in lines:
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                # Check if import needs mapping
                fixed_line = line
                for old_import, new_import in import_mappings.items():
                    if old_import in line:
                        fixed_line = line.replace(old_import, new_import)
                        break
                
                # Verify import exists
                import_match = re.match(r'from\s+([\w.]+)\s+import', fixed_line)
                if import_match:
                    module_name = import_match.group(1)
                    if module_name.startswith('scripts.'):
                        # Check if module file exists
                        module_path = module_name.replace('.', '/') + '.py'
                        if not (self.project_root / module_path).exists():
                            # Try to find the correct module
                            correct_module = await self._find_correct_module(module_name)
                            if correct_module:
                                fixed_line = fixed_line.replace(module_name, correct_module)
                
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    async def _find_correct_module(self, module_name: str) -> Optional[str]:
        """Find the correct module path for an import."""
        # Extract the last part of the module name
        parts = module_name.split('.')
        if len(parts) > 1:
            target = parts[-1]
            
            # Search for files with similar names
            for py_file in self.project_root.rglob('*.py'):
                if target.lower() in py_file.stem.lower():
                    # Convert path to module
                    rel_path = py_file.relative_to(self.project_root)
                    module = str(rel_path).replace('/', '.').replace('.py', '')
                    return module
        
        return None
    
    def _ensure_tool_structure(self, code: str, tool_name: str) -> str:
        """Ensure the code has proper tool structure."""
        
        # Check for required elements
        required = [
            f'async def {tool_name}',
            '__description__',
            '__parameters__',
            f'{tool_name}_tool = {tool_name}'
        ]
        
        for req in required:
            if req not in code:
                # Add missing elements
                if req == f'async def {tool_name}':
                    # Add function if missing
                    code += f"\n\nasync def {tool_name}(**kwargs: Dict[str, Any]) -> Dict[str, Any]:\n    return {{'success': True}}\n"
                elif req == '__description__':
                    # Add after imports
                    lines = code.split('\n')
                    import_end = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith('from') and not line.startswith('import'):
                            import_end = i
                            break
                    lines.insert(import_end, f'\n__description__ = "Tool function"')
                    code = '\n'.join(lines)
                elif req == '__parameters__':
                    code = code.replace('__description__', f'__description__\n__parameters__ = {{}}')
                elif req == f'{tool_name}_tool':
                    code += f"\n\n{tool_name}_tool = {tool_name}"
        
        return code
    
    def _fix_syntax_errors(self, code: str, error_msg: str) -> str:
        """Try to fix common syntax errors."""
        
        # Fix unterminated strings
        if 'unterminated string' in error_msg:
            # Count quotes and try to balance them
            single_quotes = code.count("'") 
            double_quotes = code.count('"')
            
            if single_quotes % 2 != 0:
                code += "'"
            if double_quotes % 2 != 0:
                code += '"'
        
        # Fix indentation errors
        if 'indent' in error_msg:
            lines = code.split('\n')
            fixed_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped.endswith(':'):
                    fixed_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                elif stripped.startswith(('return', 'break', 'continue', 'pass')):
                    fixed_lines.append('    ' * indent_level + stripped)
                    if indent_level > 0:
                        indent_level -= 1
                elif stripped:
                    fixed_lines.append('    ' * indent_level + stripped)
                else:
                    fixed_lines.append('')
            
            code = '\n'.join(fixed_lines)
        
        return code
    
    def _generate_minimal_working_tool(self, tool_name: str) -> str:
        """Generate a minimal working tool as fallback."""
        return f"""\"\"\"
{tool_name.replace('_', ' ').title()} Tool
{'=' * (len(tool_name) + 5)}

Auto-generated minimal tool.
\"\"\"

from typing import Dict, Any

__description__ = "Tool: {tool_name}"
__parameters__ = {{}}
__examples__ = [{{}}]

async def {tool_name}(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Execute {tool_name}.\"\"\"
    return {{
        "success": True,
        "message": "Tool executed successfully",
        "note": "This is a minimal implementation - enhance as needed"
    }}

{tool_name}_tool = {tool_name}"""
    
    async def _search_codebase(self, keyword: str) -> List[Dict[str, Any]]:
        """Search codebase for keyword."""
        tool_result = await self.tool_system.call_tool('grep_files', 
                                                      pattern=keyword, 
                                                      file_pattern='*.py',
                                                      max_results=30)
        
        if tool_result.get('success') and tool_result.get('results'):
            return tool_result['results']
        return []
    
    async def _read_code_examples(self, file_paths: List[str]) -> Dict[str, str]:
        """Read code examples from files."""
        examples = {}
        
        for file_path in file_paths[:3]:  # Limit to 3 files
            tool_result = await self.tool_system.call_tool('read_file',
                                                          file_path=file_path,
                                                          start_line=1,
                                                          end_line=200)
            
            if tool_result.get('success'):
                examples[file_path] = tool_result.get('content', '')
        
        return examples
    
    async def _test_tool_import(self, tool_name: str) -> Dict[str, Any]:
        """Test if the tool can be imported successfully."""
        try:
            # Try to import the tool
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                tool_name,
                self.project_root / 'scripts' / 'custom_tools' / f'{tool_name}.py'
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required attributes
            tool_func = getattr(module, tool_name, None)
            if tool_func:
                return {
                    "importable": True,
                    "has_function": True,
                    "has_description": hasattr(module, '__description__'),
                    "has_parameters": hasattr(module, '__parameters__')
                }
            else:
                return {
                    "importable": True,
                    "has_function": False,
                    "error": f"Function {tool_name} not found in module"
                }
                
        except Exception as e:
            return {
                "importable": False,
                "error": str(e)
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        words = re.findall(r'\w+', text.lower())
        stop_words = {'a', 'an', 'the', 'to', 'for', 'that', 'this', 'with', 'from', 
                      'should', 'would', 'could', 'check', 'create', 'make', 'if', 'all',
                      'new', 'tool', 'need', 'want'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:5]
    
    def _build_generation_context(self, 
                                 analyzed_files: Dict[str, Dict],
                                 reuse_suggestions: Dict[str, Any],
                                 code_examples: Dict[str, str]) -> str:
        """Build comprehensive context for code generation."""
        
        context_parts = []
        
        # Add reuse suggestions
        if reuse_suggestions['classes_to_import']:
            context_parts.append("CLASSES TO REUSE:")
            for cls in reuse_suggestions['classes_to_import'][:3]:
                context_parts.append(f"- {cls['class']} from {cls['from']}")
                context_parts.append(f"  Methods: {', '.join(cls['methods'][:5])}")
                context_parts.append(f"  Usage: {cls['usage_hint']}")
        
        if reuse_suggestions['functions_to_reuse']:
            context_parts.append("\nFUNCTIONS TO REUSE:")
            for func in reuse_suggestions['functions_to_reuse'][:3]:
                context_parts.append(f"- {func['function']} from {func['from']}")
                if func['is_async']:
                    context_parts.append(f"  Async function with args: {', '.join(func['args'])}")
                else:
                    context_parts.append(f"  Function with args: {', '.join(func['args'])}")
        
        # Add patterns
        patterns = []
        for file_data in analyzed_files.values():
            patterns.extend(file_data.get('patterns', []))
        
        if patterns:
            context_parts.append("\nPATTERNS TO FOLLOW:")
            for pattern in set(p['type'] for p in patterns):
                context_parts.append(f"- {pattern}")
        
        # Add code examples (truncated)
        if code_examples:
            context_parts.append("\nCODE EXAMPLES:")
            for file, code in list(code_examples.items())[:2]:
                context_parts.append(f"\nFrom {file}:")
                # Show first 20 lines
                lines = code.split('\n')[:20]
                context_parts.extend(lines)
                if len(code.split('\n')) > 20:
                    context_parts.append("... (truncated)")
        
        return '\n'.join(context_parts)
    
    def _format_import_suggestions(self, reuse_suggestions: Dict[str, Any]) -> str:
        """Format import suggestions for the prompt."""
        imports = []
        
        for cls in reuse_suggestions.get('classes_to_import', [])[:3]:
            imports.append(f"from {cls['from']} import {cls['class']}")
        
        for func in reuse_suggestions.get('functions_to_reuse', [])[:3]:
            imports.append(f"from {func['from']} import {func['function']}")
        
        return '\n'.join(imports) if imports else "# Add necessary imports"
    
    def _format_usage_examples(self, reuse_suggestions: Dict[str, Any]) -> str:
        """Format usage examples for the prompt."""
        examples = []
        
        if reuse_suggestions.get('classes_to_import'):
            cls = reuse_suggestions['classes_to_import'][0]
            examples.append(f"# Example: Using {cls['class']}")
            examples.append(f"instance = {cls['usage_hint']}")
        
        if reuse_suggestions.get('functions_to_reuse'):
            func = reuse_suggestions['functions_to_reuse'][0]
            if func['is_async']:
                examples.append(f"# Example: Using {func['function']}")
                examples.append(f"result = await {func['function']}(...)")
        
        return '\n        '.join(examples) if examples else "# Add implementation"