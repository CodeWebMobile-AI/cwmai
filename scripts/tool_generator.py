"""
Tool Generator

Automatically creates new tools and capabilities based on identified needs.
Uses research to find best implementations and integrates them into the system.
"""

import os
import ast
import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import inspect
import textwrap


class ToolGenerator:
    """Creates new tools based on system needs."""
    
    def __init__(self, ai_brain, capability_analyzer):
        """Initialize tool generator.
        
        Args:
            ai_brain: AI brain for code generation and research
            capability_analyzer: System capability analyzer
        """
        self.ai_brain = ai_brain
        self.capability_analyzer = capability_analyzer
        self.base_path = Path(__file__).parent
        self.generated_tools = []
        self.tool_templates = self._load_tool_templates()
        
    async def generate_tool(self, 
                           tool_spec: Dict[str, Any],
                           research_context: bool = True) -> Dict[str, Any]:
        """Generate a new tool based on specifications.
        
        Args:
            tool_spec: Tool specifications including purpose, inputs, outputs
            research_context: Whether to research best practices
            
        Returns:
            Generation result with file path and integration details
        """
        print(f"Generating tool: {tool_spec.get('name', 'Unknown')}")
        
        # Research best practices if requested
        implementation_research = {}
        if research_context:
            implementation_research = await self._research_implementation(tool_spec)
        
        # Design tool architecture
        tool_design = await self._design_tool_architecture(tool_spec, implementation_research)
        
        # Generate tool code
        tool_code = await self._generate_tool_code(tool_design)
        
        # Validate generated code
        validation_result = await self._validate_tool_code(tool_code)
        
        if not validation_result['valid']:
            # Try to fix issues
            tool_code = await self._fix_code_issues(tool_code, validation_result['issues'])
            
        # Save tool
        file_path = await self._save_tool(tool_spec['name'], tool_code)
        
        # Generate integration code
        integration = await self._generate_integration_code(tool_spec, file_path)
        
        # Record generation
        result = {
            'name': tool_spec['name'],
            'file_path': file_path,
            'purpose': tool_spec['purpose'],
            'integration': integration,
            'validation': validation_result,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'research_used': bool(implementation_research)
        }
        
        self.generated_tools.append(result)
        
        return result
    
    async def identify_needed_tools(self) -> List[Dict[str, Any]]:
        """Identify tools the system needs but doesn't have.
        
        Returns:
            List of needed tool specifications
        """
        # Get current capabilities
        capabilities = await self.capability_analyzer.analyze_current_capabilities()
        gaps = await self.capability_analyzer.identify_gaps()
        
        # Get recent task failures
        # This would connect to task history in production
        recent_failures = []
        
        prompt = f"""
        Identify specific tools that would fill capability gaps and improve the system.
        
        Current Capabilities:
        {json.dumps(capabilities.get('capability_coverage', {}), indent=2)}
        
        Identified Gaps:
        {json.dumps(gaps.get('missing_capabilities', []), indent=2)}
        
        Recent Task Failures (if any):
        {json.dumps(recent_failures, indent=2)}
        
        For each needed tool, specify:
        1. name: Tool name (e.g., "dependency_analyzer", "code_visualizer")
        2. purpose: What problem it solves
        3. inputs: Required inputs
        4. outputs: Expected outputs
        5. priority: How important (high/medium/low)
        6. category: Type of tool (analysis/generation/optimization/integration)
        7. estimated_complexity: simple/moderate/complex
        
        Focus on tools that would:
        - Fill identified capability gaps
        - Improve task success rates
        - Enable new types of tasks
        - Increase system efficiency
        
        Return as JSON array of tool specifications.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        needed_tools = self._parse_json_response(response)
        
        if isinstance(needed_tools, list):
            # Sort by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            needed_tools.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
            
        return needed_tools if isinstance(needed_tools, list) else []
    
    async def _research_implementation(self, tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Research best practices for implementing the tool.
        
        Args:
            tool_spec: Tool specifications
            
        Returns:
            Research findings
        """
        research_query = f"""
        Research best practices for implementing: {tool_spec['name']}
        Purpose: {tool_spec['purpose']}
        
        Find:
        1. Common design patterns for this type of tool
        2. Python libraries that could help
        3. Performance considerations
        4. Security best practices
        5. Example implementations or similar tools
        """
        
        # Use AI brain for research
        if self.ai_brain:
            prompt = f"""
            Research implementation approaches for a {tool_spec['name']} tool.
            
            Tool Purpose: {tool_spec['purpose']}
            Category: {tool_spec.get('category', 'general')}
            
            Provide:
            1. Recommended design patterns
            2. Useful Python libraries (stdlib preferred)
            3. Code structure suggestions
            4. Common pitfalls to avoid
            5. Performance optimization tips
            
            Format as JSON with sections for each topic.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            return self._parse_json_response(response)
        
        return {}
    
    async def _design_tool_architecture(self, 
                                       tool_spec: Dict[str, Any],
                                       research: Dict[str, Any]) -> Dict[str, Any]:
        """Design the tool's architecture.
        
        Args:
            tool_spec: Tool specifications
            research: Implementation research
            
        Returns:
            Tool design
        """
        prompt = f"""
        Design the architecture for this tool:
        
        Tool Specification:
        {json.dumps(tool_spec, indent=2)}
        
        Research Findings:
        {json.dumps(research, indent=2)}
        
        Design should include:
        1. Class structure (if needed)
        2. Main functions and their signatures
        3. Error handling approach
        4. Integration points with existing system
        5. Testing approach
        
        Follow these principles:
        - Keep it simple and focused
        - Use type hints
        - Include docstrings
        - Handle errors gracefully
        - Make it easy to test
        - Follow existing code patterns in the system
        
        Return as JSON with:
        - classes: List of classes with methods
        - functions: Standalone functions
        - imports: Required imports
        - integration: How it connects to the system
        - test_cases: Example test cases
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    async def _generate_tool_code(self, design: Dict[str, Any]) -> str:
        """Generate the actual tool code.
        
        Args:
            design: Tool design
            
        Returns:
            Generated Python code
        """
        prompt = f"""
        Generate complete Python code for this tool based on the design:
        
        Design:
        {json.dumps(design, indent=2)}
        
        Requirements:
        1. Include all imports at the top
        2. Add comprehensive docstrings
        3. Use type hints throughout
        4. Include error handling
        5. Follow PEP 8 style
        6. Make it production-ready
        7. Include a main() function for testing
        
        The code should be complete and runnable.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        
        # Extract code from response
        code = response.get('content', '')
        
        # Clean up code
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
            
        return code.strip()
    
    async def _validate_tool_code(self, code: str) -> Dict[str, Any]:
        """Validate generated tool code.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation result
        """
        issues = []
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return {'valid': False, 'issues': issues}
        
        # Check imports
        tree = ast.parse(code)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # Check for forbidden imports
        forbidden = {'os', 'subprocess', 'eval', 'exec', '__import__'}
        forbidden_found = imports.intersection(forbidden)
        if forbidden_found:
            issues.append(f"Forbidden imports: {forbidden_found}")
        
        # Check for docstrings
        has_module_docstring = ast.get_docstring(tree) is not None
        if not has_module_docstring:
            issues.append("Missing module docstring")
        
        # Check for type hints
        functions_without_hints = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.returns and node.name != '__init__':
                    functions_without_hints.append(node.name)
        
        if functions_without_hints:
            issues.append(f"Functions without return type hints: {functions_without_hints}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'imports': list(imports),
            'has_tests': 'test_' in code or 'unittest' in code
        }
    
    async def _fix_code_issues(self, code: str, issues: List[str]) -> str:
        """Try to fix code issues.
        
        Args:
            code: Original code
            issues: List of issues
            
        Returns:
            Fixed code
        """
        prompt = f"""
        Fix these issues in the Python code:
        
        Issues:
        {json.dumps(issues, indent=2)}
        
        Original Code:
        ```python
        {code}
        ```
        
        Fix all issues while maintaining functionality.
        Return only the fixed code.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        
        # Extract fixed code
        fixed_code = response.get('content', '')
        if '```python' in fixed_code:
            fixed_code = fixed_code.split('```python')[1].split('```')[0]
        elif '```' in fixed_code:
            fixed_code = fixed_code.split('```')[1].split('```')[0]
            
        return fixed_code.strip()
    
    async def _save_tool(self, name: str, code: str) -> str:
        """Save tool to file.
        
        Args:
            name: Tool name
            code: Tool code
            
        Returns:
            File path
        """
        # Create tools directory if needed
        tools_dir = self.base_path / "generated_tools"
        tools_dir.mkdir(exist_ok=True)
        
        # Generate file name
        file_name = f"{name.lower().replace(' ', '_')}.py"
        file_path = tools_dir / file_name
        
        # Add generation header
        header = f'''"""
Generated Tool: {name}
Generated at: {datetime.now(timezone.utc).isoformat()}
Generated by: AI Tool Generator

This tool was automatically generated based on identified system needs.
"""

'''
        
        # Write file
        with open(file_path, 'w') as f:
            f.write(header + code)
        
        print(f"Saved tool to: {file_path}")
        
        return str(file_path)
    
    async def _generate_integration_code(self, 
                                        tool_spec: Dict[str, Any],
                                        file_path: str) -> Dict[str, Any]:
        """Generate code to integrate tool into system.
        
        Args:
            tool_spec: Tool specifications
            file_path: Path to saved tool
            
        Returns:
            Integration details
        """
        prompt = f"""
        Generate integration code for this new tool:
        
        Tool: {tool_spec['name']}
        Purpose: {tool_spec['purpose']}
        File Path: {file_path}
        
        Generate:
        1. Import statement
        2. Initialization code
        3. Usage example
        4. How to add it to existing modules
        
        Consider integration with:
        - capability_analyzer.py (if it's an analysis tool)
        - ai_brain.py (if it enhances AI capabilities)
        - task_manager.py (if it helps with tasks)
        - self_modification_engine.py (if it's for self-improvement)
        
        Return as JSON with code snippets.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        integration = self._parse_json_response(response)
        
        # Add basic integration if parsing failed
        if not integration:
            module_name = Path(file_path).stem
            integration = {
                'import_statement': f"from generated_tools.{module_name} import *",
                'initialization': f"# Initialize {tool_spec['name']}",
                'usage_example': f"# Use {tool_spec['name']} for {tool_spec['purpose']}"
            }
        
        return integration
    
    def _load_tool_templates(self) -> Dict[str, str]:
        """Load common tool templates.
        
        Returns:
            Dictionary of templates
        """
        templates = {
            'analyzer': '''
class {ClassName}:
    """Analyzes {target} for {purpose}."""
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Perform analysis.
        
        Args:
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        results = {{}}
        
        # Analysis logic here
        
        return results
''',
            'generator': '''
class {ClassName}:
    """Generates {output} based on {input}."""
    
    def __init__(self):
        """Initialize generator."""
        pass
    
    def generate(self, spec: Dict[str, Any]) -> Any:
        """Generate output.
        
        Args:
            spec: Generation specifications
            
        Returns:
            Generated output
        """
        # Generation logic here
        
        return None
''',
            'optimizer': '''
class {ClassName}:
    """Optimizes {target} for {goal}."""
    
    def __init__(self):
        """Initialize optimizer."""
        pass
    
    def optimize(self, target: Any) -> Any:
        """Perform optimization.
        
        Args:
            target: Target to optimize
            
        Returns:
            Optimized result
        """
        # Optimization logic here
        
        return target
'''
        }
        
        return templates
    
    async def generate_tool_tests(self, tool_path: str) -> str:
        """Generate tests for a tool.
        
        Args:
            tool_path: Path to tool file
            
        Returns:
            Path to test file
        """
        # Read tool code
        with open(tool_path, 'r') as f:
            tool_code = f.read()
        
        # Parse to understand structure
        tree = ast.parse(tool_code)
        
        # Find classes and functions
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({'name': node.name, 'methods': methods})
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                functions.append(node.name)
        
        # Generate test code
        prompt = f"""
        Generate comprehensive unit tests for this tool:
        
        Tool Code Structure:
        Classes: {json.dumps(classes, indent=2)}
        Functions: {json.dumps(functions, indent=2)}
        
        Generate tests that:
        1. Test all public methods
        2. Test edge cases
        3. Test error handling
        4. Use unittest framework
        5. Include setUp and tearDown if needed
        6. Have descriptive test names
        
        Make tests thorough but practical.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        test_code = response.get('content', '')
        
        # Clean up code
        if '```python' in test_code:
            test_code = test_code.split('```python')[1].split('```')[0]
        
        # Save test file
        test_path = tool_path.replace('.py', '_test.py')
        
        with open(test_path, 'w') as f:
            f.write(test_code.strip())
        
        return test_path
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed JSON or empty dict/list
        """
        content = response.get('content', '')
        
        try:
            # Try to find JSON in response
            import re
            
            # Look for JSON array
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                return json.loads(array_match.group())
            
            # Look for JSON object
            obj_match = re.search(r'\{[\s\S]*\}', content)
            if obj_match:
                return json.loads(obj_match.group())
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        
        return {}
    
    def get_generated_tools_summary(self) -> Dict[str, Any]:
        """Get summary of generated tools.
        
        Returns:
            Summary of tool generation activities
        """
        return {
            'total_generated': len(self.generated_tools),
            'tools': [
                {
                    'name': tool['name'],
                    'purpose': tool['purpose'],
                    'file_path': tool['file_path'],
                    'timestamp': tool['timestamp']
                }
                for tool in self.generated_tools
            ],
            'categories': self._categorize_tools(),
            'success_rate': self._calculate_success_rate()
        }
    
    def _categorize_tools(self) -> Dict[str, int]:
        """Categorize generated tools.
        
        Returns:
            Tool counts by category
        """
        categories = {}
        
        for tool in self.generated_tools:
            # Simple categorization based on name/purpose
            name_lower = tool['name'].lower()
            
            if 'analyz' in name_lower:
                category = 'analysis'
            elif 'generat' in name_lower:
                category = 'generation'
            elif 'optim' in name_lower:
                category = 'optimization'
            elif 'integrat' in name_lower:
                category = 'integration'
            else:
                category = 'other'
            
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def _calculate_success_rate(self) -> float:
        """Calculate tool generation success rate.
        
        Returns:
            Success rate
        """
        if not self.generated_tools:
            return 0.0
        
        successful = sum(
            1 for tool in self.generated_tools 
            if tool.get('validation', {}).get('valid', False)
        )
        
        return successful / len(self.generated_tools)


async def demonstrate_tool_generator():
    """Demonstrate tool generation capabilities."""
    print("=== Tool Generator Demo ===\n")
    
    # Mock components for demo
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt):
            # Return mock responses for demo
            if "identify specific tools" in prompt:
                return {
                    'content': '''[
                        {
                            "name": "code_complexity_analyzer",
                            "purpose": "Analyze code complexity and suggest simplifications",
                            "inputs": ["file_path", "complexity_threshold"],
                            "outputs": ["complexity_score", "hotspots", "suggestions"],
                            "priority": "high",
                            "category": "analysis",
                            "estimated_complexity": "moderate"
                        }
                    ]'''
                }
            elif "Design the architecture" in prompt:
                return {
                    'content': '''{
                        "classes": [{
                            "name": "CodeComplexityAnalyzer",
                            "methods": ["analyze_file", "calculate_complexity", "find_hotspots"]
                        }],
                        "functions": ["get_complexity_metrics"],
                        "imports": ["ast", "typing"],
                        "integration": "Integrates with capability_analyzer",
                        "test_cases": ["test_simple_function", "test_complex_class"]
                    }'''
                }
            else:
                return {'content': 'Generated code here'}
    
    ai_brain = MockAIBrain()
    generator = ToolGenerator(ai_brain, None, None)
    
    # Identify needed tools
    print("Identifying needed tools...")
    needed_tools = await generator.identify_needed_tools()
    
    print(f"\nFound {len(needed_tools)} needed tools:")
    for tool in needed_tools:
        print(f"- {tool['name']}: {tool['purpose']}")
    
    # Generate a tool
    if needed_tools:
        print(f"\nGenerating tool: {needed_tools[0]['name']}")
        result = await generator.generate_tool(needed_tools[0])
        
        print(f"\nGeneration result:")
        print(f"- File: {result['file_path']}")
        print(f"- Valid: {result['validation']['valid']}")
        print(f"- Integration: {result['integration']}")
    
    # Show summary
    print("\n=== Tool Generation Summary ===")
    summary = generator.get_generated_tools_summary()
    print(f"Total tools generated: {summary['total_generated']}")
    print(f"Success rate: {summary['success_rate']:.0%}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_tool_generator())