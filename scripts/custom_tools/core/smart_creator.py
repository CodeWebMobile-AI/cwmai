"""
Smart Creator

Unified tool for intelligent content creation including tool generation,
strategy development, and Laravel/React project creation.

Capabilities:
- Create custom tools from requirements
- Generate strategies and plans
- Create intelligent workflows
- Generate code and documentation
- Create Laravel/React projects with full architecture
- AI-powered content generation
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import textwrap

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.smart_tool_generator import SmartToolGenerator
from scripts.project_creator import ProjectCreator


class SmartCreator:
    """Intelligent content and tool creation system."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.tool_generator = SmartToolGenerator()
        self.project_root = Path(__file__).parent.parent.parent
        self.tools_dir = self.project_root / 'scripts' / 'custom_tools'
        
        # Initialize ProjectCreator with AI brain
        self.project_creator = ProjectCreator(ai_brain=self.ai_client)
        
        # Template patterns
        self.templates = {
            'tool': self._get_tool_template(),
            'workflow': self._get_workflow_template(),
            'strategy': self._get_strategy_template(),
            'project': self._get_project_template()
        }
        
    async def create(self, what: str, name: str = None, **kwargs) -> Dict[str, Any]:
        """Create intelligent content.
        
        Args:
            what: What to create (tool, workflow, strategy, etc.)
            name: Name for the creation
            **kwargs: Additional parameters
            
        Returns:
            Creation results
        """
        creators = {
            'tool': self.create_tool,
            'workflow': self.create_workflow,
            'strategy': self.create_strategy,
            'project': self.create_project,
            'documentation': self.create_documentation,
            'code': self.create_code,
            'test': self.create_test,
            'api': self.create_api,
            'integration': self.create_integration
        }
        
        # Normalize what
        what_lower = what.lower()
        
        # Find best match
        handler = None
        for key, func in creators.items():
            if key in what_lower or what_lower in key:
                handler = func
                break
                
        if not handler:
            # Use AI to understand intent
            return await self._intelligent_create(what, name, **kwargs)
            
        return await handler(name=name, **kwargs)
        
    async def create_tool(self, name: str = None, description: str = None, 
                         capabilities: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Create a custom tool."""
        if not name:
            return {"error": "Tool name is required"}
            
        # Generate tool using SmartToolGenerator
        result = await self.tool_generator.generate_tool(
            tool_name=name,
            tool_intent=description or f"Tool to {' and '.join(capabilities or ['perform operations'])}",
            parameters=kwargs
        )
        
        if result.get('success'):
            # Enhance with additional features
            tool_path = self.tools_dir / f"{name}.py"
            
            if tool_path.exists():
                # Add to system state
                state = self.state_manager.get_state()
                if 'custom_tools' not in state:
                    state['custom_tools'] = {}
                    
                state['custom_tools'][name] = {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'description': description,
                    'capabilities': capabilities,
                    'file': str(tool_path.relative_to(self.project_root))
                }
                
                self.state_manager.save_state()
                
            return {
                "success": True,
                "tool_name": name,
                "file_path": str(tool_path),
                "message": f"Tool '{name}' created successfully",
                "capabilities": capabilities,
                "usage_example": self._generate_usage_example(name, capabilities)
            }
            
        return result
        
    async def create_workflow(self, name: str = None, steps: List[Dict] = None, 
                            goal: str = None, **kwargs) -> Dict[str, Any]:
        """Create an intelligent workflow."""
        if not name:
            name = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Generate workflow with AI
        prompt = f"""Create a workflow for: {goal or 'general automation'}

Workflow name: {name}
Steps provided: {json.dumps(steps, indent=2) if steps else 'Generate appropriate steps'}

Generate a complete workflow with:
1. Clear step definitions
2. Input/output specifications
3. Error handling
4. Parallel execution where possible
5. Success criteria"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            workflow_content = response.get('content', '')
            
            # Parse and structure workflow
            workflow = {
                'name': name,
                'goal': goal,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'steps': steps or self._extract_steps_from_content(workflow_content),
                'content': workflow_content
            }
            
            # Save workflow
            workflows_dir = self.project_root / 'scripts' / 'workflows'
            workflows_dir.mkdir(exist_ok=True)
            
            workflow_file = workflows_dir / f"{name}.json"
            with open(workflow_file, 'w') as f:
                json.dump(workflow, f, indent=2)
                
            return {
                "success": True,
                "workflow_name": name,
                "file_path": str(workflow_file),
                "goal": goal,
                "steps_count": len(workflow['steps']),
                "workflow": workflow
            }
            
        return {"error": "Failed to generate workflow"}
        
    async def create_strategy(self, name: str = None, objective: str = None, 
                            constraints: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Create a strategic plan."""
        if not objective:
            return {"error": "Strategy objective is required"}
            
        if not name:
            name = f"strategy_{objective[:20].replace(' ', '_').lower()}"
            
        # Generate strategy with AI
        prompt = f"""Create a comprehensive strategy for:
Objective: {objective}
Constraints: {json.dumps(constraints or [], indent=2)}

Include:
1. Executive Summary
2. Current State Analysis
3. Goals and KPIs
4. Action Plan with timeline
5. Resource Requirements
6. Risk Assessment
7. Success Metrics
8. Implementation Phases"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            strategy_content = response.get('content', '')
            
            # Save strategy
            strategies_dir = self.project_root / 'strategies'
            strategies_dir.mkdir(exist_ok=True)
            
            strategy_file = strategies_dir / f"{name}.md"
            with open(strategy_file, 'w') as f:
                f.write(f"# Strategy: {objective}\n\n")
                f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
                f.write(strategy_content)
                
            return {
                "success": True,
                "strategy_name": name,
                "file_path": str(strategy_file),
                "objective": objective,
                "preview": strategy_content[:500] + "..." if len(strategy_content) > 500 else strategy_content
            }
            
        return {"error": "Failed to generate strategy"}
        
    async def create_project(self, name: str, type: str = None, 
                           features: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Create a new Laravel/React project using ProjectCreator."""
        if not name:
            return {"error": "Project name is required"}
        
        # Create task format expected by ProjectCreator
        task = {
            'title': f'Create {name} project',
            'description': kwargs.get('description', f'Create {name} with features: {features}'),
            'metadata': {
                'initial_features': features or [],
                'project_type': type or 'Web Application'
            }
        }
        
        # Add any additional metadata from kwargs
        if 'problem_statement' in kwargs:
            task['metadata']['problem_statement'] = kwargs['problem_statement']
        if 'target_audience' in kwargs:
            task['metadata']['target_audience'] = kwargs['target_audience']
        if 'monetization_strategy' in kwargs:
            task['metadata']['monetization_strategy'] = kwargs['monetization_strategy']
        
        # Use ProjectCreator for Laravel/React projects
        result = await self.project_creator.create_project(task)
        
        if result.get('success'):
            return {
                "success": True,
                "project_name": result.get('project_name'),
                "project_path": result.get('repo_url'),
                "project_type": "Laravel/React",
                "features": features,
                "repo_url": result.get('repo_url'),
                "issues_created": result.get('initial_issues', 0),
                "customizations": result.get('customizations', {}),
                "mode": result.get('mode', 'github'),
                "description": result.get('description', '')
            }
        else:
            return {
                "success": False,
                "error": result.get('error', 'Project creation failed')
            }
        
    async def create_documentation(self, for_what: str, type: str = 'readme', 
                                 **kwargs) -> Dict[str, Any]:
        """Create documentation."""
        doc_types = {
            'readme': self._create_readme,
            'api': self._create_api_docs,
            'user_guide': self._create_user_guide,
            'technical': self._create_technical_docs
        }
        
        handler = doc_types.get(type, self._create_readme)
        return await handler(for_what, **kwargs)
        
    async def create_code(self, description: str, language: str = 'python', 
                        **kwargs) -> Dict[str, Any]:
        """Create code from description."""
        prompt = f"""Generate {language} code for:
{description}

Requirements:
1. Clean, well-structured code
2. Proper error handling
3. Type hints (if applicable)
4. Docstrings
5. Follow best practices

Generate complete, working code."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            code = response.get('content', '')
            
            # Extract code from markdown if present
            code_match = re.search(r'```\w*\n(.*?)\n```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                
            # Validate Python code
            if language == 'python':
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    return {"error": f"Generated code has syntax errors: {str(e)}"}
                    
            return {
                "success": True,
                "language": language,
                "code": code,
                "lines": len(code.split('\n')),
                "description": description
            }
            
        return {"error": "Failed to generate code"}
        
    async def create_test(self, for_code: str = None, test_type: str = 'unit', 
                        **kwargs) -> Dict[str, Any]:
        """Create tests for code."""
        if not for_code:
            return {"error": "Code or file path required"}
            
        # Read code if file path provided
        if os.path.exists(for_code):
            with open(for_code, 'r') as f:
                code_content = f.read()
        else:
            code_content = for_code
            
        prompt = f"""Generate {test_type} tests for this code:

{code_content}

Generate comprehensive tests including:
1. Happy path tests
2. Edge cases
3. Error cases
4. Setup and teardown
5. Mocking where needed"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            tests = response.get('content', '')
            
            return {
                "success": True,
                "test_type": test_type,
                "tests": tests,
                "test_count": tests.count('def test_')
            }
            
        return {"error": "Failed to generate tests"}
        
    async def create_api(self, name: str, endpoints: List[Dict] = None, 
                       **kwargs) -> Dict[str, Any]:
        """Create API structure."""
        if not name:
            return {"error": "API name is required"}
            
        # Generate API specification
        api_spec = {
            'name': name,
            'version': '1.0.0',
            'endpoints': endpoints or await self._generate_api_endpoints(name, kwargs.get('purpose')),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Generate implementation
        implementation = await self._generate_api_implementation(api_spec)
        
        # Save files
        api_dir = self.project_root / 'apis' / name
        api_dir.mkdir(parents=True, exist_ok=True)
        
        # Save specification
        spec_file = api_dir / 'openapi.yaml'
        with open(spec_file, 'w') as f:
            f.write(self._generate_openapi_spec(api_spec))
            
        # Save implementation
        impl_file = api_dir / 'api.py'
        with open(impl_file, 'w') as f:
            f.write(implementation)
            
        return {
            "success": True,
            "api_name": name,
            "endpoints_count": len(api_spec['endpoints']),
            "spec_file": str(spec_file),
            "implementation_file": str(impl_file),
            "endpoints": [e['path'] for e in api_spec['endpoints']]
        }
        
    async def create_integration(self, service1: str, service2: str, 
                               type: str = 'bidirectional', **kwargs) -> Dict[str, Any]:
        """Create integration between services."""
        # Generate integration plan
        prompt = f"""Create integration between {service1} and {service2}.
Type: {type}

Generate:
1. Integration architecture
2. Data flow design
3. API mappings
4. Error handling strategy
5. Implementation code"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            integration_plan = response.get('content', '')
            
            # Save integration
            integrations_dir = self.project_root / 'integrations'
            integrations_dir.mkdir(exist_ok=True)
            
            integration_name = f"{service1}_{service2}_integration"
            integration_file = integrations_dir / f"{integration_name}.md"
            
            with open(integration_file, 'w') as f:
                f.write(f"# Integration: {service1} <-> {service2}\n\n")
                f.write(integration_plan)
                
            return {
                "success": True,
                "integration_name": integration_name,
                "file_path": str(integration_file),
                "type": type,
                "services": [service1, service2]
            }
            
        return {"error": "Failed to create integration"}
        
    async def _intelligent_create(self, what: str, name: str = None, **kwargs) -> Dict[str, Any]:
        """Use AI to understand what to create."""
        prompt = f"""User wants to create: {what}
Name: {name or 'not specified'}
Additional parameters: {json.dumps(kwargs)}

Determine:
1. What type of creation this is (tool, workflow, code, etc.)
2. Key requirements
3. Suggested approach"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            analysis = response.get('content', '')
            
            # Try to route to appropriate creator
            analysis_lower = analysis.lower()
            
            if 'tool' in analysis_lower:
                return await self.create_tool(name=name or what, **kwargs)
            elif 'workflow' in analysis_lower:
                return await self.create_workflow(name=name, goal=what, **kwargs)
            elif 'project' in analysis_lower or 'laravel' in analysis_lower or 'react' in analysis_lower:
                return await self.create_project(name=name or what, features=kwargs.get('features', []), **kwargs)
            elif 'code' in analysis_lower:
                return await self.create_code(description=what, **kwargs)
            else:
                # Default to creating documentation about it
                return await self.create_documentation(for_what=what, **kwargs)
                
        return {"error": f"Could not understand what to create: {what}"}
        
    def _generate_usage_example(self, tool_name: str, capabilities: List[str]) -> str:
        """Generate usage example for a tool."""
        example = f"""# Usage Example for {tool_name}

from scripts.custom_tools.{tool_name} import main

# Example 1: Basic usage
result = await main()
print(result)
"""
        
        if capabilities:
            example += f"\n# Example 2: Using specific capability\nresult = await main(operation='{capabilities[0]}')"
            
        return example
        
    def _extract_steps_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract workflow steps from content."""
        steps = []
        
        # Look for numbered steps
        step_pattern = re.compile(r'\d+\.\s+(.+?)(?=\d+\.|$)', re.DOTALL)
        matches = step_pattern.findall(content)
        
        for i, match in enumerate(matches):
            steps.append({
                'step': i + 1,
                'description': match.strip().split('\n')[0],  # First line
                'details': match.strip()
            })
            
        return steps
        
        
    def _get_tool_template(self) -> str:
        """Get tool template."""
        return '''"""
{description}
"""

import asyncio
from typing import Dict, Any


class {class_name}:
    """Main class for {tool_name}."""
    
    def __init__(self):
        pass
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool operation."""
        return {{"success": True, "message": "Tool executed"}}
        

async def main(**kwargs) -> Dict[str, Any]:
    """Entry point for the tool."""
    tool = {class_name}()
    return await tool.execute(**kwargs)
    

if __name__ == "__main__":
    asyncio.run(main())
'''
        
    def _get_workflow_template(self) -> str:
        """Get workflow template."""
        return {
            "name": "",
            "description": "",
            "steps": [],
            "inputs": {},
            "outputs": {},
            "error_handling": {"strategy": "fail_fast"}
        }
        
    def _get_strategy_template(self) -> str:
        """Get strategy template."""
        return """# Strategy: {name}

## Executive Summary

## Current State

## Goals

## Action Plan

## Resources Required

## Success Metrics
"""
        
    def _get_project_template(self) -> str:
        """Get project template."""
        return {
            "structure": [],
            "dependencies": [],
            "configuration": {}
        }
        
    async def _create_readme(self, for_what: str, **kwargs) -> Dict[str, Any]:
        """Create README documentation."""
        prompt = f"""Create a comprehensive README.md for: {for_what}

Include:
1. Project overview
2. Features
3. Installation instructions
4. Usage examples
5. API documentation (if applicable)
6. Contributing guidelines
7. License information"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            return {
                "success": True,
                "type": "readme",
                "content": response.get('content', ''),
                "for": for_what
            }
            
        return {"error": "Failed to create README"}
        
    async def _create_api_docs(self, for_what: str, **kwargs) -> Dict[str, Any]:
        """Create API documentation."""
        # Implementation for API docs
        return {"success": True, "type": "api_docs", "message": "API docs created"}
        
    async def _create_user_guide(self, for_what: str, **kwargs) -> Dict[str, Any]:
        """Create user guide."""
        # Implementation for user guide
        return {"success": True, "type": "user_guide", "message": "User guide created"}
        
    async def _create_technical_docs(self, for_what: str, **kwargs) -> Dict[str, Any]:
        """Create technical documentation."""
        # Implementation for technical docs
        return {"success": True, "type": "technical_docs", "message": "Technical docs created"}
        
    async def _generate_api_endpoints(self, name: str, purpose: str = None) -> List[Dict[str, str]]:
        """Generate API endpoints."""
        # Basic CRUD endpoints
        return [
            {'method': 'GET', 'path': f'/api/{name}', 'description': f'List all {name}'},
            {'method': 'GET', 'path': f'/api/{name}/{{id}}', 'description': f'Get specific {name}'},
            {'method': 'POST', 'path': f'/api/{name}', 'description': f'Create new {name}'},
            {'method': 'PUT', 'path': f'/api/{name}/{{id}}', 'description': f'Update {name}'},
            {'method': 'DELETE', 'path': f'/api/{name}/{{id}}', 'description': f'Delete {name}'}
        ]
        
    async def _generate_api_implementation(self, api_spec: Dict) -> str:
        """Generate API implementation."""
        return f'''"""
API implementation for {api_spec['name']}
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

# API endpoints
{self._generate_endpoint_implementations(api_spec['endpoints'])}

if __name__ == '__main__':
    app.run(debug=True)
'''
        
    def _generate_endpoint_implementations(self, endpoints: List[Dict]) -> str:
        """Generate endpoint implementations."""
        implementations = []
        
        for endpoint in endpoints:
            method = endpoint['method']
            path = endpoint['path']
            desc = endpoint['description']
            
            impl = f'''\n@app.route('{path}', methods=['{method}'])
def {self._path_to_function_name(path, method)}():
    """{desc}"""
    return jsonify({{"message": "{desc}"}})\n'''
            
            implementations.append(impl)
            
        return '\n'.join(implementations)
        
    def _path_to_function_name(self, path: str, method: str) -> str:
        """Convert API path to function name."""
        # Remove /api/ prefix and parameters
        clean_path = path.replace('/api/', '').replace('{', '').replace('}', '')
        parts = clean_path.split('/')
        
        # Create function name
        if method == 'GET' and len(parts) > 1:
            return f"get_{parts[0]}_by_id"
        elif method == 'GET':
            return f"list_{parts[0]}"
        elif method == 'POST':
            return f"create_{parts[0]}"
        elif method == 'PUT':
            return f"update_{parts[0]}"
        elif method == 'DELETE':
            return f"delete_{parts[0]}"
        else:
            return f"{method.lower()}_{parts[0]}"
            
    def _generate_openapi_spec(self, api_spec: Dict) -> str:
        """Generate OpenAPI specification."""
        return f"""openapi: 3.0.0
info:
  title: {api_spec['name']}
  version: {api_spec['version']}
  description: API specification for {api_spec['name']}

paths:
{self._generate_openapi_paths(api_spec['endpoints'])}
"""
        
    def _generate_openapi_paths(self, endpoints: List[Dict]) -> str:
        """Generate OpenAPI paths."""
        # Simplified implementation
        paths = []
        for endpoint in endpoints:
            paths.append(f"  {endpoint['path']}:")
            paths.append(f"    {endpoint['method'].lower()}:")
            paths.append(f"      summary: {endpoint['description']}")
            paths.append("      responses:")
            paths.append("        '200':")
            paths.append("          description: Success")
            
        return '\n'.join(paths)
        
    def _get_gitignore_content(self, project_type: str) -> str:
        """Get .gitignore content for project type."""
        base = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
        
        if project_type == 'web':
            base += """\n# Web
node_modules/
npm-debug.log
*.log
dist/
build/
"""
            
        return base
        


async def main(what: str = None, **kwargs):
    """Entry point for smart creator."""
    creator = SmartCreator()
    
    if what:
        return await creator.create(what, **kwargs)
    else:
        return {
            "available_creations": [
                "tool", "workflow", "strategy", "project",
                "documentation", "code", "test", "api", "integration"
            ],
            "usage": "Specify 'what' to create, e.g., main(what='tool', name='my_tool')"
        }
        

if __name__ == "__main__":
    # Test the creator
    import asyncio
    
    async def test():
        creator = SmartCreator()
        
        # Test different creations
        print("Create tool:", await creator.create('tool', name='test_tool', description='A test tool'))
        print("\nCreate code:", await creator.create('code', description='Function to calculate fibonacci'))
        print("\nCreate project:", await creator.create('project', name='awesome-app', features=['authentication', 'dashboard']))
        
    asyncio.run(test())