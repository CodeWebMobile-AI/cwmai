#!/usr/bin/env python3
"""
Integration module to enhance tool generation in the CWMAI system
Provides better context and validation for AI-generated tools
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from scripts.improved_tool_generator import ImprovedToolGenerator
from scripts.tool_generation_templates import ToolGenerationTemplates
from scripts.enhanced_tool_validation import EnhancedToolValidator


class EnhancedToolGeneratorIntegration:
    """Integrates improved tool generation into the system."""
    
    def __init__(self, tool_calling_system=None):
        self.tool_system = tool_calling_system
        self.generator = ImprovedToolGenerator()
        self.templates = ToolGenerationTemplates()
        self.validator = EnhancedToolValidator()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def create_tool_from_query(self, query: str) -> Dict[str, Any]:
        """Create a tool from a natural language query."""
        
        # Analyze query to extract tool requirements
        tool_spec = await self._analyze_query(query)
        
        if not tool_spec['understood']:
            return {
                "success": False,
                "error": "Could not understand the query",
                "suggestion": "Please be more specific about what the tool should do"
            }
        
        # Generate tool name from query
        tool_name = self._generate_tool_name(query)
        
        # Create enhanced prompt
        prompt = self.templates.create_enhanced_prompt(
            name=tool_name,
            description=tool_spec['description'],
            requirements=tool_spec['requirements'],
            category=tool_spec.get('category')
        )
        
        # Generate tool with validation
        max_attempts = 3
        for attempt in range(max_attempts):
            result = await self.generator.generate_tool(
                name=tool_name,
                description=tool_spec['description'],
                requirements=tool_spec['requirements']
            )
            
            if result['success']:
                # Save the tool
                tool_path = await self._save_tool(tool_name, result['code'])
                
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "tool_path": str(tool_path),
                    "description": tool_spec['description'],
                    "validation": result['validation']
                }
            
            # If validation failed, try to understand why
            self.logger.warning(f"Attempt {attempt + 1} failed: {result['validation']['issues']}")
            
            # Add context about errors for next attempt
            tool_spec['requirements'] += f"\n\nPREVIOUS ATTEMPT FAILED WITH: {result['validation']['issues']}"
        
        return {
            "success": False,
            "error": "Failed to generate valid tool after multiple attempts",
            "last_issues": result['validation']['issues']
        }
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a natural language query to extract tool requirements."""
        
        # Use AI to understand the query
        analysis_prompt = f"""Analyze this user query and determine what kind of tool they want:

Query: "{query}"

Extract:
1. What the tool should do (description)
2. Specific requirements and features
3. Category (file_operations, data_analysis, system_operations, git_operations, or other)
4. Input parameters needed
5. Expected output format

Format as JSON with keys: understood (boolean), description, requirements, category, parameters, output_format
"""
        
        from scripts.http_ai_client import HTTPAIClient
        ai_client = HTTPAIClient()
        
        response = await ai_client.generate_enhanced_response(analysis_prompt)
        
        try:
            import json
            content = response.get('content', '{}')
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '{' in content:
                # Find JSON object
                start = content.find('{')
                end = content.rfind('}') + 1
                content = content[start:end]
            
            analysis = json.loads(content)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze query: {e}")
            
            # Fallback analysis
            return {
                "understood": True,
                "description": query,
                "requirements": f"Tool to: {query}",
                "category": "other",
                "parameters": {},
                "output_format": "dictionary with results"
            }
    
    def _generate_tool_name(self, query: str) -> str:
        """Generate a valid tool name from a query."""
        import re
        
        # Convert to lowercase and replace spaces with underscores
        name = query.lower()
        
        # Extract key action words
        action_words = ['count', 'find', 'get', 'list', 'check', 'analyze', 
                       'calculate', 'search', 'create', 'delete', 'update']
        
        # Find the main action
        main_action = None
        for word in action_words:
            if word in name:
                main_action = word
                break
        
        if not main_action:
            main_action = 'process'
        
        # Clean up the name
        name = re.sub(r'[^a-z0-9_\s]', '', name)
        words = name.split()
        
        # Build tool name
        if main_action in words:
            words.remove(main_action)
        
        # Take meaningful words
        meaningful_words = [w for w in words if len(w) > 2][:3]
        
        tool_name = f"{main_action}_{'_'.join(meaningful_words)}"
        
        # Ensure valid Python identifier
        tool_name = re.sub(r'[^a-z0-9_]', '_', tool_name)
        tool_name = re.sub(r'_+', '_', tool_name).strip('_')
        
        # Limit length
        if len(tool_name) > 30:
            tool_name = tool_name[:30].rstrip('_')
        
        return tool_name or 'custom_tool'
    
    async def _save_tool(self, name: str, code: str) -> Path:
        """Save a generated tool to the custom tools directory."""
        custom_tools_dir = Path("scripts/custom_tools")
        custom_tools_dir.mkdir(exist_ok=True)
        
        tool_path = custom_tools_dir / f"{name}.py"
        tool_path.write_text(code)
        
        self.logger.info(f"Saved tool to: {tool_path}")
        
        return tool_path
    
    async def enhance_existing_tool(self, tool_name: str) -> Dict[str, Any]:
        """Enhance an existing tool with better error handling and validation."""
        
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        
        if not tool_path.exists():
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        
        current_code = tool_path.read_text()
        
        # Analyze current issues
        validation = await self.validator.validate_tool_code(current_code)
        
        if validation['valid']:
            return {"success": True, "message": "Tool is already valid"}
        
        # Create enhancement prompt
        enhancement_prompt = f"""Enhance this existing tool to fix the following issues:

CURRENT CODE:
{current_code}

ISSUES TO FIX:
{json.dumps(validation['issues'], indent=2)}

ENHANCEMENT REQUIREMENTS:
1. Fix all identified issues
2. Add comprehensive error handling
3. Improve input validation
4. Add missing imports
5. Ensure consistent return format
6. Add helpful error messages
7. Maintain backward compatibility

{self.templates.get_import_context()}

Generate ONLY the enhanced Python code.
"""
        
        from scripts.http_ai_client import HTTPAIClient
        ai_client = HTTPAIClient()
        
        response = await ai_client.generate_enhanced_response(enhancement_prompt)
        
        if response.get('content'):
            enhanced_code = self._extract_code(response['content'])
            
            # Validate enhanced code
            new_validation = await self.validator.validate_tool_code(enhanced_code)
            
            if new_validation['valid']:
                # Backup and save
                backup_path = tool_path.with_suffix('.py.backup')
                backup_path.write_text(current_code)
                
                tool_path.write_text(enhanced_code)
                
                return {
                    "success": True,
                    "message": f"Enhanced tool {tool_name}",
                    "backup": str(backup_path),
                    "fixed_issues": validation['issues']
                }
        
        return {"success": False, "error": "Failed to enhance tool"}
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from AI response."""
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            code = response.split('```')[1].split('```')[0]
        else:
            code = response
        
        return code.strip()


async def test_enhanced_generation():
    """Test the enhanced tool generation."""
    integration = EnhancedToolGeneratorIntegration()
    
    test_queries = [
        "Count how many Python files have more than 100 lines",
        "Find all functions that are longer than 50 lines",
        "Calculate the average file size in each directory"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = await integration.create_tool_from_query(query)
        
        if result['success']:
            print(f"✓ Created tool: {result['tool_name']}")
            print(f"  Path: {result['tool_path']}")
            print(f"  Description: {result['description']}")
        else:
            print(f"✗ Failed: {result['error']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_generation())