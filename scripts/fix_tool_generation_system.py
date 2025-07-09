"""
Fix Tool Generation System - Intelligently patches the existing system
to prevent unnecessary complexity and imports
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ToolGenerationFixer")


class ToolGenerationSystemFixer:
    """Fixes the tool generation system to be smarter about imports and complexity."""
    
    def __init__(self):
        self.scripts_path = Path(__file__).parent
        self.smart_generator_path = self.scripts_path / "smart_tool_generator.py"
        
    async def analyze_problem(self) -> Dict[str, Any]:
        """Analyze the current problems with tool generation."""
        problems = {
            "overly_broad_rules": [],
            "unnecessary_dependencies": [],
            "complexity_issues": [],
            "prompt_issues": []
        }
        
        # Read the smart tool generator
        if self.smart_generator_path.exists():
            content = self.smart_generator_path.read_text()
            
            # Find problematic patterns
            if "For repository analysis: MUST use GitHub API via MCPGitHubIssueCreator" in content:
                problems["overly_broad_rules"].append(
                    "Forces GitHub API for ALL repository analysis, even when not needed"
                )
            
            if "NEVER analyze current directory as a substitute" in content:
                problems["overly_broad_rules"].append(
                    "Prevents simple file checking operations"
                )
                
            # Check the validation rules
            if 'issues.append("Analyzes code without using GitHub API' in content:
                problems["complexity_issues"].append(
                    "Validator forces GitHub API usage for simple file operations"
                )
        
        return problems
    
    def create_intelligent_analyzer(self) -> str:
        """Create an intelligent requirement analyzer."""
        return '''
    def _analyze_tool_requirements_intelligently(self, tool_name: str, description: str, 
                                                capabilities: List[str]) -> Dict[str, Any]:
        """Intelligently analyze what a tool ACTUALLY needs based on its specific purpose."""
        
        # Combine all context
        full_context = f"{tool_name} {description} {' '.join(capabilities)}".lower()
        
        requirements = {
            "needs_github_api": False,
            "needs_file_operations": False,
            "needs_state_management": False,
            "needs_subprocess": False,
            "primary_purpose": "unknown",
            "suggested_approach": "simple",
            "avoid_imports": []
        }
        
        # Determine primary purpose first
        if "architecture" in full_context and "check" in full_context:
            requirements["primary_purpose"] = "file_checking"
            requirements["needs_file_operations"] = True
            requirements["suggested_approach"] = "use gh CLI to check file existence"
            requirements["avoid_imports"] = ["MCPGitHubIssueCreator", "AICodeAnalyzer"]
            
        elif "create issue" in full_context or "github issue" in full_context:
            requirements["primary_purpose"] = "github_issue_management"
            requirements["needs_github_api"] = True
            requirements["suggested_approach"] = "use MCPGitHubIssueCreator for issue operations"
            
        elif "analyze" in full_context and "code" in full_context:
            # Only use GitHub API if explicitly analyzing GitHub repos
            if "github" in full_context or "repository metrics" in full_context:
                requirements["needs_github_api"] = True
            else:
                requirements["needs_file_operations"] = True
                requirements["avoid_imports"] = ["MCPGitHubIssueCreator"]
                
        elif "count" in full_context or "list" in full_context:
            requirements["primary_purpose"] = "data_aggregation"
            requirements["needs_state_management"] = True
            requirements["avoid_imports"] = ["MCPGitHubIssueCreator", "AICodeAnalyzer"]
            
        # Be very specific about when GitHub API is needed
        github_specific_keywords = [
            "create issue", "close issue", "update issue", "github api",
            "pull request", "github metrics", "repository statistics from github"
        ]
        
        if not any(keyword in full_context for keyword in github_specific_keywords):
            requirements["needs_github_api"] = False
            requirements["avoid_imports"].append("MCPGitHubIssueCreator")
            
        return requirements
'''
    
    def create_smart_prompt_builder(self) -> str:
        """Create a smart prompt builder that generates focused prompts."""
        return '''
    def _build_intelligent_prompt(self, tool_name: str, description: str, 
                                 capabilities: List[str], requirements: Dict[str, Any]) -> str:
        """Build an intelligent prompt based on actual requirements."""
        
        # Start with the core request
        prompt = f"""Create a Python tool that does EXACTLY what is requested - no more, no less.

Tool Name: {tool_name}
Description: {description}
Capabilities:
{chr(10).join(f'- {cap}' for cap in capabilities)}

PRIMARY PURPOSE: {requirements.get('primary_purpose', 'general')}
SUGGESTED APPROACH: {requirements.get('suggested_approach', 'keep it simple')}
"""

        # Add specific instructions based on requirements
        if requirements.get('avoid_imports'):
            prompt += f"""
DO NOT IMPORT: {', '.join(requirements['avoid_imports'])}
These modules are NOT needed for this specific task.
"""

        # Add appropriate imports section
        prompt += "\nREQUIRED IMPORTS:\n"
        
        if requirements.get('needs_file_operations'):
            prompt += """- from pathlib import Path
- import os
- import subprocess (for gh CLI commands)
"""
        
        if requirements.get('needs_state_management'):
            prompt += "- from scripts.state_manager import StateManager\n"
            
        if requirements.get('needs_github_api'):
            prompt += """- from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
(ONLY if actually creating/managing GitHub issues)
"""
        
        prompt += "- from scripts.conversational_ai_logger import setup_logger\n"
        prompt += "- Standard library imports as needed\n"
        
        # Add examples based on purpose
        if requirements['primary_purpose'] == 'file_checking':
            prompt += """
EXAMPLE for checking files in repositories:
```python
# Use gh CLI to check if file exists
cmd = f"gh api repos/{repo_name}/contents/ARCHITECTURE.md --jq '.name' 2>/dev/null"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
exists = result.returncode == 0
```
"""
        
        prompt += """
GOLDEN RULES:
1. KISS - Keep It Simple, Stupid
2. Only import what you actually use
3. Don't add features that weren't requested
4. Use the simplest solution that works
5. If you don't need to create GitHub issues, don't import GitHub modules

Return a complete, working tool that does exactly what was requested.
"""
        
        return prompt
'''
    
    async def patch_smart_generator(self) -> Dict[str, Any]:
        """Patch the smart tool generator with intelligent analysis."""
        try:
            if not self.smart_generator_path.exists():
                return {
                    "success": False,
                    "error": "smart_tool_generator.py not found"
                }
            
            content = self.smart_generator_path.read_text()
            
            # Find where to insert our intelligent analyzer
            class_start = content.find("class SmartToolGenerator:")
            if class_start == -1:
                return {"success": False, "error": "Could not find SmartToolGenerator class"}
            
            # Find the generate_tool method
            generate_method_start = content.find("async def generate_tool(", class_start)
            if generate_method_start == -1:
                return {"success": False, "error": "Could not find generate_tool method"}
            
            # Insert our intelligent analyzer before generate_tool
            insert_position = content.rfind("\n", class_start, generate_method_start)
            
            # Create the new content
            new_methods = self.create_intelligent_analyzer() + "\n" + self.create_smart_prompt_builder()
            
            # Also update the generate_tool method to use our intelligent approach
            updated_generate = '''
    async def generate_tool(self, tool_name: str, tool_intent: str, capabilities: List[str] = None) -> Dict[str, Any]:
        """Generate a tool with INTELLIGENT dependency handling."""
        
        if capabilities is None:
            capabilities = [tool_intent]  # Use intent as capability if none provided
            
        # Use intelligent analysis instead of simple keyword matching
        requirements = self._analyze_tool_requirements_intelligently(
            tool_name, tool_intent, capabilities
        )
        
        # Build intelligent prompt
        prompt = self._build_intelligent_prompt(
            tool_name, tool_intent, capabilities, requirements
        )
        
        try:
            # Get AI response
            response_dict = await self.ai_client.generate_enhanced_response(prompt)
            
            if isinstance(response_dict, dict):
                response = response_dict.get('content', '') or response_dict.get('result', '') or response_dict.get('response', '')
            else:
                response = str(response_dict)
            
            # Extract code
            code = self._extract_code_from_response(response)
            
            # Simple validation - don't over-validate
            if requirements.get('avoid_imports'):
                for avoid in requirements['avoid_imports']:
                    if avoid in code:
                        logger.warning(f"Generated code contains unnecessary import: {avoid}")
                        # Remove the unnecessary import
                        code = re.sub(f".*{avoid}.*\\n", "", code)
            
            # Save the tool
            tool_path = self.custom_tools_dir / f"{tool_name}.py"
            tool_path.write_text(code)
            
            return {
                'success': True,
                'tool_name': tool_name,
                'tool_path': str(tool_path),
                'message': f'Tool {tool_name} created successfully',
                'approach': requirements.get('suggested_approach', 'simple')
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tool: {e}")
            return {'success': False, 'error': str(e)}
'''
            
            # Create backup
            backup_path = self.smart_generator_path.with_suffix('.py.backup')
            backup_path.write_text(content)
            
            # Insert our improvements
            new_content = (
                content[:insert_position] + 
                "\n" + new_methods + "\n" +
                content[insert_position:generate_method_start] +
                updated_generate +
                content[content.find("\n", generate_method_start + 1000):]  # Skip old generate_tool
            )
            
            # Write the updated file
            self.smart_generator_path.write_text(new_content)
            
            return {
                "success": True,
                "message": "Successfully patched smart_tool_generator.py",
                "backup": str(backup_path)
            }
            
        except Exception as e:
            logger.error(f"Error patching smart generator: {e}")
            return {"success": False, "error": str(e)}
    
    async def fix_validation_rules(self) -> Dict[str, Any]:
        """Fix overly strict validation rules."""
        try:
            content = self.smart_generator_path.read_text()
            
            # Fix the overly broad validation rule
            old_rule = '''if 'calculate_lines_of_code' in code or 'CodeAnalyzer' in code:
            if 'MCPGitHubIssueCreator' not in code and 'GitHubIssueCreator' not in code:
                issues.append("Analyzes code without using GitHub API (must use MCPGitHubIssueCreator)")'''
            
            new_rule = '''if 'calculate_lines_of_code' in code or 'CodeAnalyzer' in code:
            # Only require GitHub API if actually analyzing GitHub repositories
            if 'github' in code.lower() or 'repository statistics' in code.lower():
                if 'MCPGitHubIssueCreator' not in code and 'GitHubIssueCreator' not in code:
                    issues.append("Analyzing GitHub repos requires GitHub API")'''
            
            if old_rule in content:
                content = content.replace(old_rule, new_rule)
                self.smart_generator_path.write_text(content)
                return {"success": True, "message": "Fixed validation rules"}
            
            return {"success": True, "message": "Validation rules already fixed or not found"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


async def main():
    """Run the fixer."""
    fixer = ToolGenerationSystemFixer()
    
    print("🔧 Analyzing tool generation problems...")
    problems = await fixer.analyze_problem()
    print(f"Found issues: {json.dumps(problems, indent=2)}")
    
    print("\n🔨 Patching smart tool generator...")
    result = await fixer.patch_smart_generator()
    print(f"Patch result: {result}")
    
    print("\n🔨 Fixing validation rules...")
    validation_result = await fixer.fix_validation_rules()
    print(f"Validation fix result: {validation_result}")
    
    print("\n✅ Tool generation system has been improved!")
    print("The system will now:")
    print("- Only use GitHub API when actually needed")
    print("- Generate simpler, more focused tools")
    print("- Avoid unnecessary imports")
    print("- Use the most appropriate solution for each task")


if __name__ == "__main__":
    asyncio.run(main())