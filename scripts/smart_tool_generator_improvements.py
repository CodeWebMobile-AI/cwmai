"""
Smart Tool Generator Improvements
Apply targeted improvements to make tool generation smarter
"""

import re
from pathlib import Path
from typing import Dict, List, Any

def apply_smart_improvements():
    """Apply smart improvements to the tool generator."""
    
    generator_path = Path(__file__).parent / "smart_tool_generator.py"
    
    if not generator_path.exists():
        print("smart_tool_generator.py not found")
        return False
        
    content = generator_path.read_text()
    
    # Fix 1: Update the analyze_intent_requirements to be smarter
    old_analysis = '''def _analyze_intent_requirements(self, intent: str) -> bool:
        """Analyze the intent to determine if GitHub API is required."""
        intent_lower = intent.lower()
        
        # Keywords that indicate repository/code analysis
        github_api_keywords = [
            'repository', 'repo', 'github', 'code analysis', 'project statistics',
            'lines of code', 'loc', 'cyclomatic complexity', 'code duplication',
            'project size', 'code metrics', 'repository metrics', 'analyze project',
            'calculate statistics', 'project analysis', 'code statistics',
            'repository analysis', 'analyze code', 'code quality'
        ]'''
    
    new_analysis = '''def _analyze_intent_requirements(self, intent: str) -> bool:
        """Analyze the intent to determine if GitHub API is required."""
        intent_lower = intent.lower()
        
        # Only require GitHub API for SPECIFIC GitHub operations
        github_api_keywords = [
            'create github issue', 'close github issue', 'update github issue',
            'pull request', 'github api', 'github metrics', 'github statistics'
        ]'''
    
    content = content.replace(old_analysis, new_analysis)
    
    # Fix 2: Update validation to be less strict
    old_validation = '''# Check for repository analysis without GitHub API
        if 'calculate_lines_of_code' in code or 'CodeAnalyzer' in code:
            if 'MCPGitHubIssueCreator' not in code and 'GitHubIssueCreator' not in code:
                issues.append("Analyzes code without using GitHub API (must use MCPGitHubIssueCreator)")'''
    
    new_validation = '''# Check for repository analysis without GitHub API
        # Only require GitHub API if explicitly working with GitHub
        if ('github' in code.lower() or 'repository statistics' in code.lower()) and 'calculate_lines_of_code' in code:
            if 'MCPGitHubIssueCreator' not in code and 'GitHubIssueCreator' not in code:
                issues.append("GitHub repository analysis requires GitHub API")'''
    
    content = content.replace(old_validation, new_validation)
    
    # Fix 3: Update the additional_context logic
    old_context = '''# Build additional context based on requirements
        additional_context = ""
        if requires_github_api:
            additional_context = """
REQUIRED: This tool MUST use the GitHub API because it involves repository analysis.
- Import: from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
- Use methods like get_repository_languages(), get_repository_contributors(), etc.
- DO NOT analyze local files - fetch all data from GitHub
- DO NOT use placeholder paths like "." or local directory analysis
"""'''
    
    new_context = '''# Build additional context based on requirements
        additional_context = ""
        # Be specific about when GitHub API is needed
        if requires_github_api and ('github issue' in tool_intent.lower() or 'github api' in tool_intent.lower()):
            additional_context = """
This tool needs GitHub API for issue management.
- Import: from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
- Use for creating, updating, closing GitHub issues
"""
        elif 'architecture' in tool_intent.lower() and 'check' in tool_intent.lower():
            additional_context = """
This tool checks for file existence in repositories.
- Use subprocess with gh CLI: subprocess.run(f"gh api repos/{repo}/contents/ARCHITECTURE.md", ...)
- DO NOT import MCPGitHubIssueCreator unless creating issues
- Keep it simple - just check if files exist
"""'''
    
    content = content.replace(old_context, new_context)
    
    # Fix 4: Add smarter prompt instructions
    old_prompt_start = '''prompt = f"""Create a Python tool with these specifications:'''
    
    new_prompt_start = '''prompt = f"""Create a SIMPLE Python tool that does EXACTLY what is requested:'''
    
    content = content.replace(old_prompt_start, new_prompt_start)
    
    # Save the improved version
    generator_path.write_text(content)
    
    print("✅ Successfully applied smart improvements to tool generator")
    print("The system will now:")
    print("- Only require GitHub API for actual GitHub operations")
    print("- Generate simpler tools for file checking operations")  
    print("- Avoid unnecessary imports")
    print("- Focus on the specific task requested")
    
    return True


if __name__ == "__main__":
    apply_smart_improvements()