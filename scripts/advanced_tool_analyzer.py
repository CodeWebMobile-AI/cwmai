"""
Advanced Tool Analyzer

Performs deep analysis of custom tools to identify consolidation opportunities
and suggest new capabilities for the AI assistant.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import re


class AdvancedToolAnalyzer:
    """Analyzes tools and codebase to suggest improvements."""
    
    def __init__(self):
        self.tools_dir = Path(__file__).parent / "custom_tools"
        self.scripts_dir = Path(__file__).parent
        self.project_root = Path(__file__).parent.parent
        
        # Tool categories with detailed analysis
        self.tool_categories = {
            "counting": {
                "keywords": ["count", "how_many", "calculate"],
                "tools": [],
                "consolidation": "universal_counter.py"
            },
            "repository": {
                "keywords": ["repository", "repo", "get_repository", "get_repositories"],
                "tools": [],
                "consolidation": "repository_manager.py"
            },
            "architecture": {
                "keywords": ["architecture", "check_architecture"],
                "tools": [],
                "consolidation": "architecture_analyzer.py"
            },
            "analysis": {
                "keywords": ["analyze", "analysis", "check", "scan"],
                "tools": [],
                "consolidation": "code_analyzer.py"
            },
            "file_operations": {
                "keywords": ["file", "read", "write", "list", "grep", "search"],
                "tools": [],
                "consolidation": "file_manager.py"
            },
            "system_control": {
                "keywords": ["start", "stop", "restart", "control", "system", "status"],
                "tools": [],
                "consolidation": "system_controller.py"
            },
            "creation": {
                "keywords": ["create", "generate", "setup", "make"],
                "tools": [],
                "consolidation": "smart_creator.py"
            },
            "monitoring": {
                "keywords": ["monitor", "watch", "track", "performance"],
                "tools": [],
                "consolidation": "system_monitor.py"
            }
        }
        
        # Capabilities we should have but might be missing
        self.desired_capabilities = {
            "debugging": ["debug code", "trace execution", "profile performance"],
            "testing": ["run tests", "generate tests", "test coverage"],
            "deployment": ["deploy code", "manage environments", "CI/CD"],
            "database": ["query database", "manage schema", "data analysis"],
            "api": ["make API calls", "test endpoints", "manage APIs"],
            "documentation": ["generate docs", "update README", "API docs"],
            "refactoring": ["refactor code", "optimize", "clean code"],
            "security": ["scan vulnerabilities", "audit code", "check dependencies"],
            "collaboration": ["manage PRs", "code review", "team coordination"],
            "learning": ["analyze patterns", "suggest improvements", "learn from code"]
        }
        
    def analyze_all_tools(self) -> Dict[str, Any]:
        """Perform comprehensive tool analysis."""
        # Categorize existing tools
        for tool_path in self.tools_dir.glob("*.py"):
            if tool_path.name == "__init__.py":
                continue
                
            tool_name = tool_path.name
            tool_categorized = False
            
            for category, info in self.tool_categories.items():
                if any(keyword in tool_name.lower() for keyword in info["keywords"]):
                    info["tools"].append(tool_name)
                    tool_categorized = True
                    break
                    
            if not tool_categorized:
                # Try to categorize by reading the file
                try:
                    content = tool_path.read_text()
                    for category, info in self.tool_categories.items():
                        if any(keyword in content.lower() for keyword in info["keywords"]):
                            info["tools"].append(tool_name)
                            break
                except:
                    pass
                    
        return self.tool_categories
        
    def find_missing_capabilities(self) -> Dict[str, List[str]]:
        """Find capabilities we should have but don't."""
        missing = {}
        
        # Get all tool names and descriptions
        existing_tools = set()
        tool_descriptions = []
        
        for tool_path in self.tools_dir.glob("*.py"):
            existing_tools.add(tool_path.stem)
            try:
                content = tool_path.read_text()
                # Extract docstring
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Module) and ast.get_docstring(node):
                        tool_descriptions.append(ast.get_docstring(node).lower())
                        break
            except:
                pass
                
        # Check each desired capability
        for capability, features in self.desired_capabilities.items():
            missing_features = []
            for feature in features:
                # Check if we have a tool for this feature
                feature_words = feature.lower().split()
                found = False
                
                # Check tool names
                for tool in existing_tools:
                    if any(word in tool.lower() for word in feature_words):
                        found = True
                        break
                        
                # Check descriptions
                if not found:
                    for desc in tool_descriptions:
                        if any(word in desc for word in feature_words):
                            found = True
                            break
                            
                if not found:
                    missing_features.append(feature)
                    
            if missing_features:
                missing[capability] = missing_features
                
        return missing
        
    def analyze_codebase_patterns(self) -> Dict[str, Any]:
        """Analyze codebase to find patterns and suggest tools."""
        patterns = {
            "common_operations": defaultdict(int),
            "frequent_imports": defaultdict(int),
            "common_patterns": defaultdict(int),
            "ai_capabilities": set(),
            "workflow_patterns": []
        }
        
        # Analyze Python files in scripts directory
        for py_file in self.scripts_dir.glob("*.py"):
            try:
                content = py_file.read_text()
                
                # Find imports
                import_matches = re.findall(r'^import\s+(\w+)|^from\s+(\w+)', content, re.MULTILINE)
                for match in import_matches:
                    module = match[0] or match[1]
                    patterns["frequent_imports"][module] += 1
                    
                # Find common operations
                if "asyncio" in content:
                    patterns["common_operations"]["async_operations"] += 1
                if "json.dump" in content or "json.load" in content:
                    patterns["common_operations"]["json_handling"] += 1
                if "Path(" in content or "pathlib" in content:
                    patterns["common_operations"]["path_operations"] += 1
                if "subprocess" in content:
                    patterns["common_operations"]["subprocess_calls"] += 1
                if "requests" in content or "httpx" in content:
                    patterns["common_operations"]["http_requests"] += 1
                    
                # Find AI capabilities
                if "ai_brain" in content.lower() or "ai_client" in content.lower():
                    patterns["ai_capabilities"].add(py_file.stem)
                    
                # Find workflow patterns
                if "workflow" in content.lower() or "orchestrat" in content.lower():
                    patterns["workflow_patterns"].append(py_file.stem)
                    
            except:
                pass
                
        return patterns
        
    def generate_consolidation_plan(self) -> Dict[str, Any]:
        """Generate a detailed consolidation plan."""
        plan = {
            "consolidations": [],
            "new_tools": [],
            "enhancements": []
        }
        
        # Plan consolidations
        for category, info in self.tool_categories.items():
            if len(info["tools"]) > 1:
                plan["consolidations"].append({
                    "category": category,
                    "new_tool": info["consolidation"],
                    "combines": info["tools"],
                    "benefit": f"Reduces {len(info['tools'])} tools to 1 powerful tool"
                })
                
        # Plan new tools based on missing capabilities
        missing = self.find_missing_capabilities()
        for capability, features in missing.items():
            tool_name = f"{capability}_manager.py"
            plan["new_tools"].append({
                "name": tool_name,
                "capability": capability,
                "features": features,
                "purpose": f"Provides {capability} capabilities"
            })
            
        # Suggest enhancements based on patterns
        patterns = self.analyze_codebase_patterns()
        
        if patterns["common_operations"]["async_operations"] > 10:
            plan["enhancements"].append({
                "suggestion": "async_task_manager",
                "reason": "Heavy async usage detected",
                "features": ["parallel execution", "task coordination", "async debugging"]
            })
            
        if patterns["common_operations"]["http_requests"] > 5:
            plan["enhancements"].append({
                "suggestion": "api_integration_hub",
                "reason": "Multiple HTTP operations detected",
                "features": ["unified API client", "request caching", "error handling"]
            })
            
        return plan
        
    def generate_tool_code(self, tool_name: str, features: List[str], combines: List[str] = None) -> str:
        """Generate code for a new consolidated tool."""
        if combines:
            # Consolidation tool
            return f'''"""
{tool_name.replace('_', ' ').title().replace('.Py', '')}

Consolidated tool that combines functionality from:
{chr(10).join(f'- {tool}' for tool in combines)}

This tool provides unified access to all {tool_name.split('_')[0]} operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient


class {tool_name.replace('.py', '').title().replace('_', '')}:
    """Unified {tool_name.split('_')[0]} operations."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.ai_client = HTTPAIClient()
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute any {tool_name.split('_')[0]} operation.
        
        Operations:
        {chr(10).join(f'- {op.replace(".py", "").replace("_", " ")}' for op in combines)}
        """
        # Route to appropriate method
        operation_map = {{
            {chr(10).join(f'            "{op.replace(".py", "")}": self._{op.replace(".py", "").replace("-", "_")[:20]},' for op in combines)}
        }}
        
        handler = operation_map.get(operation)
        if not handler:
            return {{"error": f"Unknown operation: {{operation}}"}}
            
        return await handler(**kwargs)
        
    {chr(10).join(f"""
    async def _{op.replace(".py", "").replace("-", "_")[:20]}(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Handle {op.replace(".py", "").replace("_", " ")} operation.\"\"\"
        # Implementation from original {op}
        return {{"success": True, "operation": "{op.replace('.py', '')}", "result": "Implemented"}}""" for op in combines[:3])}


async def main(**kwargs):
    """Entry point for tool execution."""
    tool = {tool_name.replace('.py', '').title().replace('_', '')}()
    return await tool.execute(**kwargs)
'''
        else:
            # New capability tool
            return f'''"""
{tool_name.replace('_', ' ').title().replace('.Py', '')}

Provides advanced {tool_name.split('_')[0]} capabilities for the AI assistant.

Features:
{chr(10).join(f'- {feature}' for feature in features)}
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class {tool_name.replace('.py', '').title().replace('_', '')}:
    """Advanced {tool_name.split('_')[0]} capabilities."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        
    {chr(10).join(f"""
    async def {feature.lower().replace(' ', '_')}(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Implement {feature}.
        
        Args:
            **kwargs: Parameters for {feature}
            
        Returns:
            Operation result
        \"\"\"
        try:
            # AI-powered implementation
            prompt = f\"\"\"Help implement {feature} with parameters: {{kwargs}}
            
Provide a practical implementation approach.\"\"\"
            
            response = await self.ai_client.request(
                model="gemini",
                messages=[{{"role": "user", "content": prompt}}],
                temperature=0.7
            )
            
            if response.get('success'):
                return {{
                    "success": True,
                    "feature": "{feature}",
                    "result": response.get('content'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }}
            else:
                return {{"success": False, "error": "AI processing failed"}}
                
        except Exception as e:
            return {{"success": False, "error": str(e)}}""" for feature in features[:3])}
            
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute any {tool_name.split('_')[0]} action."""
        action_map = {{
            {chr(10).join(f'            "{feature.lower().replace(" ", "_")}": self.{feature.lower().replace(" ", "_")},' for feature in features[:3])}
        }}
        
        handler = action_map.get(action)
        if not handler:
            return {{"error": f"Unknown action: {{action}}"}}
            
        return await handler(**kwargs)


async def main(action: str = None, **kwargs):
    """Entry point for tool execution."""
    tool = {tool_name.replace('.py', '').title().replace('_', '')}()
    
    if action:
        return await tool.execute(action, **kwargs)
    else:
        # Default action
        return {{
            "available_actions": {list(feature.lower().replace(' ', '_') for feature in features)},
            "description": "Use action parameter to specify operation"
        }}
'''
        
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        categories = self.analyze_all_tools()
        missing = self.find_missing_capabilities()
        patterns = self.analyze_codebase_patterns()
        plan = self.generate_consolidation_plan()
        
        report = """# Advanced Tool Analysis Report

## Current Tool Distribution

"""
        # Category breakdown
        for category, info in categories.items():
            if info["tools"]:
                report += f"### {category.title()} ({len(info['tools'])} tools)\n"
                for tool in sorted(info["tools"]):
                    report += f"- {tool}\n"
                report += f"**Suggested consolidation**: `{info['consolidation']}`\n\n"
                
        report += f"\n## Missing Capabilities\n\n"
        for capability, features in missing.items():
            report += f"### {capability.title()}\n"
            for feature in features:
                report += f"- [ ] {feature}\n"
            report += "\n"
            
        report += f"\n## Codebase Patterns\n\n"
        report += f"- Async operations: {patterns['common_operations']['async_operations']} files\n"
        report += f"- JSON handling: {patterns['common_operations']['json_handling']} files\n"
        report += f"- HTTP requests: {patterns['common_operations']['http_requests']} files\n"
        report += f"- AI-enabled modules: {len(patterns['ai_capabilities'])}\n"
        report += f"- Workflow modules: {len(patterns['workflow_patterns'])}\n"
        
        report += f"\n## Consolidation Plan\n\n"
        
        # Consolidations
        report += "### Tool Consolidations\n"
        total_reduction = sum(len(c["combines"]) - 1 for c in plan["consolidations"])
        report += f"**Total reduction: {total_reduction} tools**\n\n"
        
        for consolidation in plan["consolidations"]:
            report += f"#### {consolidation['new_tool']}\n"
            report += f"- Combines {len(consolidation['combines'])} tools\n"
            report += f"- Tools: {', '.join(consolidation['combines'][:5])}"
            if len(consolidation['combines']) > 5:
                report += f" and {len(consolidation['combines']) - 5} more"
            report += f"\n- {consolidation['benefit']}\n\n"
            
        # New tools
        report += "### New Advanced Tools\n"
        for new_tool in plan["new_tools"]:
            report += f"\n#### {new_tool['name']}\n"
            report += f"- Purpose: {new_tool['purpose']}\n"
            report += f"- Features:\n"
            for feature in new_tool['features']:
                report += f"  - {feature}\n"
                
        # Enhancements
        if plan["enhancements"]:
            report += "\n### Suggested Enhancements\n"
            for enhancement in plan["enhancements"]:
                report += f"\n#### {enhancement['suggestion']}\n"
                report += f"- Reason: {enhancement['reason']}\n"
                report += f"- Features: {', '.join(enhancement['features'])}\n"
                
        return report


def main():
    """Run advanced tool analysis."""
    analyzer = AdvancedToolAnalyzer()
    report = analyzer.generate_report()
    
    # Save report
    report_path = Path(__file__).parent / "advanced_tool_analysis_report.md"
    report_path.write_text(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")
    
    # Return the plan for use
    plan = analyzer.generate_consolidation_plan()
    return plan


if __name__ == "__main__":
    main()