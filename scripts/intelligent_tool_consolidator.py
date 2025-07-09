"""
Intelligent Tool Consolidator

Analyzes custom tools to identify duplicates, test tools, and similar functionality
that can be merged into more powerful, versatile tools.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import hashlib


class IntelligentToolConsolidator:
    """Consolidates and optimizes custom tools."""
    
    def __init__(self):
        self.tools_dir = Path(__file__).parent / "custom_tools"
        self.tools_metadata = {}
        self.duplicate_groups = defaultdict(list)
        self.similar_tools = defaultdict(list)
        self.test_tools = []
        self.deprecated_tools = []
        
    def analyze_all_tools(self) -> Dict[str, Any]:
        """Analyze all custom tools and categorize them."""
        print(f"Analyzing {len(list(self.tools_dir.glob('*.py')))} tools...")
        
        for tool_path in self.tools_dir.glob("*.py"):
            if tool_path.name == "__init__.py":
                continue
                
            metadata = self._analyze_tool(tool_path)
            if metadata:
                self.tools_metadata[tool_path.name] = metadata
                
                # Identify test tools
                if self._is_test_tool(tool_path.name, metadata):
                    self.test_tools.append(tool_path.name)
                    
                # Identify deprecated tools
                if self._is_deprecated(metadata):
                    self.deprecated_tools.append(tool_path.name)
                    
        # Find duplicates and similar tools
        self._find_duplicates()
        self._find_similar_tools()
        
        return {
            "total_tools": len(self.tools_metadata),
            "test_tools": len(self.test_tools),
            "deprecated_tools": len(self.deprecated_tools),
            "duplicate_groups": len(self.duplicate_groups),
            "similar_groups": len(self.similar_tools),
            "tools_to_remove": len(self.test_tools) + len(self.deprecated_tools) + sum(len(group) - 1 for group in self.duplicate_groups.values())
        }
        
    def _analyze_tool(self, tool_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single tool file."""
        try:
            content = tool_path.read_text()
            tree = ast.parse(content)
            
            metadata = {
                "path": str(tool_path),
                "size": len(content),
                "functions": [],
                "imports": [],
                "description": "",
                "category": self._categorize_tool(tool_path.name, content),
                "content_hash": hashlib.md5(content.encode()).hexdigest()
            }
            
            # Extract docstring
            for node in ast.walk(tree):
                if isinstance(node, ast.Module) and ast.get_docstring(node):
                    metadata["description"] = ast.get_docstring(node)
                    break
                    
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metadata["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or ""
                    })
                    
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        metadata["imports"].append(f"{module}.{alias.name}")
                        
            return metadata
            
        except Exception as e:
            print(f"Error analyzing {tool_path.name}: {e}")
            return None
            
    def _is_test_tool(self, filename: str, metadata: Dict) -> bool:
        """Check if a tool is a test tool."""
        test_indicators = [
            "test_", "demo_", "example_", "sample_", "tool_task_",
            "_test", "_demo", "_example", "_temp", "_old"
        ]
        
        # Check filename
        for indicator in test_indicators:
            if indicator in filename.lower():
                return True
                
        # Check description
        desc_lower = metadata.get("description", "").lower()
        if any(word in desc_lower for word in ["test", "demo", "example", "temporary"]):
            return True
            
        # Check if it's a simple task tool (tool_task_1, tool_task_2, etc.)
        if re.match(r"tool_task_\d+\.py", filename):
            return True
            
        return False
        
    def _is_deprecated(self, metadata: Dict) -> bool:
        """Check if a tool is deprecated."""
        desc_lower = metadata.get("description", "").lower()
        return any(word in desc_lower for word in ["deprecated", "obsolete", "do not use", "replaced by"])
        
    def _categorize_tool(self, filename: str, content: str) -> str:
        """Categorize a tool based on its name and content."""
        categories = {
            "analysis": ["analyze", "check", "inspect", "examine", "review"],
            "creation": ["create", "generate", "make", "build", "setup"],
            "system": ["system", "control", "start", "stop", "restart", "status"],
            "repository": ["repository", "repo", "project", "github"],
            "counting": ["count", "calculate", "statistics", "how_many"],
            "search": ["find", "search", "grep", "list", "get"],
            "file": ["file", "read", "write", "edit"],
        }
        
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        for category, keywords in categories.items():
            if any(keyword in filename_lower or keyword in content_lower for keyword in keywords):
                return category
                
        return "other"
        
    def _find_duplicates(self):
        """Find duplicate tools based on content hash."""
        hash_to_tools = defaultdict(list)
        
        for tool_name, metadata in self.tools_metadata.items():
            hash_to_tools[metadata["content_hash"]].append(tool_name)
            
        for hash_val, tools in hash_to_tools.items():
            if len(tools) > 1:
                self.duplicate_groups[hash_val] = tools
                
    def _find_similar_tools(self):
        """Find tools with similar functionality."""
        # Group by category first
        category_tools = defaultdict(list)
        for tool_name, metadata in self.tools_metadata.items():
            category_tools[metadata["category"]].append(tool_name)
            
        # Within each category, find similar tools
        for category, tools in category_tools.items():
            if len(tools) > 1:
                # Simple similarity based on function names
                for i, tool1 in enumerate(tools):
                    similar_group = [tool1]
                    meta1 = self.tools_metadata[tool1]
                    
                    for tool2 in tools[i+1:]:
                        meta2 = self.tools_metadata[tool2]
                        
                        # Check if they have similar functions
                        funcs1 = {f["name"] for f in meta1["functions"]}
                        funcs2 = {f["name"] for f in meta2["functions"]}
                        
                        if funcs1 and funcs2:
                            overlap = len(funcs1 & funcs2) / min(len(funcs1), len(funcs2))
                            if overlap > 0.5:  # More than 50% overlap
                                similar_group.append(tool2)
                                
                    if len(similar_group) > 1:
                        # Use frozenset to avoid duplicate groups
                        group_key = frozenset(similar_group)
                        if group_key not in self.similar_tools:
                            self.similar_tools[group_key] = list(similar_group)
                            
    def get_consolidation_plan(self) -> Dict[str, Any]:
        """Get a plan for consolidating tools."""
        plan = {
            "tools_to_remove": [],
            "tools_to_merge": {},
            "core_tools_to_keep": [],
            "new_consolidated_tools": []
        }
        
        # Mark test and deprecated tools for removal
        plan["tools_to_remove"].extend(self.test_tools)
        plan["tools_to_remove"].extend(self.deprecated_tools)
        
        # Handle duplicates - keep one from each group
        for hash_val, tools in self.duplicate_groups.items():
            # Keep the shortest filename (usually the most generic)
            keep = min(tools, key=len)
            plan["core_tools_to_keep"].append(keep)
            plan["tools_to_remove"].extend([t for t in tools if t != keep])
            
        # Handle similar tools - suggest merging
        for group in self.similar_tools.values():
            if not any(tool in plan["tools_to_remove"] for tool in group):
                # Suggest a merged tool name
                common_words = self._find_common_words(group)
                if common_words:
                    merged_name = f"{'_'.join(common_words)}_unified.py"
                    plan["tools_to_merge"][merged_name] = group
                    plan["new_consolidated_tools"].append({
                        "name": merged_name,
                        "combines": group,
                        "category": self.tools_metadata[group[0]]["category"]
                    })
                    
        # Identify truly unique core tools
        all_involved = set(plan["tools_to_remove"]) | set(plan["core_tools_to_keep"])
        for group_tools in plan["tools_to_merge"].values():
            all_involved.update(group_tools)
            
        for tool_name in self.tools_metadata:
            if tool_name not in all_involved and tool_name not in self.test_tools:
                plan["core_tools_to_keep"].append(tool_name)
                
        # Remove duplicates from lists
        plan["tools_to_remove"] = list(set(plan["tools_to_remove"]))
        plan["core_tools_to_keep"] = list(set(plan["core_tools_to_keep"]))
        
        return plan
        
    def _find_common_words(self, tool_names: List[str]) -> List[str]:
        """Find common words in tool names."""
        if not tool_names:
            return []
            
        # Extract words from all tool names
        all_words = []
        for name in tool_names:
            words = name.replace(".py", "").split("_")
            all_words.append(words)
            
        # Find common words
        common = set(all_words[0])
        for words in all_words[1:]:
            common &= set(words)
            
        return list(common)
        
    def generate_report(self) -> str:
        """Generate a detailed report of the analysis."""
        analysis = self.analyze_all_tools()
        plan = self.get_consolidation_plan()
        
        report = f"""
# Tool Consolidation Report

## Summary
- Total tools analyzed: {analysis['total_tools']}
- Test/demo tools found: {analysis['test_tools']}
- Deprecated tools found: {analysis['deprecated_tools']}
- Duplicate groups found: {analysis['duplicate_groups']}
- Similar tool groups found: {analysis['similar_groups']}
- **Tools to remove: {analysis['tools_to_remove']}**
- **Core tools to keep: {len(plan['core_tools_to_keep'])}**

## Tools to Remove
"""
        for tool in sorted(plan["tools_to_remove"]):
            report += f"- {tool}\n"
            
        report += "\n## Core Tools to Keep\n"
        for tool in sorted(plan["core_tools_to_keep"]):
            report += f"- {tool}\n"
            
        if plan["tools_to_merge"]:
            report += "\n## Suggested Tool Mergers\n"
            for new_name, tools in plan["tools_to_merge"].items():
                report += f"\n### {new_name}\nCombines:\n"
                for tool in tools:
                    report += f"  - {tool}\n"
                    
        return report


def main():
    """Run the tool consolidation analysis."""
    consolidator = IntelligentToolConsolidator()
    report = consolidator.generate_report()
    
    # Save report
    report_path = Path(__file__).parent / "tool_consolidation_report.md"
    report_path.write_text(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")
    
    # Return the plan for use by other scripts
    return consolidator.get_consolidation_plan()


if __name__ == "__main__":
    main()