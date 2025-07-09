#!/usr/bin/env python3
"""Fix path issues in zip folder custom tools to make them self-contained."""

import os
import re
from pathlib import Path


def fix_custom_tool_paths():
    """Fix all path issues in custom tools."""
    zip_tools_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/scripts/custom_tools")
    
    if not zip_tools_dir.exists():
        print(f"Error: {zip_tools_dir} does not exist")
        return
    
    issues_fixed = 0
    
    # Walk through all Python files in custom_tools
    for tool_file in zip_tools_dir.rglob("*.py"):
        print(f"\nChecking: {tool_file.relative_to(zip_tools_dir)}")
        
        with open(tool_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Replace parent.parent.parent references
        # These escape the zip folder and point to parent
        if "parent.parent.parent" in content:
            print(f"  ⚠️  Found parent.parent.parent reference")
            # Replace with parent.parent to stay within zip folder
            content = content.replace(
                "Path(__file__).parent.parent.parent",
                "Path(__file__).parent.parent"
            )
            content = content.replace(
                "path.parent.parent.parent",
                "path.parent.parent"
            )
            print(f"  ✓ Fixed parent directory references")
        
        # Fix 2: Fix sys.path.append with parent.parent.parent
        if "sys.path.append(str(Path(__file__).parent.parent.parent))" in content:
            print(f"  ⚠️  Found sys.path.append escaping zip folder")
            content = content.replace(
                "sys.path.append(str(Path(__file__).parent.parent.parent))",
                "sys.path.append(str(Path(__file__).parent.parent))"
            )
            print(f"  ✓ Fixed sys.path.append")
        
        # Fix 3: Fix hardcoded absolute paths to parent
        if "/Users/fortressarigo/PhpstormProjects/cwmai/" in content and "/zip/" not in content:
            print(f"  ⚠️  Found hardcoded path to parent directory")
            # This is pointing to parent, redirect to zip
            content = content.replace(
                "/Users/fortressarigo/PhpstormProjects/cwmai/",
                "/Users/fortressarigo/PhpstormProjects/cwmai/zip/"
            )
            print(f"  ✓ Fixed hardcoded parent paths")
        
        # Fix 4: Fix conversation_history.json path
        if "conversation_history.json" in content and "parent.parent.parent" in content:
            print(f"  ⚠️  Found conversation history path escaping zip")
            # Look for the pattern and fix it
            pattern = r'Path\(__file__\)\.parent\.parent\.parent\s*/\s*"conversation_history\.json"'
            replacement = 'Path(__file__).parent.parent / "conversation_history.json"'
            content = re.sub(pattern, replacement, content)
            print(f"  ✓ Fixed conversation history path")
        
        # Fix 5: Add proper path setup for imports if not present
        if "from scripts." in content and "sys.path" not in content:
            print(f"  ⚠️  Found scripts imports without path setup")
            # Add proper path setup after imports
            import_section_end = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('import') and not line.startswith('from'):
                    import_section_end = i
                    break
            
            path_setup = """
# Ensure we use zip folder's modules
import sys
from pathlib import Path
zip_root = Path(__file__).parent.parent
if str(zip_root) not in sys.path:
    sys.path.insert(0, str(zip_root))
"""
            lines.insert(import_section_end, path_setup)
            content = '\n'.join(lines)
            print(f"  ✓ Added proper path setup for imports")
        
        # Write back if changes were made
        if content != original_content:
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(content)
            issues_fixed += 1
            print(f"  ✅ Fixed and saved {tool_file.name}")
        else:
            print(f"  ✔️  No issues found")
    
    print(f"\n{'='*60}")
    print(f"Total files fixed: {issues_fixed}")
    print(f"\nRecommendations:")
    print(f"1. Test each tool to ensure it works correctly")
    print(f"2. Consider using relative imports instead of absolute")
    print(f"3. Use environment variables for base paths instead of hardcoding")


if __name__ == "__main__":
    fix_custom_tool_paths()