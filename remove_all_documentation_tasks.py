#!/usr/bin/env python3
"""
Remove ALL documentation task generation from the system.
Clean up all the complications we added.
"""

import os
from pathlib import Path

def remove_documentation_tasks():
    """Remove all documentation task generation."""
    
    print("🧹 Removing ALL documentation task generation...\n")
    
    # 1. Clean up intelligent_work_finder.py
    print("1️⃣ Cleaning up intelligent_work_finder.py...")
    work_finder_path = Path("scripts/intelligent_work_finder.py")
    content = work_finder_path.read_text()
    
    # Remove the entire documentation section
    lines = content.split('\n')
    new_lines = []
    skip_until = None
    
    for i, line in enumerate(lines):
        # Skip the entire documentation generation block
        if "# Check for any missing documentation files" in line:
            skip_until = "opportunities.append(opportunity)"
            continue
        
        if skip_until and skip_until in line:
            skip_until = None
            continue
            
        if skip_until:
            continue
            
        # Remove pass statement we added
        if line.strip() == "pass  # Documentation task generation disabled":
            continue
            
        new_lines.append(line)
    
    work_finder_path.write_text('\n'.join(new_lines))
    print("  ✓ Removed documentation task generation block")
    
    # 2. Clean up repository_analyzer.py
    print("\n2️⃣ Cleaning up repository_analyzer.py...")
    analyzer_path = Path("scripts/repository_analyzer.py")
    content = analyzer_path.read_text()
    
    # Remove documentation checks from _identify_specific_needs
    lines = content.split('\n')
    new_lines = []
    in_needs_method = False
    skip_block = False
    
    for i, line in enumerate(lines):
        # Track if we're in the needs method
        if "def _identify_specific_needs" in line:
            in_needs_method = True
            new_lines.append(line)
            continue
            
        # End of method
        if in_needs_method and line.strip() and not line.startswith(' '):
            in_needs_method = False
            
        # Skip documentation-related blocks in needs method
        if in_needs_method:
            if any(x in line for x in [
                "Check for missing architecture documentation",
                "Check for missing monetization strategy", 
                "Check for all 3 required documentation",
                "doc_file, doc_info in required_docs.items()",
                "'type': 'developer_guide_documentation'",
                "'type': 'api_documentation'",
                "'type': 'user_guide_documentation'"
            ]):
                skip_block = True
                
            if skip_block and "needs.append({" in line:
                # Skip until the closing }
                bracket_count = 1
                continue
                
            if skip_block and "})" in line:
                skip_block = False
                continue
                
            if skip_block:
                continue
        
        new_lines.append(line)
    
    analyzer_path.write_text('\n'.join(new_lines))
    print("  ✓ Removed documentation need identification")
    
    # 3. Remove documentation types from task_types.py filter
    print("\n3️⃣ Cleaning up task_types.py...")
    task_types_path = Path("scripts/task_types.py")
    content = task_types_path.read_text()
    
    # Remove the filter we added
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Skip the filter we added
        if "# Filter out all documentation task types" in line:
            continue
        if "appropriate_types = [t for t in appropriate_types if 'DOCUMENTATION' not in t.name]" in line:
            continue
            
        new_lines.append(line)
    
    task_types_path.write_text('\n'.join(new_lines))
    print("  ✓ Removed documentation task filter")
    
    # 4. Clean up portfolio_intelligence.py
    print("\n4️⃣ Cleaning up portfolio_intelligence.py...")
    portfolio_path = Path("scripts/portfolio_intelligence.py")
    content = portfolio_path.read_text()
    
    # Remove the documentation check we added
    lines = content.split('\n')
    new_lines = []
    skip_block = False
    
    for line in lines:
        if "# Check for existing documentation files in local mode" in line:
            skip_block = True
            
        if skip_block and "gaps = filtered_gaps" in line:
            new_lines.append(line)
            skip_block = False
            continue
            
        if skip_block:
            continue
            
        new_lines.append(line)
    
    portfolio_path.write_text('\n'.join(new_lines))
    print("  ✓ Removed documentation existence check from portfolio analysis")
    
    print("\n✅ All documentation task generation removed!")
    print("\nThe system will now:")
    print("  - NOT generate any documentation tasks")
    print("  - Focus on other types of development work")
    print("  - Not check for documentation files")
    
    return True

if __name__ == "__main__":
    if remove_documentation_tasks():
        print("\n🎉 Documentation task generation completely removed!")
        print("\nThe system is now clean and will not create documentation tasks.")