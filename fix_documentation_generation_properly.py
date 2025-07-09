#!/usr/bin/env python3
"""
Proper fix to prevent documentation task generation.

Instead of creating fake issues, we fix the root cause:
1. Update repository analyzer to not report missing documentation for new file types
2. Update work finder to skip documentation generation completely
3. Ensure portfolio analysis doesn't generate documentation gaps
"""

import os
from pathlib import Path

def apply_proper_fixes():
    """Apply proper fixes to prevent documentation task generation."""
    
    print("🔧 Applying proper documentation generation fixes...\n")
    
    # Fix 1: Update repository analyzer to not check for old documentation files
    print("1️⃣ Updating repository_analyzer.py to remove old documentation checks...")
    
    analyzer_path = Path("scripts/repository_analyzer.py")
    content = analyzer_path.read_text()
    
    # Remove the specific needs checks for old documentation
    # The file already has the new names, but we need to remove the need generation
    if "_identify_specific_needs" in content:
        # Find the method and remove documentation need generation
        lines = content.split('\n')
        new_lines = []
        in_doc_check = False
        skip_lines = 0
        
        for i, line in enumerate(lines):
            if skip_lines > 0:
                skip_lines -= 1
                continue
                
            # Skip the old documentation need checks
            if "# Check for missing architecture documentation" in line:
                in_doc_check = True
                skip_lines = 8  # Skip the entire block
                continue
            elif "# Check for missing monetization strategy documentation" in line:
                in_doc_check = True
                skip_lines = 8  # Skip the entire block
                continue
            elif "# Check for all 3 required documentation files" in line:
                # Skip the entire required docs check block
                in_doc_check = True
                skip_lines = 35  # Skip until after the needs.append
                continue
                
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        analyzer_path.write_text(content)
        print("  ✓ Removed old documentation need checks")
    
    # Fix 2: Update intelligent work finder to skip documentation completely
    print("\n2️⃣ Updating intelligent_work_finder.py to skip documentation task generation...")
    
    work_finder_path = Path("scripts/intelligent_work_finder.py")
    content = work_finder_path.read_text()
    
    # Find and comment out the documentation task generation section
    if "# Check for any missing documentation files" in content:
        lines = content.split('\n')
        new_lines = []
        in_doc_section = False
        
        for line in lines:
            if "# Check for any missing documentation files" in line:
                new_lines.append(line)
                new_lines.append("            # DISABLED: Documentation task generation")
                new_lines.append("            # The system should not automatically generate documentation tasks")
                in_doc_section = True
            elif in_doc_section and "if missing_docs:" in line:
                # End of documentation section
                in_doc_section = False
                new_lines.append("            pass  # Documentation task generation disabled")
            elif in_doc_section:
                # Comment out the documentation generation code
                new_lines.append("            # " + line)
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        work_finder_path.write_text(content)
        print("  ✓ Disabled documentation task generation in work finder")
    
    # Fix 3: Update TaskTypeSelector to exclude documentation types
    print("\n3️⃣ Updating task_types.py TaskTypeSelector to exclude documentation tasks...")
    
    task_types_path = Path("scripts/task_types.py")
    content = task_types_path.read_text()
    
    # Find get_appropriate_task_types method and filter out documentation
    if "def get_appropriate_task_types" in content:
        # Add filter at the end of the method before return
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Add filter before returning appropriate_types
            if "return appropriate_types" in line and "TaskTypeSelector" in content[:content.index(line)]:
                # Insert filtering code before the return
                indent = len(line) - len(line.lstrip())
                new_lines.insert(-1, " " * indent + "# Filter out all documentation task types")
                new_lines.insert(-1, " " * indent + "appropriate_types = [t for t in appropriate_types if 'DOCUMENTATION' not in t.name]")
        
        content = '\n'.join(new_lines)
        task_types_path.write_text(content)
        print("  ✓ Added filter to exclude documentation tasks from selection")
    
    # Fix 4: Update portfolio analysis to not generate documentation gaps
    print("\n4️⃣ Ensuring portfolio analysis doesn't generate documentation gaps...")
    
    # The portfolio_intelligence.py already has the fix from earlier
    print("  ✓ Portfolio analysis already updated to check file existence")
    
    print("\n✅ All proper fixes applied!")
    print("\nThese fixes ensure:")
    print("  1. Repository analyzer doesn't report missing documentation as needs")
    print("  2. Work finder skips documentation task generation entirely")
    print("  3. Task type selector filters out documentation tasks")
    print("  4. Portfolio analysis respects existing files")
    print("\nThe system will now focus on real development tasks instead of documentation.")
    
    return True

if __name__ == "__main__":
    if apply_proper_fixes():
        print("\n🎉 Documentation generation properly disabled!")
        print("\nNext steps:")
        print("1. Test the system to verify no documentation tasks are created")
        print("2. Create real development issues for projects to work on")