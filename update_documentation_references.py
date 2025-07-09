#!/usr/bin/env python3
"""Update all references to old documentation task types."""

import os
import re
from pathlib import Path

def update_references():
    """Update all references to old documentation types."""
    
    print("🔧 Updating documentation type references...\n")
    
    # Files to update (excluding task_types.py which we already fixed)
    files_to_update = [
        "scripts/intelligent_work_finder.py",
        "scripts/continuous_orchestrator_no_redis.py", 
        "scripts/intelligent_task_generator.py",
        "scripts/local_issue_creator.py",
        "scripts/github_issue_creator.py",
        "scripts/simple_work_generator.py"
    ]
    
    # Mapping of old to new types
    replacements = {
        # Replace specific task type references
        "SmartTaskType.DOCUMENTATION_README": "SmartTaskType.DOCUMENTATION_USER_GUIDE",
        "SmartTaskType.DOCUMENTATION_ARCHITECTURE": "SmartTaskType.DOCUMENTATION_DEVELOPER_GUIDE",
        "SmartTaskType.DOCUMENTATION_MONETIZATION": "SmartTaskType.DOCUMENTATION_API",
        
        # Replace string references
        '"documentation_readme"': '"documentation_user_guide"',
        '"documentation_architecture"': '"documentation_developer_guide"',
        '"documentation_monetization"': '"documentation_api"',
        "'documentation_readme'": "'documentation_user_guide'",
        "'documentation_architecture'": "'documentation_developer_guide'",
        "'documentation_monetization'": "'documentation_api'",
        
        # Replace in task type checks
        "DOCUMENTATION_README": "DOCUMENTATION_USER_GUIDE",
        "DOCUMENTATION_ARCHITECTURE": "DOCUMENTATION_DEVELOPER_GUIDE", 
        "DOCUMENTATION_MONETIZATION": "DOCUMENTATION_API",
        
        # Replace in descriptions/titles
        "README.md": "USER_GUIDE.md",
        "ARCHITECTURE.md": "DEVELOPER_GUIDE.md",
        "MONETIZATION_STRATEGY.md": "API_DOCUMENTATION.md"
    }
    
    total_replacements = 0
    
    for file_path in files_to_update:
        if not Path(file_path).exists():
            print(f"⚠️  Skipping {file_path} - file not found")
            continue
            
        print(f"📝 Updating {file_path}...")
        
        content = Path(file_path).read_text()
        original_content = content
        file_replacements = 0
        
        # Apply replacements
        for old, new in replacements.items():
            count = content.count(old)
            if count > 0:
                content = content.replace(old, new)
                file_replacements += count
                print(f"   • Replaced '{old}' -> '{new}' ({count} occurrences)")
        
        # Save if changes were made
        if content != original_content:
            Path(file_path).write_text(content)
            print(f"   ✓ Updated with {file_replacements} replacements")
            total_replacements += file_replacements
        else:
            print(f"   ✓ No changes needed")
    
    print(f"\n✅ Total replacements made: {total_replacements}")
    
    # Also update any references to checking for these specific files
    print("\n📝 Updating file existence checks...")
    
    # Files that might check for old documentation files
    checker_files = [
        "scripts/repository_analyzer.py",
        "scripts/intelligent_work_finder.py",
        "scripts/portfolio_intelligence.py"
    ]
    
    doc_file_replacements = {
        # Keep README.md as is, but update the others
        "ARCHITECTURE.md": "DEVELOPER_GUIDE.md",
        "MONETIZATION_STRATEGY.md": "API_DOCUMENTATION.md",
        
        # Update task type associations
        "architecture_documentation": "developer_guide_documentation",
        "monetization_documentation": "api_documentation",
        "readme_documentation": "user_guide_documentation"
    }
    
    for file_path in checker_files:
        if not Path(file_path).exists():
            continue
            
        print(f"\n📝 Updating documentation checks in {file_path}...")
        
        content = Path(file_path).read_text()
        original_content = content
        file_replacements = 0
        
        for old, new in doc_file_replacements.items():
            count = content.count(old)
            if count > 0:
                content = content.replace(old, new)
                file_replacements += count
                print(f"   • Replaced '{old}' -> '{new}' ({count} occurrences)")
        
        if content != original_content:
            Path(file_path).write_text(content)
            print(f"   ✓ Updated with {file_replacements} replacements")
    
    print("\n✅ Documentation type references updated successfully!")
    
if __name__ == "__main__":
    update_references()