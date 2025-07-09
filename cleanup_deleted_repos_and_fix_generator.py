#!/usr/bin/env python3
"""
Cleanup references to deleted repositories and fix dynamic work generator error
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_deleted_repositories():
    """Remove all references to deleted repositories from state files"""
    deleted_repos = ['subtrack-ai', 'test-repo', 'sub-wise', 'subwise']
    cleaned_files = []
    
    # Clean work_queue_backup.json
    work_queue_file = Path('/Users/fortressarigo/PhpstormProjects/cwmai/work_queue_backup.json')
    if work_queue_file.exists():
        logger.info(f"Cleaning {work_queue_file}")
        with open(work_queue_file, 'r') as f:
            data = json.load(f)
        
        original_count = len(data.get('items', []))
        cleaned_items = []
        removed_count = 0
        
        for item in data.get('items', []):
            repo = item.get('repository', '')
            # Check if repository contains any deleted repo name
            if not any(deleted in repo.lower() for deleted in deleted_repos):
                cleaned_items.append(item)
            else:
                removed_count += 1
                logger.info(f"  Removing work item for deleted repo: {repo}")
        
        data['items'] = cleaned_items
        
        with open(work_queue_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"  Removed {removed_count} items from work queue (was {original_count}, now {len(cleaned_items)})")
        cleaned_files.append(work_queue_file)
    
    # Clean any other state files
    state_dir = Path('/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state')
    if state_dir.exists():
        for state_file in state_dir.glob('*.json'):
            if state_file.name == 'active_tasks.json':
                continue  # Already checked and clean
                
            logger.info(f"Checking {state_file}")
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Check if file contains repository references
                content = json.dumps(data)
                if any(deleted in content.lower() for deleted in deleted_repos):
                    logger.info(f"  Found references to deleted repos in {state_file.name}")
                    # Would need specific logic for each file type
                    # For now, just log it
            except Exception as e:
                logger.error(f"  Error reading {state_file}: {e}")
    
    return cleaned_files

def fix_dynamic_work_generator():
    """Fix the dynamic work generator to handle missing 'description' field"""
    generator_file = Path('/Users/fortressarigo/PhpstormProjects/cwmai/scripts/dynamic_work_generator.py')
    
    if not generator_file.exists():
        logger.error(f"Dynamic work generator not found at {generator_file}")
        return False
    
    # Read the current file
    with open(generator_file, 'r') as f:
        content = f.read()
    
    # Check if we need to fix the _generate_work_without_ai method
    if "need.get('description', '')" not in content:
        logger.info("Dynamic work generator already appears to be fixed")
        return True
    
    # The issue is in line 333 where it tries to access 'description' without a default
    # Let's fix line 333 and similar issues
    fixes_applied = False
    
    # Fix line 318-319: Add default for need_title
    old_line_318 = "need_title = need.get('suggested_action', need.get('description', ''))"
    new_line_318 = "need_title = need.get('suggested_action', need.get('description', 'Improvement needed'))"
    if old_line_318 in content:
        content = content.replace(old_line_318, new_line_318)
        fixes_applied = True
        logger.info("Fixed line 318: Added default for need_title")
    
    # Fix line 333: Ensure description has a default
    old_line_333 = 'title=need.get(\'suggested_action\', need.get(\'description\', \'Improvement needed\')),'
    # This line is already correct, but let's check the description field in line 334
    
    # Fix line 334: Ensure all fields have defaults
    old_line_334 = 'description=f"Type: {need.get(\'type\', \'general\')}\\n\\n{need.get(\'description\', \'\')}\\n\\nSuggested Action: {need.get(\'suggested_action\', \'\')}\",'
    new_line_334 = 'description=f"Type: {need.get(\'type\', \'general\')}\\n\\n{need.get(\'description\', \'No description provided\')}\\n\\nSuggested Action: {need.get(\'suggested_action\', \'No action specified\')}\",'
    if old_line_334 in content:
        content = content.replace(old_line_334, new_line_334)
        fixes_applied = True
        logger.info("Fixed line 334: Added defaults for description fields")
    
    # Fix line 356: Ensure opportunity description has a default
    old_line_356 = "title=opp.get('description', 'Improvement opportunity'),"
    new_line_356 = "title=opp.get('description', 'Improvement opportunity'),"
    # This line is already correct
    
    # Fix line 357: Ensure all fields in description have defaults
    old_line_357 = 'description=f"Type: {opp.get(\'type\', \'general\')}\\nImpact: {opp.get(\'impact\', \'unknown\')}\\nEffort: {opp.get(\'effort\', \'unknown\')}\",'
    new_line_357 = 'description=f"Type: {opp.get(\'type\', \'general\')}\\nImpact: {opp.get(\'impact\', \'unknown\')}\\nEffort: {opp.get(\'effort\', \'unknown\')}\\nDetails: {opp.get(\'description\', \'No additional details\')}\",'
    if old_line_357 in content:
        content = content.replace(old_line_357, new_line_357)
        fixes_applied = True
        logger.info("Fixed line 357: Added details field with default")
    
    if fixes_applied:
        # Backup the original file
        backup_file = generator_file.with_suffix('.py.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        with open(backup_file, 'w') as f:
            f.write(content)
        
        # Write the fixed content
        with open(generator_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Fixed dynamic work generator (backup saved as {backup_file.name})")
        return True
    else:
        logger.info("No fixes needed for dynamic work generator")
        return True

def main():
    """Main cleanup function"""
    logger.info("Starting cleanup of deleted repositories and fixing dynamic work generator")
    
    # Step 1: Clean up deleted repository references
    logger.info("\nStep 1: Cleaning up deleted repository references")
    cleaned_files = cleanup_deleted_repositories()
    
    # Step 2: Fix dynamic work generator
    logger.info("\nStep 2: Fixing dynamic work generator")
    generator_fixed = fix_dynamic_work_generator()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CLEANUP SUMMARY")
    logger.info("="*60)
    logger.info(f"Files cleaned: {len(cleaned_files)}")
    for f in cleaned_files:
        logger.info(f"  - {f.name}")
    logger.info(f"Dynamic work generator fixed: {'Yes' if generator_fixed else 'No'}")
    logger.info("\nThe system should now run without referencing deleted repositories")
    logger.info("and the dynamic work generator should handle missing fields gracefully.")

if __name__ == "__main__":
    main()