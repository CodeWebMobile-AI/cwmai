"""
Simple Dict Fixer

A focused tool to fix dict attribute access errors in Python code.
"""

import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def fix_dict_attribute_access(file_path: Path, attr_name: str) -> bool:
    """
    Fix dict attribute access errors in a file.
    
    Replaces patterns like `obj.attr` with `obj.get('attr')` or 
    `obj['attr']` when obj is likely a dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        original_content = content
        
        # Pattern to find attribute access
        # This will match things like "issue.number" or "item.title"
        pattern = rf'(\w+)\.{attr_name}\b'
        
        # Find all matches to determine context
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            logger.warning(f"No instances of .{attr_name} found in {file_path}")
            return False
            
        # Replace each match with dict-safe access
        for match in reversed(matches):  # Reverse to maintain positions
            var_name = match.group(1)
            start, end = match.span()
            
            # Check if this is in a format string
            line_start = content.rfind('\n', 0, start) + 1
            line_end = content.find('\n', end)
            if line_end == -1:
                line_end = len(content)
            line = content[line_start:line_end]
            
            # Determine replacement based on context
            if 'print' in line or 'f"' in line or "f'" in line:
                # In format string, use {var.get('attr')}
                replacement = f"{var_name}.get('{attr_name}')"
            else:
                # Regular context, use bracket notation
                replacement = f"{var_name}['{attr_name}']"
                
            # Apply replacement
            content = content[:start] + replacement + content[end:]
            
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Fixed {len(matches)} instances of .{attr_name} in {file_path}")
            return True
        else:
            logger.warning("No changes made")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing dict access: {e}")
        return False