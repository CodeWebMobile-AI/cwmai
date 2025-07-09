#!/usr/bin/env python3
"""Fix unescaped JSON examples in prompt strings."""

import re

# Read the file
with open('scripts/gemini_deep_research.py', 'r') as f:
    content = f.read()

# Pattern to find JSON examples in f-strings
# This will find patterns like:
# [
#     {
#         "field": value
#     }
# ]
# within f""" strings

def fix_json_in_fstring(match):
    """Replace single braces with double braces in JSON examples."""
    json_content = match.group(0)
    # Replace single { with {{ and single } with }}
    # but only for the JSON structure, not for f-string variables
    lines = json_content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Skip lines that contain chr() or other Python expressions
        if 'chr(' in line or '.join(' in line or '.get(' in line:
            fixed_lines.append(line)
        else:
            # Replace { with {{ and } with }} for JSON
            fixed_line = line.replace('{', '{{').replace('}', '}}')
            fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)

# Find all f-string blocks that contain JSON-like structures
pattern = r'(f"""[\s\S]*?""")'

def process_fstring(match):
    fstring = match.group(1)
    
    # Only process if it looks like it contains JSON examples
    if '"index":' in fstring or 'Return.*analysis.*with:' in fstring or 'Return.*detailed.*:' in fstring:
        # Find JSON-like structures and escape them
        # Look for patterns like [{ ... }] or { ... }
        json_pattern = r'(\s*[\[\{][^}]*\}[\],]?\s*)'
        
        lines = fstring.split('\n')
        fixed_lines = []
        in_json = False
        json_indent = 0
        
        for line in lines:
            # Check if we're starting a JSON example
            if not in_json and re.search(r'^\s*[\[\{]\s*$', line):
                in_json = True
                json_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line.replace('{', '{{').replace('[', '['))
            # Check if we're in a JSON block
            elif in_json:
                # Check if we're ending the JSON
                if re.search(r'^\s*[\]\}]\s*$', line):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= json_indent:
                        in_json = False
                        fixed_lines.append(line.replace('}', '}}').replace(']', ']'))
                    else:
                        fixed_lines.append(line.replace('{', '{{').replace('}', '}}'))
                else:
                    # We're inside JSON, escape braces unless it's a Python expression
                    if 'chr(' not in line and '.join(' not in line and '.get(' not in line:
                        fixed_lines.append(line.replace('{', '{{').replace('}', '}}'))
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    return fstring

# Process all f-strings
new_content = re.sub(pattern, process_fstring, content)

# Write the fixed content
with open('scripts/gemini_deep_research.py', 'w') as f:
    f.write(new_content)

print("Fixed JSON escaping in gemini_deep_research.py")