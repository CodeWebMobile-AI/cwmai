#!/usr/bin/env python3
"""Debug test case 5."""

import re

response = """Let me show you an example first:
`EXECUTE_TOOL::example_tool()::END_TOOL`

Now I'll actually run this:
EXECUTE_TOOL::get_system_status()::END_TOOL

Here's how to use another tool:
```
EXECUTE_TOOL::create_issue(repo='test', title='Bug')::END_TOOL
```

And let me check something else:
EXECUTE_TOOL::count_repositories()::END_TOOL"""

# First, remove any content within code blocks to prevent example execution
print("Original response:")
print(response)
print("\n" + "="*60 + "\n")

response_without_codeblocks = re.sub(r'```[\s\S]*?```', '', response)
print("After removing code blocks:")
print(response_without_codeblocks)
print("\n" + "="*60 + "\n")

response_without_inline_code = re.sub(r'`[^`]+`', '', response_without_codeblocks)
print("After removing inline code:")
print(response_without_inline_code)
print("\n" + "="*60 + "\n")

# Check for "Here's how" text
print("Checking for example indicators...")
tool_pattern = r'EXECUTE_TOOL::([\w\.]+)\s*\((.*?)\)::END_TOOL'
for match in re.finditer(tool_pattern, response_without_inline_code, re.DOTALL):
    tool_name = match.group(1)
    start_pos = match.start()
    preceding_text = response_without_inline_code[max(0, start_pos-100):start_pos].lower()
    print(f"\nTool: {tool_name}")
    print(f"Preceding text: '{preceding_text}'")