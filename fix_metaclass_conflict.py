#!/usr/bin/env python3
"""
Fix the metaclass conflict in task_type_plugin_base.py
"""

from pathlib import Path

# Fix the task_type_plugin_base.py file
plugin_file = Path("scripts/task_type_plugin_base.py")
if plugin_file.exists():
    content = plugin_file.read_text()
    
    # Fix 1: Make TaskTypeRegistryMeta inherit from ABCMeta
    old_meta_class = "class TaskTypeRegistryMeta(type):"
    new_meta_class = "class TaskTypeRegistryMeta(ABCMeta):"
    
    if old_meta_class in content:
        content = content.replace(old_meta_class, new_meta_class)
        print("✓ Fixed TaskTypeRegistryMeta to inherit from ABCMeta")
    
    # Fix 2: Add the import for ABCMeta
    if "from abc import ABC, abstractmethod" in content and "ABCMeta" not in content:
        content = content.replace(
            "from abc import ABC, abstractmethod",
            "from abc import ABC, abstractmethod, ABCMeta"
        )
        print("✓ Added ABCMeta import")
    
    # Fix 3: Remove the metaclass from ExampleTaskTypePlugin since it will inherit it
    old_example = "class ExampleTaskTypePlugin(TaskTypePlugin, metaclass=TaskTypeRegistryMeta):"
    new_example = "class ExampleTaskTypePlugin(TaskTypePlugin):"
    
    if old_example in content:
        content = content.replace(old_example, new_example)
        print("✓ Removed redundant metaclass from ExampleTaskTypePlugin")
    
    plugin_file.write_text(content)
    print("\n✅ Fixed metaclass conflict in task_type_plugin_base.py")
else:
    print("❌ File not found: scripts/task_type_plugin_base.py")