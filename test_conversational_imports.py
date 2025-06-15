#!/usr/bin/env python3
"""Test imports for conversational AI assistant."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    print("1. Importing AI brain...")
    from scripts.ai_brain import IntelligentAIBrain
    print("✓ AI brain imported")
except ImportError as e:
    print(f"✗ Failed to import AI brain: {e}")

try:
    print("\n2. Importing HTTP AI client...")
    from scripts.http_ai_client import HTTPAIClient
    print("✓ HTTP AI client imported")
except ImportError as e:
    print(f"✗ Failed to import HTTP AI client: {e}")

try:
    print("\n3. Importing task manager...")
    from scripts.task_manager import TaskManager
    print("✓ Task manager imported")
except ImportError as e:
    print(f"✗ Failed to import task manager: {e}")

try:
    print("\n4. Importing state manager...")
    from scripts.state_manager import StateManager
    print("✓ State manager imported")
except ImportError as e:
    print(f"✗ Failed to import state manager: {e}")

try:
    print("\n5. Importing natural language interface...")
    from scripts.natural_language_interface import NaturalLanguageInterface
    print("✓ Natural language interface imported")
except ImportError as e:
    print(f"✗ Failed to import natural language interface: {e}")

try:
    print("\n6. Importing smart natural language interface...")
    from scripts.smart_natural_language_interface import SmartNaturalLanguageInterface
    print("✓ Smart natural language interface imported")
except ImportError as e:
    print(f"✗ Failed to import smart natural language interface: {e}")

try:
    print("\n7. Importing conversational AI assistant...")
    from scripts.conversational_ai_assistant import ConversationalAIAssistant
    print("✓ Conversational AI assistant imported")
except ImportError as e:
    print(f"✗ Failed to import conversational AI assistant: {e}")

print("\nImport test complete!")