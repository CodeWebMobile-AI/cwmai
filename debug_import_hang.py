#!/usr/bin/env python3
"""
Debug where the import is hanging.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports step by step...")

print("1. Importing logging...")
import logging

print("2. Importing asyncio...")
import asyncio

print("3. Importing time...")
import time

print("4. Importing datetime...")
from datetime import datetime, timezone, timedelta

print("5. Importing hashlib...")
import hashlib

print("6. Importing json...")
import json

print("7. Importing traceback...")
import traceback

print("8. Importing pathlib...")
from pathlib import Path

print("9. Importing StateManager...")
from scripts.state_manager import StateManager

print("10. About to import run_evolutionary_improvements...")
print("    This is where it might hang...")

# Import just the file without running main
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_evolutionary_improvements", 
    "run_evolutionary_improvements.py"
)
module = importlib.util.module_from_spec(spec)

print("11. Module created, about to execute...")
# Don't execute the module yet
print("✅ All imports successful!")

# Now let's see what happens when we actually import it normally
print("\n12. Now importing normally (this might hang)...")
import run_evolutionary_improvements

print("✅ Normal import succeeded!")