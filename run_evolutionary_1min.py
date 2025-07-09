#!/usr/bin/env python3
"""
Simple wrapper to run evolutionary improvements for 1 minute.
"""

import sys
import os
import subprocess

# Use subprocess to avoid any import/asyncio issues
result = subprocess.run([
    'arch', '-arm64', '/usr/bin/python3', '-c',
    """
import sys
sys.path.insert(0, '.')
from run_evolutionary_improvements import EvolutionaryImprovementRunner
import asyncio

async def run():
    runner = EvolutionaryImprovementRunner(runtime_minutes=1)
    await runner.run()

asyncio.run(run())
"""
], cwd=os.path.dirname(os.path.abspath(__file__)))