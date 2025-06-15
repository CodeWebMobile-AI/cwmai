#!/usr/bin/env python3
"""
Run Smart CWMAI CLI

Main entry point for the intelligent natural language interface.
"""

import sys
import os

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.smart_cwmai_cli import main
import asyncio

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)