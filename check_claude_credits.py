#!/usr/bin/env python3
"""Check Claude credit status."""

import subprocess
import sys

def check_claude_credits():
    print("Checking Claude credit status...")
    
    # Run a minimal Claude command
    result = subprocess.run(
        ['claude', '--dangerously-skip-permissions'],
        input=b'exit',
        capture_output=True
    )
    
    print(f"\nReturn code: {result.returncode}")
    print(f"Stdout: {result.stdout.decode()}")
    print(f"Stderr: {result.stderr.decode()}")
    
    if "Credit balance is too low" in result.stdout.decode():
        print("\n❌ ISSUE FOUND: Claude account has insufficient credits!")
        print("\nTo fix this issue:")
        print("1. Visit https://console.anthropic.com/settings/billing")
        print("2. Add credits to your Claude account")
        print("3. Ensure your API key in .env.local has billing enabled")
        return False
    
    return True

if __name__ == "__main__":
    success = check_claude_credits()
    sys.exit(0 if success else 1)