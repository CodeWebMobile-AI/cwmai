#!/usr/bin/env python3
"""
Safe runner for executing Python scripts from within the zip folder.
Ensures all paths are resolved correctly relative to the zip directory.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_from_zip(script_name, *args):
    """Run a Python script from within the zip folder with proper environment setup."""
    # Get the zip directory
    zip_dir = Path(__file__).parent / "zip"
    
    if not zip_dir.exists():
        print(f"Error: Zip directory not found at {zip_dir}")
        return 1
    
    # Change to zip directory
    original_dir = os.getcwd()
    os.chdir(zip_dir)
    
    print(f"Running from: {os.getcwd()}")
    
    # Set up environment
    env = os.environ.copy()
    
    # Load .env.local from parent directory
    env_file = Path("../.env.local")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip()
    
    # Set PYTHONPATH to zip directory
    env["PYTHONPATH"] = str(zip_dir)
    
    # Build command
    cmd = ["arch", "-arm64", "/usr/bin/python3", script_name] + list(args)
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the script
        result = subprocess.run(cmd, env=env)
        return result.returncode
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 zip_safe_run.py <script_name> [args...]")
        print("Example: python3 zip_safe_run.py run_hybrid_continuous.py --workers 10")
        sys.exit(1)
    
    # Run the script
    exit_code = run_from_zip(*sys.argv[1:])
    sys.exit(exit_code)