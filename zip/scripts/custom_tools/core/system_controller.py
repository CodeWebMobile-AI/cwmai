#!/usr/bin/env python3
"""
System Controller - Simplified Version

Controls the CWMAI system using a single command:
export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Any


class SystemController:
    """Simple system controller for CWMAI."""
    
    def __init__(self):
        self.project_root = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip")
        self.env_file = self.project_root / ".env.local"
        
    def start_system(self) -> Dict[str, Any]:
        """Start the CWMAI system."""
        if not self.env_file.exists():
            return {
                "success": False,
                "error": f".env.local not found at {self.env_file}"
            }
            
        # Change to project directory
        os.chdir(self.project_root)
        
        # Build the command
        cmd = (
            f"export $(cat {self.env_file} | grep -v '^#' | xargs) && "
            f"PYTHONPATH={self.project_root} arch -arm64 /usr/bin/python3 "
            f"run_hybrid_continuous.py --workers 10"
        )
        
        try:
            # Execute the command
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return {
                "success": True,
                "message": "System started successfully",
                "pid": process.pid,
                "command": cmd
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Tool interface for the conversational AI
async def main(action: str = "start", **kwargs) -> Dict[str, Any]:
    """Entry point for system control.
    
    Args:
        action: Only 'start' is supported
        
    Returns:
        Result of the operation
    """
    controller = SystemController()
    
    if action == "start":
        return controller.start_system()
    else:
        return {
            "success": False,
            "error": f"Only 'start' action is supported. Got: {action}"
        }


if __name__ == "__main__":
    # Test
    import asyncio
    result = asyncio.run(main("start"))
    print(result)