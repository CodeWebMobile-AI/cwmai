"""
AI-Generated Tool: start_system
Description: Start all CWMAI system components
Generated: 2025-06-15
Requirements: Start the continuous AI system, workers, and monitoring
"""

import subprocess
import asyncio
import os
from pathlib import Path
from typing import Dict, Any
import psutil
import time

__description__ = "Start all CWMAI system components"
__parameters__ = {
    "mode": {
        "type": "string",
        "description": "Startup mode: 'full' (all components), 'minimal' (core only), 'workers' (workers only)",
        "required": False,
        "default": "full"
    }
}
__examples__ = [
    {"description": "Start full system", "code": "await start_system()"},
    {"description": "Start minimal system", "code": "await start_system(mode='minimal')"},
    {"description": "Start workers only", "code": "await start_system(mode='workers')"}
]


async def start_system(**kwargs) -> Dict[str, Any]:
    """Start all CWMAI system components."""
    mode = kwargs.get('mode', 'full')
    started_components = []
    failed_components = []
    
    try:
        # Check if already running
        running_processes = check_running_processes()
        if running_processes:
            return {
                "status": "warning",
                "message": "Some components already running",
                "running": running_processes,
                "summary": f"{len(running_processes)} components already running"
            }
        
        # Define components to start based on mode
        components = []
        
        if mode in ['full', 'minimal']:
            # Core components
            components.extend([
                {
                    "name": "continuous_ai",
                    "command": ["python", "run_continuous_ai.py"],
                    "description": "Continuous AI orchestrator"
                },
                {
                    "name": "dynamic_ai",
                    "command": ["python", "run_dynamic_ai.py"],
                    "description": "Dynamic AI system"
                }
            ])
        
        if mode == 'full':
            # Additional components for full mode
            components.extend([
                {
                    "name": "worker_monitor",
                    "command": ["python", "scripts/worker_status_monitor.py"],
                    "description": "Worker status monitor"
                },
                {
                    "name": "task_coordinator",
                    "command": ["python", "scripts/task_coordinator.py"],
                    "description": "Task coordination system"
                }
            ])
        
        if mode in ['full', 'workers']:
            # Worker components
            components.extend([
                {
                    "name": "ai_worker",
                    "command": ["python", "scripts/ai_worker_agent.py"],
                    "description": "AI worker agent"
                },
                {
                    "name": "swarm_intelligence",
                    "command": ["python", "scripts/swarm_intelligence.py"],
                    "description": "Swarm intelligence coordinator"
                }
            ])
        
        # Start each component
        for component in components:
            try:
                # Check if script exists
                script_path = component["command"][1]
                if not os.path.exists(script_path):
                    failed_components.append({
                        "name": component["name"],
                        "error": f"Script not found: {script_path}"
                    })
                    continue
                
                # Start the process
                process = subprocess.Popen(
                    component["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
                
                # Wait a moment to check if it started successfully
                await asyncio.sleep(0.5)
                
                if process.poll() is None:  # Process is running
                    started_components.append({
                        "name": component["name"],
                        "pid": process.pid,
                        "description": component["description"]
                    })
                else:
                    # Process failed to start
                    stdout, stderr = process.communicate()
                    failed_components.append({
                        "name": component["name"],
                        "error": stderr.decode() if stderr else "Failed to start"
                    })
                    
            except Exception as e:
                failed_components.append({
                    "name": component["name"],
                    "error": str(e)
                })
        
        # Prepare result
        total_attempted = len(components)
        total_started = len(started_components)
        total_failed = len(failed_components)
        
        if total_failed == 0:
            status = "success"
            message = f"All {total_started} components started successfully"
        elif total_started > 0:
            status = "partial"
            message = f"Started {total_started}/{total_attempted} components"
        else:
            status = "failed"
            message = "Failed to start any components"
        
        return {
            "status": status,
            "message": message,
            "started": started_components,
            "failed": failed_components,
            "mode": mode,
            "summary": f"{total_started} started, {total_failed} failed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"System startup error: {str(e)}",
            "summary": "Failed to start system"
        }


def check_running_processes():
    """Check which CWMAI processes are already running."""
    running = []
    
    # Process names to check
    process_patterns = [
        "run_continuous_ai.py",
        "run_dynamic_ai.py",
        "worker_status_monitor.py",
        "task_coordinator.py",
        "ai_worker_agent.py",
        "swarm_intelligence.py"
    ]
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                for pattern in process_patterns:
                    if pattern in cmdline_str:
                        running.append({
                            "name": pattern.replace('.py', ''),
                            "pid": proc.info['pid']
                        })
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return running