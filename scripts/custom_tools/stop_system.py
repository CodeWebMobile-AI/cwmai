"""
AI-Generated Tool: stop_system
Description: Stop all CWMAI system components gracefully
Generated: 2025-06-15
Requirements: Stop the continuous AI system, workers, and monitoring processes
"""

import subprocess
import asyncio
import os
import signal
from typing import Dict, Any, List
import psutil
import time

__description__ = "Stop all CWMAI system components gracefully"
__parameters__ = {
    "force": {
        "type": "boolean",
        "description": "Force stop (SIGKILL) if graceful shutdown fails",
        "required": False,
        "default": False
    },
    "timeout": {
        "type": "integer",
        "description": "Timeout in seconds for graceful shutdown",
        "required": False,
        "default": 10
    }
}
__examples__ = [
    {"description": "Stop system gracefully", "code": "await stop_system()"},
    {"description": "Force stop system", "code": "await stop_system(force=True)"},
    {"description": "Stop with custom timeout", "code": "await stop_system(timeout=30)"}
]


async def stop_system(**kwargs) -> Dict[str, Any]:
    """Stop all CWMAI system components gracefully."""
    force = kwargs.get('force', False)
    timeout = kwargs.get('timeout', 10)
    
    stopped_processes = []
    failed_processes = []
    
    try:
        # Find all CWMAI processes
        cwmai_processes = find_cwmai_processes()
        
        if not cwmai_processes:
            return {
                "status": "info",
                "message": "No CWMAI processes found running",
                "summary": "System already stopped"
            }
        
        # Stop each process
        for proc_info in cwmai_processes:
            try:
                process = psutil.Process(proc_info['pid'])
                proc_name = proc_info['name']
                
                # Try graceful termination first
                process.terminate()
                
                # Wait for process to stop
                try:
                    process.wait(timeout=timeout)
                    stopped_processes.append({
                        "name": proc_name,
                        "pid": proc_info['pid'],
                        "method": "graceful"
                    })
                except psutil.TimeoutExpired:
                    # Process didn't stop gracefully
                    if force:
                        process.kill()
                        try:
                            process.wait(timeout=5)
                            stopped_processes.append({
                                "name": proc_name,
                                "pid": proc_info['pid'],
                                "method": "forced"
                            })
                        except psutil.TimeoutExpired:
                            failed_processes.append({
                                "name": proc_name,
                                "pid": proc_info['pid'],
                                "error": "Failed to kill process"
                            })
                    else:
                        failed_processes.append({
                            "name": proc_name,
                            "pid": proc_info['pid'],
                            "error": "Process did not stop gracefully (use force=True)"
                        })
                        
            except psutil.NoSuchProcess:
                # Process already stopped
                stopped_processes.append({
                    "name": proc_info['name'],
                    "pid": proc_info['pid'],
                    "method": "already_stopped"
                })
            except psutil.AccessDenied:
                failed_processes.append({
                    "name": proc_info['name'],
                    "pid": proc_info['pid'],
                    "error": "Access denied"
                })
            except Exception as e:
                failed_processes.append({
                    "name": proc_info['name'],
                    "pid": proc_info['pid'],
                    "error": str(e)
                })
        
        # Clean up any orphaned resources
        await cleanup_resources()
        
        # Prepare result
        total_processes = len(cwmai_processes)
        total_stopped = len(stopped_processes)
        total_failed = len(failed_processes)
        
        if total_failed == 0:
            status = "success"
            message = f"All {total_stopped} processes stopped successfully"
        elif total_stopped > 0:
            status = "partial"
            message = f"Stopped {total_stopped}/{total_processes} processes"
        else:
            status = "failed"
            message = "Failed to stop any processes"
        
        return {
            "status": status,
            "message": message,
            "stopped": stopped_processes,
            "failed": failed_processes,
            "force_used": force,
            "summary": f"{total_stopped} stopped, {total_failed} failed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"System shutdown error: {str(e)}",
            "summary": "Failed to stop system"
        }


def find_cwmai_processes() -> List[Dict[str, Any]]:
    """Find all running CWMAI processes."""
    cwmai_processes = []
    
    # Process patterns to look for
    process_patterns = [
        ("run_continuous_ai.py", "continuous_ai"),
        ("run_dynamic_ai.py", "dynamic_ai"),
        ("continuous_orchestrator.py", "orchestrator"),
        ("worker_status_monitor", "worker_monitor"),
        ("task_coordinator.py", "task_coordinator"),
        ("ai_worker_agent.py", "ai_worker"),
        ("swarm_intelligence.py", "swarm_intelligence"),
        ("multi_repo_coordinator.py", "repo_coordinator"),
        ("intelligent_task_generator.py", "task_generator")
    ]
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                for pattern, friendly_name in process_patterns:
                    if pattern in cmdline_str:
                        cwmai_processes.append({
                            "pid": proc.info['pid'],
                            "name": friendly_name,
                            "pattern": pattern,
                            "create_time": proc.info['create_time']
                        })
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Sort by creation time (oldest first)
    cwmai_processes.sort(key=lambda x: x['create_time'])
    
    return cwmai_processes


async def cleanup_resources():
    """Clean up any orphaned resources like lock files, temp files, etc."""
    try:
        # Clean up lock files
        lock_files = [
            "continuous_orchestrator.lock",
            "worker_monitor.lock",
            "system_state.lock"
        ]
        
        for lock_file in lock_files:
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass
        
        # Clean up PID files
        pid_files = [
            "cwmai.pid",
            "orchestrator.pid",
            "worker.pid"
        ]
        
        for pid_file in pid_files:
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except:
                    pass
                    
    except Exception:
        # Cleanup is best effort, don't fail the stop operation
        pass