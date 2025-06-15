"""
AI-Generated Tool: check_system_status
Description: Check the status of all CWMAI system components
Generated: 2025-06-15
Requirements: Show detailed status of all system components, workers, and resources
"""

import psutil
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import json

from scripts.state_manager import StateManager

__description__ = "Check the status of all CWMAI system components"
__parameters__ = {
    "verbose": {
        "type": "boolean",
        "description": "Show detailed information for each component",
        "required": False,
        "default": False
    }
}
__examples__ = [
    {"description": "Quick status check", "code": "await check_system_status()"},
    {"description": "Detailed status check", "code": "await check_system_status(verbose=True)"}
]


async def check_system_status(**kwargs) -> Dict[str, Any]:
    """Check the status of all CWMAI system components."""
    verbose = kwargs.get('verbose', False)
    
    try:
        # Get component status
        components = await get_component_status(verbose)
        
        # Get system resources
        resources = get_system_resources()
        
        # Get task queue status
        task_status = await get_task_status()
        
        # Get worker status
        worker_status = await get_worker_status()
        
        # Check Redis status
        redis_status = check_redis_status()
        
        # Overall health assessment
        health = assess_system_health(components, resources, task_status, worker_status)
        
        # Build summary
        running_count = len([c for c in components if c['status'] == 'running'])
        total_count = len(components)
        
        summary = f"System {health['status']}: {running_count}/{total_count} components running"
        
        result = {
            "health": health,
            "components": components,
            "resources": resources,
            "tasks": task_status,
            "workers": worker_status,
            "redis": redis_status,
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        
        if verbose:
            result["details"] = {
                "log_files": check_log_files(),
                "config_files": check_config_files(),
                "state_files": check_state_files()
            }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Status check failed: {str(e)}",
            "summary": "Unable to determine system status"
        }


async def get_component_status(verbose: bool) -> List[Dict[str, Any]]:
    """Get status of all CWMAI components."""
    components = []
    
    # Define all components to check
    component_definitions = [
        {
            "name": "continuous_ai",
            "process_pattern": "run_continuous_ai.py",
            "description": "Main AI orchestrator",
            "critical": True
        },
        {
            "name": "dynamic_ai",
            "process_pattern": "run_dynamic_ai.py",
            "description": "Dynamic AI system",
            "critical": True
        },
        {
            "name": "worker_monitor",
            "process_pattern": "worker_status_monitor",
            "description": "Worker monitoring system",
            "critical": False
        },
        {
            "name": "task_coordinator",
            "process_pattern": "task_coordinator.py",
            "description": "Task coordination service",
            "critical": False
        },
        {
            "name": "swarm_intelligence",
            "process_pattern": "swarm_intelligence.py",
            "description": "Swarm AI coordinator",
            "critical": False
        },
        {
            "name": "repo_coordinator",
            "process_pattern": "multi_repo_coordinator.py",
            "description": "Multi-repository coordinator",
            "critical": False
        }
    ]
    
    # Check each component
    for comp_def in component_definitions:
        component = {
            "name": comp_def["name"],
            "description": comp_def["description"],
            "critical": comp_def["critical"],
            "status": "stopped",
            "pid": None,
            "uptime": None
        }
        
        # Find process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info', 'cpu_percent']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and comp_def["process_pattern"] in ' '.join(cmdline):
                    component["status"] = "running"
                    component["pid"] = proc.info['pid']
                    
                    # Calculate uptime
                    create_time = proc.info['create_time']
                    uptime_seconds = time.time() - create_time
                    component["uptime"] = format_uptime(uptime_seconds)
                    
                    if verbose:
                        # Add detailed info
                        component["memory_mb"] = proc.info['memory_info'].rss / 1024 / 1024
                        component["cpu_percent"] = proc.cpu_percent(interval=0.1)
                    
                    break
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        components.append(component)
    
    return components


def get_system_resources() -> Dict[str, Any]:
    """Get system resource usage."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "cores": cpu_count,
            "status": "ok" if cpu_percent < 80 else "high"
        },
        "memory": {
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3),
            "percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "status": "ok" if memory.percent < 80 else "high"
        },
        "disk": {
            "used_gb": disk.used / (1024**3),
            "total_gb": disk.total / (1024**3),
            "percent": disk.percent,
            "free_gb": disk.free / (1024**3),
            "status": "ok" if disk.percent < 90 else "high"
        }
    }


async def get_task_status() -> Dict[str, Any]:
    """Get task queue and execution status."""
    try:
        state_manager = StateManager()
        state = state_manager.load_state()
        
        tasks = state.get('active_tasks', [])
        completed_tasks = state.get('completed_tasks', [])
        failed_tasks = state.get('failed_tasks', [])
        
        return {
            "active": len(tasks),
            "completed_today": len([t for t in completed_tasks if t.get('completed_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))]),
            "failed_today": len([t for t in failed_tasks if t.get('failed_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))]),
            "queue_health": "healthy" if len(tasks) < 100 else "backlogged"
        }
    except:
        return {
            "active": 0,
            "completed_today": 0,
            "failed_today": 0,
            "queue_health": "unknown"
        }


async def get_worker_status() -> Dict[str, Any]:
    """Get worker status information."""
    workers = []
    total_workers = 0
    
    # Find worker processes
    worker_patterns = [
        "ai_worker_agent",
        "worker_process",
        "task_worker"
    ]
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                for pattern in worker_patterns:
                    if pattern in cmdline_str:
                        total_workers += 1
                        break
        except:
            continue
    
    return {
        "total": total_workers,
        "active": total_workers,  # Assuming all found workers are active
        "idle": 0,
        "status": "healthy" if total_workers > 0 else "no_workers"
    }


def check_redis_status() -> Dict[str, Any]:
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        
        # Get some stats
        info = r.info()
        
        return {
            "status": "connected",
            "version": info.get('redis_version', 'unknown'),
            "memory_used_mb": info.get('used_memory', 0) / (1024*1024),
            "connected_clients": info.get('connected_clients', 0)
        }
    except:
        return {
            "status": "disconnected",
            "error": "Redis not available"
        }


def assess_system_health(components, resources, tasks, workers) -> Dict[str, Any]:
    """Assess overall system health."""
    issues = []
    warnings = []
    
    # Check critical components
    critical_running = all(c['status'] == 'running' for c in components if c['critical'])
    if not critical_running:
        issues.append("Critical components not running")
    
    # Check resources
    if resources['cpu']['status'] != 'ok':
        warnings.append("High CPU usage")
    if resources['memory']['status'] != 'ok':
        warnings.append("High memory usage")
    if resources['disk']['status'] != 'ok':
        issues.append("Low disk space")
    
    # Check workers
    if workers['status'] == 'no_workers':
        warnings.append("No workers running")
    
    # Determine overall status
    if issues:
        status = "unhealthy"
    elif warnings:
        status = "degraded"
    else:
        status = "healthy"
    
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "score": calculate_health_score(components, resources, tasks, workers)
    }


def calculate_health_score(components, resources, tasks, workers) -> int:
    """Calculate a health score from 0-100."""
    score = 100
    
    # Component score (40 points)
    running = len([c for c in components if c['status'] == 'running'])
    total = len(components)
    component_score = (running / total) * 40 if total > 0 else 0
    
    # Resource score (30 points)
    resource_score = 30
    if resources['cpu']['usage_percent'] > 80:
        resource_score -= 10
    if resources['memory']['percent'] > 80:
        resource_score -= 10
    if resources['disk']['percent'] > 90:
        resource_score -= 10
    
    # Worker score (20 points)
    worker_score = 20 if workers['total'] > 0 else 0
    
    # Task score (10 points)
    task_score = 10 if tasks['queue_health'] == 'healthy' else 5
    
    total_score = int(component_score + resource_score + worker_score + task_score)
    return max(0, min(100, total_score))


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def check_log_files() -> List[Dict[str, Any]]:
    """Check status of log files."""
    log_files = []
    log_patterns = [
        "continuous_ai_log*.txt",
        "*.log",
        "logs/*.log"
    ]
    
    for pattern in log_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                log_files.append({
                    "name": str(file_path),
                    "size_mb": stat.st_size / (1024*1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    return log_files


def check_config_files() -> List[Dict[str, Any]]:
    """Check configuration files."""
    config_files = []
    configs = [".env", ".env.local", "config.json", "settings.json"]
    
    for config in configs:
        if os.path.exists(config):
            config_files.append({
                "name": config,
                "exists": True,
                "readable": os.access(config, os.R_OK)
            })
    
    return config_files


def check_state_files() -> List[Dict[str, Any]]:
    """Check state files."""
    state_files = []
    states = ["system_state.json", "task_state.json", "continuous_orchestrator_state.json"]
    
    for state_file in states:
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                state_files.append({
                    "name": state_file,
                    "valid": True,
                    "size_kb": os.path.getsize(state_file) / 1024
                })
            except:
                state_files.append({
                    "name": state_file,
                    "valid": False,
                    "error": "Invalid JSON"
                })
    
    return state_files


import time  # Add this import at the top