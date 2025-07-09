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
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import logging

from scripts.state_manager import StateManager  # Assuming this exists

__description__ = "Check the status of all CWMAI system components"
__parameters__ = {
    "verbose": {
        "type": "boolean",
        "description": "Show detailed information for each component",
        "required": False,
        "default": False
    },
    "cache_expiry": {
        "type": "integer",
        "description": "Cache expiry in seconds. 0 disables caching.",
        "required": False,
        "default": 60
    },
    "component_filter": {
        "type": "string",
        "description": "Filter components by name (comma-separated).",
        "required": False,
        "default": None
    },
    "log_level": {
        "type": "string",
        "description": "Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        "required": False,
        "default": "INFO"
    },
    "include_network_stats":{
        "type": "boolean",
        "description": "Include detailed network statistics for each component, requires verbose=True.",
        "required": False,
        "default": False
    }
}
__examples__ = [
    {"description": "Quick status check", "code": "await check_system_status()"},
    {"description": "Detailed status check", "code": "await check_system_status(verbose=True)"},
    {"description": "Cached status check (5 minutes)", "code": "await check_system_status(cache_expiry=300)"},
    {"description": "Check specific components", "code": "await check_system_status(component_filter='continuous_ai,dynamic_ai')"},
    {"description": "Set log level to debug", "code": "await check_system_status(log_level='DEBUG')"},
    {"description": "Detailed status check with network stats", "code": "await check_system_status(verbose=True, include_network_stats=True)"}
]

# Global cache
_system_status_cache: Dict[str, Any] = {}
_system_status_cache_expiry: Optional[float] = None  # Epoch time for expiry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def check_system_status(**kwargs) -> Dict[str, Any]:
    """Check the status of all CWMAI system components."""

    # Validate input parameters
    validation_schema = {
        "verbose": {"type": "boolean"},
        "cache_expiry": {"type": "integer", "min": 0},
        "component_filter": {"type": "string"},
        "log_level": {"type": "string", "allowed_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
        "include_network_stats": {"type": "boolean"}
    }
    validation_error = validate_input(kwargs, validation_schema)
    if validation_error:
        logger.error(f"Input validation error: {validation_error}")
        return {
            "status": "error",
            "error": validation_error,
            "summary": "Invalid input parameters."
        }


    verbose = kwargs.get('verbose', False)
    cache_expiry = kwargs.get('cache_expiry', 60)
    component_filter_str = kwargs.get('component_filter')
    log_level_str = kwargs.get('log_level', 'INFO').upper()
    include_network_stats = kwargs.get('include_network_stats', False)


    try:
        log_level = getattr(logging, log_level_str)  # Convert string to logging level
        logger.setLevel(log_level)
    except AttributeError:
        logger.warning(f"Invalid log level '{log_level_str}'.  Using default: INFO")


    cache_key = f"system_status_verbose_{verbose}_networkstats_{include_network_stats}" #Include network stats in cache key

    if cache_expiry > 0 and _system_status_cache.get(cache_key) and _system_status_cache_expiry and time.time() < _system_status_cache_expiry:
        logger.info("Using cached system status.")
        return _system_status_cache[cache_key]

    try:
        # Get component status
        components = await get_component_status(verbose, component_filter_str, include_network_stats)

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

        # Cache the result
        if cache_expiry > 0:
            _system_status_cache[cache_key] = result
            _system_status_cache_expiry = time.time() + cache_expiry
            logger.info(f"System status cached for {cache_expiry} seconds.")

        return result

    except Exception as e:
        error_message = f"Status check failed: {str(e)}"
        logger.exception(error_message)  # Use logger.exception for full stack trace
        return {
            "status": "error",
            "error": error_message,
            "summary": "Unable to determine system status"
        }


async def get_component_status(verbose: bool, component_filter: Optional[str] = None, include_network_stats:bool=False) -> List[Dict[str, Any]]:
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

    # Apply component filter
    if component_filter:
        filtered_components = [c for c in component_definitions if c["name"] in component_filter.split(',')]
        if not filtered_components:
            logger.warning(f"No components found matching filter: {component_filter}")
            return []  # Or raise an exception if filter is required
        component_definitions = filtered_components

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
        process = find_process(comp_def["process_pattern"])
        if process:
            component["status"] = "running"
            component["pid"] = process.pid

            # Calculate uptime
            create_time = process.create_time()
            uptime_seconds = time.time() - create_time
            component["uptime"] = format_uptime(uptime_seconds)

            if verbose:
                # Add detailed info
                try:
                    component["memory_mb"] = process.memory_info().rss / 1024 / 1024
                    component["cpu_percent"] = process.cpu_percent(interval=0.1)
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {process.pid} no longer exists while fetching detailed information.")
                    component["status"] = "stopped" # process disappeared
                    continue
                except Exception as e:
                     logger.error(f"Error getting process details for {process.pid}: {e}")

                if include_network_stats:
                    try:
                        net_io = process.io_counters()
                        component["network"] = {
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "read_count": net_io.read_count,
                            "write_count": net_io.write_count
                        }
                    except psutil.NoSuchProcess:
                        logger.warning(f"Process {process.pid} no longer exists while fetching network information.")
                        component["status"] = "stopped"
                        continue
                    except Exception as e:
                        logger.error(f"Error getting network stats for {process.pid}: {e}")


        components.append(component)

    return components

def find_process(process_pattern:str) -> Optional[psutil.Process]:
    """Helper function to find a process by its command line."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info', 'cpu_percent']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and process_pattern in ' '.join(cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Skipping process: {e}")
                continue
            except Exception as e:
                logger.error(f"Error getting process info: {e}")  # Log specific errors
                continue
    except Exception as e:
        logger.error(f"Error iterating processes: {e}")
    return None

def get_system_resources() -> Dict[str, Any]:
    """Get system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory usage
        memory = psutil.virtual_memory()

        # Disk usage
        disk = psutil.disk_usage('/')

        # Network usage
        net_io = psutil.net_io_counters()

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
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {
            "cpu": {"usage_percent": 0, "cores": 0, "status": "unknown"},
            "memory": {"used_gb": 0, "total_gb": 0, "percent": 0, "available_gb": 0, "status": "unknown"},
            "disk": {"used_gb": 0, "total_gb": 0, "percent": 0, "free_gb": 0, "status": "unknown"},
            "network": {"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0}
        }


async def get_task_status() -> Dict[str, Any]:
    """Get task queue and execution status."""
    try:
        state_manager = StateManager()
        state = state_manager.load_state()

        tasks = state.get('active_tasks', [])
        completed_tasks = state.get('completed_tasks', [])
        failed_tasks = state.get('failed_tasks', [])
        total_tasks = len(tasks) + len(completed_tasks) + len(failed_tasks)

        # Calculate average task execution time
        completed_task_times = [t.get('completed_at') - t.get('created_at') for t in completed_tasks if t.get('completed_at') and t.get('created_at')]
        avg_task_time = sum(completed_task_times) / len(completed_task_times) if completed_task_times else 0

        # Get detailed information about the oldest task
        oldest_task = min(tasks, key=lambda t: t.get('created_at', datetime.now()), default=None)

        return {
            "active": len(tasks),
            "completed_today": len([t for t in completed_tasks if t.get('completed_at', datetime.min).strftime('%Y-%m-%d') == datetime.now().strftime('%Y-%m-%d')]),
            "failed_today": len([t for t in failed_tasks if t.get('failed_at', datetime.min).strftime('%Y-%m-%d') == datetime.now().strftime('%Y-%m-%d')]),
            "queue_health": "healthy" if len(tasks) < 100 else "backlogged",
            "total": total_tasks,
            "average_execution_time": avg_task_time,
            "oldest_task": oldest_task
        }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return {
            "active": 0,
            "completed_today": 0,
            "failed_today": 0,
            "queue_health": "unknown",
            "total": 0,
            "average_execution_time": 0,
            "oldest_task": None
        }


async def get_worker_status() -> Dict[str, Any]:
    """Get worker status information."""
    workers = []
    total_workers = 0
    active_workers = 0 # Track number of workers that are not idle
    idle_workers = 0

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

                        # Simulate some logic to determine if a worker is active vs idle
                        # This is a placeholder - replace with actual status check.
                        cpu_usage = proc.cpu_percent(interval=0.1)
                        if cpu_usage > 1.0:  # Assuming CPU usage is a rough indicator
                            active_workers += 1
                        else:
                            idle_workers += 1
                        break # Stop after finding a matching worker pattern

        except Exception as e:
            logger.error(f"Error getting worker process info: {e}")
            continue

    status = "healthy" if total_workers > 0 else "no_workers"

    if total_workers > 0 and active_workers == 0:
        status = "idle" # All workers found but all idle
    elif total_workers == 0:
        status = "no_workers"
    elif active_workers < total_workers:
        status = "partially_idle" #Some workers idle

    return {
        "total": total_workers,
        "active": active_workers,
        "idle": idle_workers,
        "status": status
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
            "connected_clients": info.get('connected_clients', 0),
            "uptime_seconds": info.get('uptime_in_seconds', 0)
        }
    except Exception as e:
        logger.error(f"Error checking Redis status: {e}")
        return {
            "status": "disconnected",
            "error": f"Redis not available: {e}"
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
    elif workers['status'] == "idle":
        warnings.append("All workers idle")

    # Task queue health
    if tasks['queue_health'] == "backlogged":
        warnings.append("Task queue is backlogged")

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
    worker_score = 20
    if workers['total'] == 0:
        worker_score = 0
    elif workers['status'] == "idle":
        worker_score = 10 # Reduce points for all idle

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
                try:
                    stat = file_path.stat()
                    log_files.append({
                        "name": str(file_path),
                        "size_mb": stat.st_size / (1024*1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "path": str(file_path.resolve())  #Absolute path
                    })
                except Exception as e:
                    logger.error(f"Error getting log file stats for {file_path}: {e}")  # Log file access issues

    return log_files


def check_config_files() -> List[Dict[str, Any]]:
    """Check configuration files."""
    config_files = []
    configs = [".env", ".env.local", "config.json", "settings.json"]

    for config in configs:
        try:
            if os.path.exists(config):
                config_files.append({
                    "name": config,
                    "exists": True,
                    "readable": os.access(config, os.R_OK),
                    "path": str(Path(config).resolve()) #Absolute Path
                })
        except Exception as e:
            logger.error(f"Error checking config file {config}: {e}")  # log config issues

    return config_files


def check_state_files() -> List[Dict[str, Any]]:
    """Check state files."""
    state_files = []
    states = ["system_state.json", "task_state.json", "continuous_orchestrator_state.json"]

    for state_file in states:
        try:
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                    state_files.append({
                        "name": state_file,
                        "valid": True,
                        "size_kb": os.path.getsize(state_file) / 1024,
                        "path": str(Path(state_file).resolve()), #Absolute path
                        "data_preview": str(data)[:100] + "..." if isinstance(data, dict) else str(data)[:100] + "..." #Preview
                    })
                except json.JSONDecodeError as e:
                    state_files.append({
                        "name": state_file,
                        "valid": False,
                        "error": f"Invalid JSON: {e}"
                    })
                except Exception as e:
                    state_files.append({
                        "name": state_file,
                        "valid": False,
                        "error": f"Error reading file: {e}"
                    })
        except Exception as e:
            logger.error(f"Error checking state file {state_file}: {e}")

    return state_files


def validate_input(data: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """Validate input data against a schema.  Returns an error message, or None if valid."""
    for field, constraints in schema.items():
        if 'required' in constraints and constraints['required'] and field not in data:
            return f"Missing required field: {field}"

        if field in data and 'type' in constraints:
            expected_type = constraints['type']
            actual_type = type(data[field]).__name__ #get string name of type
            if actual_type != expected_type:
                return f"Field '{field}' should be of type '{expected_type}', but got '{actual_type}'."

            if 'min' in constraints and isinstance(data[field], (int, float)) and data[field] < constraints['min']:
                return f"Field '{field}' should be at least {constraints['min']}."

            if 'max' in constraints and isinstance(data[field], (int, float)) and data[field] > constraints['max']:
                return f"Field '{field}' should be at most {constraints['max']}."

            if 'allowed_values' in constraints and data[field] not in constraints['allowed_values']:
                  return f"Field '{field}' must be one of the following: {constraints['allowed_values']}"

    return None  # No errors found


# Example usage (replace with actual calls)
async def main():
    # Example 1: Quick status check
    status1 = await check_system_status()
    print("Status 1:", status1['summary'])

    # Example 2: Detailed status check
    status2 = await check_system_status(verbose=True)
    print("Status 2:", status2['summary'])
    if "details" in status2:
        print("Log Files:", status2['details']['log_files'])

    # Example 2a: Detailed status check with network stats
    status2a = await check_system_status(verbose=True, include_network_stats=True)
    print("Status 2a:", status2a['summary'])
    if "components" in status2a:
        for component in status2a["components"]:
            if "network" in component:
                print(f"Component {component['name']} Network Stats: {component['network']}")


    # Example 3: Cached status check (5 minutes)
    status3 = await check_system_status(cache_expiry=300)
    print("Status 3:", status3['summary'])

    # Example 4: Filtered Component Check
    status4 = await check_system_status(component_filter="continuous_ai,dynamic_ai")
    print("Status 4:", status4['summary'])
    print("Components:", status4['components'])

    # Example 5: Debug logging
    status5 = await check_system_status(log_level="DEBUG", verbose=True)
    print("Status 5:", status5['summary'])

    # Example 6: Invalid Log level
    status6 = await check_system_status(log_level="BLAH", verbose=True)
    print("Status 6:", status6['summary'])



if __name__ == "__main__":
    asyncio.run(main())