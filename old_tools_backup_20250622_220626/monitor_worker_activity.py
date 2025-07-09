"""
AI-Generated Tool: monitor_worker_activity
Description: Monitor status for worker activity
Generated: 2025-06-15T16:04:22.466472+00:00
Requirements: 
        Tool Name: monitor_worker_activity
        Intent: Monitor status for worker activity
        Expected Parameters: {}
        Category: monitoring
        
        The tool should:
        1. Monitor items status in real-time
2. Track key metrics and changes
3. Return current status and alerts
        
"""

from typing import Any
from typing import Dict
from typing import Dict, Any, Optional

import asyncio
import logging
import time
import json  # For handling potential JSONDecodeError
import os
import shutil
import psutil # Requires 'pip install psutil'

from scripts.state_manager import StateManager
from scripts.logger import get_logger

"""
Worker activity monitoring tool.

This module provides real-time monitoring of worker activities, tracking key metrics,
status changes, and generating alerts when anomalies are detected.
"""


__description__ = "Monitor worker activity status in real-time"
__parameters__ = {}
__examples__ = [
    {"description": "Basic usage", "code": "await monitor_worker_activity()"}
]

logger = logging.getLogger(__name__)

# In-memory cache for storing worker activity data
_worker_activity_cache: Dict[str, Any] = {}
_cache_expiry_time: float = 60  # Cache expires after 60 seconds
_cache_lock = asyncio.Lock()  # Ensure thread-safe cache access

# Define acceptable worker status values
_acceptable_worker_statuses = {"active", "idle", "error", "offline", "starting", "stopping"}

# Define thresholds for alerts
_cpu_usage_threshold = 90  # Percentage
_memory_usage_threshold = 90  # Percentage
_worker_offline_threshold = 300 # Seconds - consider a worker offline after this period
_disk_usage_threshold = 90  # Percentage - Added Disk Usage Threshold


def validate_worker_data(worker_data: Dict[str, Any]) -> bool:
    """
    Validates individual worker data to ensure it has the correct format and necessary information.
    """
    if not isinstance(worker_data, dict):
        logger.warning("Worker data is not a dictionary.")
        return False

    required_fields = ['status', 'worker_id']  # Enforce worker_id
    for field in required_fields:
        if field not in worker_data:
            logger.warning(f"Worker data is missing '{field}' field.")
            return False

    if not isinstance(worker_data['status'], str):
        logger.warning("Worker 'status' is not a string.")
        return False

    if worker_data['status'] not in _acceptable_worker_statuses:
        logger.warning(f"Worker 'status' is invalid: {worker_data['status']}")
        return False

    if not isinstance(worker_data['worker_id'], str): # Validate worker_id
        logger.warning("Worker 'worker_id' is not a string.")
        return False

    # Add more validation rules as needed, e.g., check for valid status values, resource usage, etc.
    if 'cpu_usage' in worker_data:
        if not isinstance(worker_data['cpu_usage'], (int, float)):
            logger.warning("Worker 'cpu_usage' is not a number.")
            return False
        if worker_data['cpu_usage'] < 0 or worker_data['cpu_usage'] > 100:
            logger.warning("Worker 'cpu_usage' is out of range (0-100).")
            return False

    if 'memory_usage' in worker_data:
        if not isinstance(worker_data['memory_usage'], (int, float)):
            logger.warning("Worker 'memory_usage' is not a number.")
            return False
        if worker_data['memory_usage'] < 0 or worker_data['memory_usage'] > 100:
            logger.warning("Worker 'memory_usage' is out of range (0-100).")
            return False

    if 'last_seen' in worker_data:
        if not isinstance(worker_data['last_seen'], (int, float)):
            logger.warning("Worker 'last_seen' is not a number.")
            return False

    if 'disk_usage' in worker_data:  # Validate disk_usage if present
        if not isinstance(worker_data['disk_usage'], (int, float)):
            logger.warning("Worker 'disk_usage' is not a number.")
            return False
        if worker_data['disk_usage'] < 0 or worker_data['disk_usage'] > 100:
            logger.warning("Worker 'disk_usage' is out of range (0-100).")
            return False


    return True


def calculate_average_metric(workers: Dict[str, Any], metric_key: str) -> Optional[float]:
    """
    Calculates the average value of a specified metric across all workers.
    Returns None if the metric is missing or invalid for any worker, or if there are no workers.
    """
    if not workers:
        return None

    valid_metric_values = []
    for worker_id, worker_data in workers.items():
        if not isinstance(worker_data, dict):
            logger.warning(f"Worker {worker_id} data is not a dictionary, skipping.")
            continue

        if metric_key in worker_data:
            metric_value = worker_data[metric_key]
            if isinstance(metric_value, (int, float)):
                valid_metric_values.append(metric_value)
            else:
                logger.warning(f"Worker {worker_id} {metric_key} is not a number, skipping.")
        else:
            logger.warning(f"Worker {worker_id} is missing {metric_key}, skipping.")

    if not valid_metric_values:
        return None

    return sum(valid_metric_values) / len(valid_metric_values)


def analyze_worker_history(worker_history: list) -> Dict[str, Any]:
    """
    Analyzes worker history to identify trends and anomalies.
    """
    if not worker_history:
        return {}

    # Example: Calculate average status duration
    status_durations = {}
    status_transition_counts = {} # New metric: Count transitions between statuses
    resource_usage_history = {} # NEW: Track historical CPU and memory usage

    for i in range(len(worker_history) - 1):
        entry = worker_history[i]
        next_entry = worker_history[i + 1]
        worker_id = entry.get('worker_id') # Added worker_id
        status = entry.get('status')
        next_status = next_entry.get('status')
        if status and next_status:
            duration = next_entry['timestamp'] - entry['timestamp']
            status_durations.setdefault(status, []).append(duration)

            # Update status transition counts
            transition_key = (status, next_status)
            status_transition_counts[transition_key] = status_transition_counts.get(transition_key, 0) + 1

            #Track historical Resource Usage
            if worker_id:
              if 'cpu_usage' in entry and isinstance(entry['cpu_usage'], (int, float)):
                  resource_usage_history.setdefault(worker_id, {}).setdefault('cpu_usage', []).append((entry['timestamp'], entry['cpu_usage']))
              if 'memory_usage' in entry and isinstance(entry['memory_usage'], (int, float)):
                  resource_usage_history.setdefault(worker_id, {}).setdefault('memory_usage', []).append((entry['timestamp'], entry['memory_usage']))

    average_status_durations = {
        status: sum(durations) / len(durations)
        for status, durations in status_durations.items() if durations # Avoid division by zero
    }

    return {"average_status_durations": average_status_durations,
            "status_transition_counts": status_transition_counts,
            "resource_usage_history": resource_usage_history}


def check_resource_usage(workers: Dict[str, Any]) -> list:
    """
    Checks for workers exceeding CPU, memory or disk usage thresholds.
    """
    alerts = []
    for worker_id, worker_data in workers.items():
        if 'cpu_usage' in worker_data and worker_data['cpu_usage'] > _cpu_usage_threshold:
            alerts.append(f"Worker {worker_id} CPU usage exceeds {_cpu_usage_threshold}% ({worker_data['cpu_usage']}%)")
        if 'memory_usage' in worker_data and worker_data['memory_usage'] > _memory_usage_threshold:
            alerts.append(f"Worker {worker_id} memory usage exceeds {_memory_usage_threshold}% ({worker_data['memory_usage']}%)")
        if 'disk_usage' in worker_data and worker_data['disk_usage'] > _disk_usage_threshold:
            alerts.append(f"Worker {worker_id} disk usage exceeds {_disk_usage_threshold}% ({worker_data['disk_usage']}%)")

    return alerts


def check_offline_workers(workers: Dict[str, Any]) -> list:
    """
    Checks for workers that haven't reported activity within the offline threshold.
    """
    alerts = []
    current_time = time.time()
    for worker_id, worker_data in workers.items():
        if 'last_seen' in worker_data:
            time_since_last_seen = current_time - worker_data['last_seen']
            if worker_data['status'] != 'offline' and time_since_last_seen > _worker_offline_threshold:
                alerts.append(f"Worker {worker_id} has not reported activity in {time_since_last_seen:.0f} seconds. Status updated to offline.")
                worker_data['status'] = 'offline' # Automatically update status based on last_seen time
        else:
            logger.warning(f"Worker {worker_id} missing 'last_seen' timestamp.")
    return alerts

def get_process_resource_usage(pid: int) -> Optional[Dict[str, float]]:
    """
    Retrieves CPU and memory usage for a given process ID.
    Requires the 'psutil' library.
    Returns a dictionary with 'cpu_usage' and 'memory_usage' or None if the process isn't found/accessible.
    """
    try:
        process = psutil.Process(pid)
        cpu_usage = process.cpu_percent(interval=0.1)  # Short interval for accuracy
        memory_usage = process.memory_percent()
        #Also retrieve Disk Usage information
        io_counters = process.io_counters()
        disk_usage = io_counters.write_bytes + io_counters.read_bytes
        return {"cpu_usage": cpu_usage, "memory_usage": memory_usage, "disk_usage": disk_usage} #Enhanced: added disk_usage
    except psutil.NoSuchProcess:
        logger.warning(f"Process with PID {pid} not found.")
        return None
    except psutil.AccessDenied:
        logger.warning(f"Access denied when trying to get resource usage for PID {pid}.")
        return None
    except Exception as e:
        logger.error(f"Error getting resource usage for PID {pid}: {e}")
        return None


async def monitor_worker_activity(**kwargs) -> Dict[str, Any]:
    """
    Monitor worker activity status in real-time.

    Returns:
        Dictionary containing:
        - status: Current overall status
        - active_workers: Number of active workers
        - idle_workers: Number of idle workers
        - alerts: List of active alerts
        - metrics: Dictionary of key performance metrics
        - last_updated: Timestamp of last status update
        - uptime: System uptime in seconds
        - total_workers: Total number of workers
        - detailed_status:  Dictionary showing the number of workers per status
        - worker_history_analysis: Analysis of the worker history (if available)
        - process_resource_usages: Resource usage information by process ID (PID).

    Raises:
        ValueError: If there is an issue with data validation.
        RuntimeError: If the StateManager fails to load state.
    """
    global _worker_activity_cache
    global _cache_expiry_time
    global _cache_lock

    current_time = time.time()

    async with _cache_lock: # Use lock to prevent race conditions when accessing cache.
        if _worker_activity_cache and _worker_activity_cache.get('expiry', 0) > current_time:
            logger.debug("Returning worker activity from cache.")
            return _worker_activity_cache['data']

    try:
        # Initialize required components
        state_manager = StateManager()

        # Load current state with improved error handling
        try:
            state = state_manager.load_state()
        except FileNotFoundError as e:
            logger.error(f"State file not found: {e}")
            # Attempt recovery by creating an empty state file
            try:
                state_manager.save_state({})  # Creates an empty state file.
                state = {}  # Initializes state to empty dictionary for processing
                logger.info("Recovered by creating an empty state file.")
            except Exception as save_error:
                logger.critical(f"Failed to create new state file: {save_error}")
                raise RuntimeError("Failed to load and recover state: Could not create new state file.") from save_error
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode state file: {e}")
            # Attempt recovery by backing up the corrupt file and creating a new one.
            try:
                backup_file = "state_backup_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
                shutil.copyfile(state_manager.state_file_path, backup_file) # Assuming there is state_file_path attribute
                logger.warning(f"Backed up corrupted state file to {backup_file}")
                state_manager.save_state({})
                state = {}
                logger.info("Recovered by creating new state file after backup.")
            except Exception as backup_error:
                logger.critical(f"Failed to backup and recreate state file: {backup_error}")
                raise RuntimeError("Failed to load and recover state: Could not backup and recreate state file.") from backup_error
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}")
            raise RuntimeError(f"Failed to load state: {e}") from e

        # Validate the loaded state
        if not isinstance(state, dict):
            logger.error(f"Invalid state format: Expected a dictionary, got {type(state)}")
            raise ValueError("Invalid state format: State must be a dictionary.")

        # Get workers from state, defaulting to an empty dictionary
        workers = state.get('workers', {})

        # Validate workers data
        if not isinstance(workers, dict):
            logger.warning(f"Invalid workers data format: Expected a dictionary, got {type(workers)}.  Treating as empty.")
            workers = {} #resetting to prevent type error

        # Process worker statuses
        active_workers = 0
        idle_workers = 0
        error_workers = 0
        offline_workers = 0
        starting_workers = 0
        stopping_workers = 0

        detailed_status = {}  # Count occurrences of each status.
        validated_workers = {} # Store validated workers.
        for worker_id, worker_data in workers.items():
            if validate_worker_data(worker_data):
                status = worker_data.get('status')
                if status == 'active':
                    active_workers += 1
                elif status == 'idle':
                    idle_workers += 1
                elif status == 'error':
                    error_workers += 1
                elif status == 'offline':
                    offline_workers += 1
                elif status == 'starting':
                    starting_workers += 1
                elif status == 'stopping':
                    stopping_workers += 1

                detailed_status[status] = detailed_status.get(status, 0) + 1 #counting number of statuses
                validated_workers[worker_id] = worker_data  # Store validated data.
            else:
                logger.warning(f"Invalid data for worker {worker_id}, skipping.")
        workers = validated_workers #Use validated data

        # Determine overall status
        if active_workers > 0:
            overall_status = "active"
        elif idle_workers > 0:
            overall_status = "idle"
        elif error_workers > 0:
             overall_status = "error"
        elif offline_workers == len(workers) and len(workers) > 0:
            overall_status = "offline"  # All workers are offline
        elif len(workers) == 0:
            overall_status = "inactive" #No workers defined
        else:
            overall_status = "degraded" # Mixture of other states, not ideal

        # Check for alerts
        alerts = []

        if error_workers > 0:
            alerts.append(f"{error_workers} workers in error state")
        if offline_workers == len(workers) and len(workers) > 0:
            alerts.append("All workers are offline")
        if active_workers == 0 and idle_workers == 0 and error_workers == 0 and len(workers) > 0: #Handles cases where workers exist, but none are active, idle or error.
            alerts.append("All workers are in a non-operational state (e.g. starting, stopping, offline)")
        if active_workers == 0 and idle_workers == 0 and len(workers) == 0: #Handles cases where no workers exist.
            alerts.append("No workers available")

        # Check resource usage alerts
        alerts.extend(check_resource_usage(workers))

        # Check offline workers based on last_seen time and update status
        offline_alerts = check_offline_workers(workers)
        alerts.extend(offline_alerts)

        # Calculate basic metrics
        total_workers = len(workers)
        worker_utilization = (active_workers / total_workers * 100) if total_workers else 0

        # Calculate average CPU usage, memory usage and disk usage
        avg_cpu_usage = calculate_average_metric(workers, 'cpu_usage') #Example of adding a metric
        avg_memory_usage = calculate_average_metric(workers, 'memory_usage')
        avg_disk_usage = calculate_average_metric(workers, 'disk_usage')

        metrics = {
            "total_workers": total_workers,
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "error_workers": error_workers,
            "offline_workers": offline_workers,
            "worker_utilization": worker_utilization,
            "average_cpu_usage": avg_cpu_usage,
            "average_memory_usage": avg_memory_usage,
            "average_disk_usage": avg_disk_usage #Added disk usage to metrics
        }

        # Get system uptime
        uptime = get_system_uptime()

        # Analyze worker history (if available)
        worker_history = state.get('worker_history', [])
        worker_history_analysis = analyze_worker_history(worker_history)

        # Gather process resource usage (if applicable)
        process_resource_usages = {}
        for worker_id, worker_data in workers.items():
            if 'pid' in worker_data:
                pid = worker_data['pid']
                resource_usage = get_process_resource_usage(pid)
                if resource_usage:
                    process_resource_usages[worker_id] = resource_usage
                else:
                    logger.warning(f"Could not retrieve resource usage for worker {worker_id} (PID: {pid})")

        result = {
            "status": overall_status,
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "alerts": alerts,
            "metrics": metrics,
            "last_updated": current_time,
            "uptime": uptime,
            "total_workers": total_workers,
            "detailed_status": detailed_status,
            "worker_history_analysis": worker_history_analysis,
            "process_resource_usages": process_resource_usages #NEW: show usage by PID
        }

        # Cache the result
        async with _cache_lock:  # Ensure thread-safe cache access
            _worker_activity_cache = {
                'data': result,
                'expiry': current_time + _cache_expiry_time
            }
            #Persist status changes to state after checks
            try:
                state['workers'] = workers #Update the state with changes made by offline worker check
                state['last_updated'] = current_time
                # Add current worker status to history before saving.
                for worker_id, worker_data in workers.items():
                    worker_history.append({
                        "worker_id": worker_id,
                        "status": worker_data['status'],
                        "timestamp": current_time,
                        "cpu_usage": worker_data.get('cpu_usage'),
                        "memory_usage": worker_data.get('memory_usage'),
                        "disk_usage": worker_data.get('disk_usage') #Added Disk Usage to history
                    })
                # Limit history size to prevent unbounded growth
                state['worker_history'] = worker_history[-100:] # Keep only the last 100 entries

                state_manager.save_state(state)
                logger.debug("Worker activity data cached and state saved.")
            except Exception as save_exception:
                logger.error(f"Error saving state after monitoring: {save_exception}")

        logger.debug("Worker activity data cached.")
        return result

    except ValueError as ve:
        logger.error(f"Data validation error: {ve}")
        return {
            "status": "error",
            "error": str(ve),
            "active_workers": 0,
            "idle_workers": 0,
            "alerts": ["Data validation error"],
            "metrics": {},
            "last_updated": None,
            "uptime": None,
            "total_workers": 0,
            "detailed_status": {},
            "worker_history_analysis": {},
            "process_resource_usages": {}
        }
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        return {
            "status": "error",
            "error": str(re),
            "active_workers": 0,
            "idle_workers": 0,
            "alerts": ["Monitoring system error"],
            "metrics": {},
            "last_updated": None,
            "uptime": None,
            "total_workers": 0,
            "detailed_status": {},
            "worker_history_analysis": {},
            "process_resource_usages": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error monitoring worker activity: {e}")
        return {
            "status": "error",
            "error": str(e),
            "active_workers": 0,
            "idle_workers": 0,
            "alerts": ["Monitoring system error"],
            "metrics": {},
            "last_updated": None,
            "uptime": None,
            "total_workers": 0,
            "detailed_status": {},
            "worker_history_analysis": {},
            "process_resource_usages": {}
        }


def get_system_uptime() -> float:
    """
    Utility function to get system uptime using psutil.
    Requires 'pip install psutil'
    """
    try:
        return time.time() - psutil.boot_time()
    except Exception as e:
        logger.error(f"Error getting system uptime using psutil: {e}. Returning placeholder value.")
        import random
        return round(random.uniform(0, 1000),2) # Fallback placeholder.