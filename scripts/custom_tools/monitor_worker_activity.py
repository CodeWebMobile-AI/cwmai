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
from typing import Dict, Any

import asyncio
import logging

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
    """
    try:
        # Initialize required components
        state_manager = StateManager()
        
        # Load current state
        state = state_manager.load_state()
        
        # Get workers from state
        workers = state.get('workers', {})
        
        # Process worker statuses
        active_workers = sum(1 for w in workers.values() if isinstance(w, dict) and w.get('status') == 'active')
        idle_workers = sum(1 for w in workers.values() if isinstance(w, dict) and w.get('status') == 'idle')
        
        # Determine overall status
        if active_workers == 0 and idle_workers > 0:
            overall_status = "idle"
        elif active_workers > 0:
            overall_status = "active"
        else:
            overall_status = "inactive"
        
        # Check for alerts
        alerts = []
        error_workers = sum(1 for w in workers.values() if isinstance(w, dict) and w.get('status') == 'error')
        
        if error_workers > 0:
            alerts.append(f"{error_workers} workers in error state")
        if active_workers == 0 and idle_workers == 0:
            alerts.append("No workers available")
        
        # Calculate basic metrics
        metrics = {
            "total_workers": len(workers),
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "error_workers": error_workers,
            "worker_utilization": (active_workers / len(workers) * 100) if workers else 0
        }
        
        return {
            "status": overall_status,
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "alerts": alerts,
            "metrics": metrics,
            "last_updated": state.get('last_updated', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Failed to monitor worker activity: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "active_workers": 0,
            "idle_workers": 0,
            "alerts": ["Monitoring system error"],
            "metrics": {},
            "last_updated": None
        }
