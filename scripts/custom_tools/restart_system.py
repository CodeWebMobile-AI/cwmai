"""
AI-Generated Tool: restart_system
Description: Restart the CWMAI system (stop all components and start them again)
Generated: 2025-06-15
Requirements: Gracefully restart all system components with minimal downtime
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

# Import our other tools
from scripts.custom_tools.stop_system import stop_system
from scripts.custom_tools.start_system import start_system
from scripts.custom_tools.check_system_status import check_system_status

__description__ = "Restart the CWMAI system (stop and start all components)"
__parameters__ = {
    "mode": {
        "type": "string",
        "description": "Restart mode: 'full', 'minimal', 'workers'",
        "required": False,
        "default": "full"
    },
    "wait_time": {
        "type": "integer",
        "description": "Seconds to wait between stop and start",
        "required": False,
        "default": 3
    },
    "force": {
        "type": "boolean",
        "description": "Force stop if graceful shutdown fails",
        "required": False,
        "default": False
    }
}
__examples__ = [
    {"description": "Restart full system", "code": "await restart_system()"},
    {"description": "Restart with force", "code": "await restart_system(force=True)"},
    {"description": "Quick restart", "code": "await restart_system(wait_time=1)"}
]


async def restart_system(**kwargs) -> Dict[str, Any]:
    """Restart the CWMAI system."""
    mode = kwargs.get('mode', 'full')
    wait_time = kwargs.get('wait_time', 3)
    force = kwargs.get('force', False)
    
    restart_log = []
    start_time = datetime.now()
    
    try:
        # Step 1: Check initial status
        initial_status = await check_system_status()
        running_before = len([c for c in initial_status.get('components', []) if c['status'] == 'running'])
        restart_log.append({
            "step": "initial_check",
            "timestamp": datetime.now().isoformat(),
            "running_components": running_before
        })
        
        # Step 2: Stop the system
        stop_result = await stop_system(force=force)
        restart_log.append({
            "step": "stop_system",
            "timestamp": datetime.now().isoformat(),
            "status": stop_result['status'],
            "stopped": len(stop_result.get('stopped', [])),
            "failed": len(stop_result.get('failed', []))
        })
        
        if stop_result['status'] == 'failed':
            return {
                "status": "failed",
                "message": "Failed to stop system for restart",
                "stop_result": stop_result,
                "restart_log": restart_log,
                "summary": "Restart aborted - could not stop system"
            }
        
        # Step 3: Wait before starting
        if wait_time > 0:
            restart_log.append({
                "step": "waiting",
                "timestamp": datetime.now().isoformat(),
                "wait_seconds": wait_time
            })
            await asyncio.sleep(wait_time)
        
        # Step 4: Start the system
        start_result = await start_system(mode=mode)
        restart_log.append({
            "step": "start_system",
            "timestamp": datetime.now().isoformat(),
            "status": start_result['status'],
            "started": len(start_result.get('started', [])),
            "failed": len(start_result.get('failed', []))
        })
        
        # Step 5: Verify restart
        await asyncio.sleep(2)  # Give components time to fully start
        final_status = await check_system_status()
        running_after = len([c for c in final_status.get('components', []) if c['status'] == 'running'])
        restart_log.append({
            "step": "final_check",
            "timestamp": datetime.now().isoformat(),
            "running_components": running_after,
            "health": final_status.get('health', {}).get('status', 'unknown')
        })
        
        # Calculate restart duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Determine overall success
        if start_result['status'] == 'success' and running_after >= running_before:
            status = "success"
            message = f"System restarted successfully in {duration:.1f} seconds"
        elif start_result['status'] == 'partial' or running_after > 0:
            status = "partial"
            message = f"System partially restarted ({running_after}/{running_before} components running)"
        else:
            status = "failed"
            message = "System restart failed"
        
        return {
            "status": status,
            "message": message,
            "mode": mode,
            "duration_seconds": duration,
            "components_before": running_before,
            "components_after": running_after,
            "stop_result": stop_result,
            "start_result": start_result,
            "restart_log": restart_log,
            "health": final_status.get('health', {}),
            "summary": f"Restarted in {duration:.1f}s: {running_after} components running"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Restart error: {str(e)}",
            "restart_log": restart_log,
            "summary": "Restart failed with error"
        }