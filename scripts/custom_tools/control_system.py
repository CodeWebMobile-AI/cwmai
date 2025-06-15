"""
AI-Generated Tool: control_system
Description: Control the CWMAI system using natural language commands
Generated: 2025-06-15
Requirements: Interpret natural language commands like "start the system", "stop everything", etc.
"""

import asyncio
from typing import Dict, Any
import re

# Import our system control tools
from scripts.custom_tools.start_system import start_system
from scripts.custom_tools.stop_system import stop_system
from scripts.custom_tools.restart_system import restart_system
from scripts.custom_tools.check_system_status import check_system_status

__description__ = "Control the CWMAI system using natural language commands"
__parameters__ = {
    "command": {
        "type": "string",
        "description": "Natural language command like 'start the system', 'stop everything', 'check status'",
        "required": True
    }
}
__examples__ = [
    {"description": "Start the system", "code": "await control_system(command='start the system')"},
    {"description": "Stop everything", "code": "await control_system(command='stop the whole system')"},
    {"description": "Check health", "code": "await control_system(command='how is the system doing?')"},
    {"description": "Restart", "code": "await control_system(command='restart everything')"}
]


async def control_system(**kwargs) -> Dict[str, Any]:
    """Control the CWMAI system using natural language commands."""
    command = kwargs.get('command', '').lower()
    
    if not command:
        return {
            "error": "No command provided",
            "suggestion": "Try 'start the system', 'stop everything', or 'check status'",
            "summary": "No command given"
        }
    
    try:
        # Parse the command to determine action
        action = parse_command(command)
        
        if action['type'] == 'start':
            # Handle start commands
            result = await start_system(mode=action.get('mode', 'full'))
            return {
                "action": "start",
                "success": result['status'] in ['success', 'partial'],
                "result": result,
                "summary": result.get('summary', 'Started system')
            }
            
        elif action['type'] == 'stop':
            # Handle stop commands
            force = 'force' in command or 'kill' in command
            result = await stop_system(force=force)
            return {
                "action": "stop",
                "success": result['status'] in ['success', 'partial'],
                "result": result,
                "summary": result.get('summary', 'Stopped system')
            }
            
        elif action['type'] == 'restart':
            # Handle restart commands
            force = 'force' in command
            result = await restart_system(force=force, mode=action.get('mode', 'full'))
            return {
                "action": "restart",
                "success": result['status'] in ['success', 'partial'],
                "result": result,
                "summary": result.get('summary', 'Restarted system')
            }
            
        elif action['type'] == 'status':
            # Handle status/check commands
            verbose = 'detail' in command or 'verbose' in command
            result = await check_system_status(verbose=verbose)
            
            # Create a friendly summary
            health = result.get('health', {})
            status = health.get('status', 'unknown')
            
            if status == 'healthy':
                friendly_status = "The system is running well! 🟢"
            elif status == 'degraded':
                friendly_status = "The system is running but has some issues 🟡"
            elif status == 'unhealthy':
                friendly_status = "The system has problems and needs attention 🔴"
            else:
                friendly_status = "Unable to determine system status"
            
            return {
                "action": "status",
                "success": True,
                "result": result,
                "friendly_status": friendly_status,
                "summary": result.get('summary', 'Checked system status')
            }
            
        else:
            # Command not recognized
            return {
                "action": "unknown",
                "success": False,
                "error": f"I don't understand the command: '{command}'",
                "suggestions": [
                    "start the system",
                    "stop everything",
                    "restart the system",
                    "check system status",
                    "how is the system doing?"
                ],
                "summary": "Command not recognized"
            }
            
    except Exception as e:
        return {
            "action": "error",
            "success": False,
            "error": f"Error executing command: {str(e)}",
            "summary": "Command failed with error"
        }


def parse_command(command: str) -> Dict[str, Any]:
    """Parse natural language command to determine action and parameters."""
    
    # Start patterns
    start_patterns = [
        r'start',
        r'launch',
        r'run',
        r'initialize',
        r'begin',
        r'activate',
        r'turn on',
        r'boot'
    ]
    
    # Stop patterns
    stop_patterns = [
        r'stop',
        r'halt',
        r'shutdown',
        r'terminate',
        r'end',
        r'quit',
        r'turn off',
        r'deactivate'
    ]
    
    # Restart patterns
    restart_patterns = [
        r'restart',
        r'reboot',
        r'reload',
        r'refresh',
        r'cycle'
    ]
    
    # Status patterns
    status_patterns = [
        r'status',
        r'check',
        r'health',
        r'how.*doing',
        r'running',
        r'state',
        r'diagnose',
        r'monitor'
    ]
    
    # Check for action type
    for pattern in start_patterns:
        if re.search(pattern, command):
            # Determine mode
            if 'minimal' in command or 'basic' in command:
                mode = 'minimal'
            elif 'worker' in command:
                mode = 'workers'
            else:
                mode = 'full'
            return {'type': 'start', 'mode': mode}
    
    for pattern in stop_patterns:
        if re.search(pattern, command):
            return {'type': 'stop'}
    
    for pattern in restart_patterns:
        if re.search(pattern, command):
            # Determine mode
            if 'minimal' in command or 'basic' in command:
                mode = 'minimal'
            elif 'worker' in command:
                mode = 'workers'
            else:
                mode = 'full'
            return {'type': 'restart', 'mode': mode}
    
    for pattern in status_patterns:
        if re.search(pattern, command):
            return {'type': 'status'}
    
    return {'type': 'unknown'}