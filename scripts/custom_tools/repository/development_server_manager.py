"""
AI-Generated Tool: development_server_manager
Description: Manage development servers for local repositories
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import subprocess
import os
import signal
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil

__description__ = "Start, stop, and manage development servers for repositories"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["start", "stop", "restart", "status", "list", "logs"],
        "description": "Server operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "port": {
        "type": "integer",
        "required": False,
        "description": "Port to run server on (auto-assigned if not specified)"
    },
    "env": {
        "type": "string",
        "required": False,
        "default": "development",
        "description": "Environment to run in"
    }
}
__examples__ = [
    "development_server_manager(operation='start', repo_name='my-app')",
    "development_server_manager(operation='start', repo_name='my-app', port=3000)",
    "development_server_manager(operation='stop', repo_name='my-app')",
    "development_server_manager(operation='status', repo_name='my-app')",
    "development_server_manager(operation='list', repo_name='all')"
]

# Base paths
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"
SERVERS_STATE_FILE = Path(__file__).parent.parent.parent.parent / "local_repos" / ".dev_servers.json"

# Global server state
RUNNING_SERVERS = {}

async def development_server_manager(**kwargs) -> Dict[str, Any]:
    """
    Manage development servers for repositories.
    """
    operation = kwargs.get('operation')
    repo_name = kwargs.get('repo_name')
    
    if not operation:
        return {
            "status": "error",
            "message": "operation parameter is required"
        }
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name parameter is required"
        }
    
    # Load server state
    await _load_server_state()
    
    # Route to appropriate operation
    operations = {
        'start': start_server,
        'stop': stop_server,
        'restart': restart_server,
        'status': get_server_status,
        'list': list_servers,
        'logs': get_server_logs
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def start_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Start a development server."""
    repo_name = params['repo_name']
    port = params.get('port')
    env = params.get('env', 'development')
    
    # Check if repository exists
    repo_path = REPOSITORIES_PATH / repo_name
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Check if server is already running
    if repo_name in RUNNING_SERVERS:
        server_info = RUNNING_SERVERS[repo_name]
        if await _is_process_running(server_info['pid']):
            return {
                "status": "error",
                "message": f"Server already running on port {server_info['port']}",
                "server_info": server_info
            }
    
    # Detect project type and get start command
    project_type = await _detect_project_type(repo_path)
    if not project_type:
        return {
            "status": "error",
            "message": "Unable to detect project type"
        }
    
    # Auto-assign port if not specified
    if not port:
        port = await _find_available_port()
    
    # Get start command
    start_cmd = await _get_start_command(repo_path, project_type, port)
    if not start_cmd:
        return {
            "status": "error",
            "message": "No start script found for this project type"
        }
    
    # Set up environment
    process_env = os.environ.copy()
    process_env['NODE_ENV'] = env
    process_env['PORT'] = str(port)
    
    # For Laravel projects
    if project_type == 'laravel':
        process_env['APP_ENV'] = env
    
    try:
        # Start the server process
        process = subprocess.Popen(
            start_cmd,
            cwd=repo_path,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        # Wait a moment to check if it started successfully
        await asyncio.sleep(2)
        
        if process.poll() is not None:
            # Process died immediately
            stderr = process.stderr.read().decode() if process.stderr else ''
            return {
                "status": "error",
                "message": "Server failed to start",
                "error": stderr
            }
        
        # Store server info
        server_info = {
            'pid': process.pid,
            'port': port,
            'started_at': datetime.now().isoformat(),
            'project_type': project_type,
            'command': start_cmd,
            'env': env
        }
        
        RUNNING_SERVERS[repo_name] = server_info
        await _save_server_state()
        
        return {
            "status": "success",
            "message": f"Development server started on port {port}",
            "server_info": {
                "url": f"http://localhost:{port}",
                "port": port,
                "pid": process.pid,
                "project_type": project_type
            },
            "next_steps": [
                f"Open http://localhost:{port} in your browser",
                f"Use 'development_server_manager(operation=\"stop\", repo_name=\"{repo_name}\")' to stop",
                f"Use 'development_server_manager(operation=\"logs\", repo_name=\"{repo_name}\")' to view logs"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start server: {str(e)}"
        }

async def stop_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Stop a development server."""
    repo_name = params['repo_name']
    
    if repo_name not in RUNNING_SERVERS:
        return {
            "status": "error",
            "message": f"No server running for repository '{repo_name}'"
        }
    
    server_info = RUNNING_SERVERS[repo_name]
    pid = server_info['pid']
    
    try:
        if await _is_process_running(pid):
            # Try graceful shutdown first
            if os.name == 'nt':
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            
            # Wait for process to die
            await asyncio.sleep(2)
            
            # Force kill if still running
            if await _is_process_running(pid):
                if os.name == 'nt':
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
        
        # Remove from running servers
        del RUNNING_SERVERS[repo_name]
        await _save_server_state()
        
        return {
            "status": "success",
            "message": f"Server stopped for repository '{repo_name}'",
            "port_freed": server_info['port']
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to stop server: {str(e)}"
        }

async def restart_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Restart a development server."""
    repo_name = params['repo_name']
    
    # Get current server info if running
    current_port = None
    if repo_name in RUNNING_SERVERS:
        current_port = RUNNING_SERVERS[repo_name]['port']
        
        # Stop the server
        stop_result = await stop_server(params)
        if stop_result['status'] != 'success':
            return stop_result
        
        # Wait a moment
        await asyncio.sleep(1)
    
    # Start with same port if it was running
    if current_port:
        params['port'] = current_port
    
    return await start_server(params)

async def get_server_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get status of a development server."""
    repo_name = params['repo_name']
    
    if repo_name not in RUNNING_SERVERS:
        return {
            "status": "info",
            "message": f"No server running for repository '{repo_name}'",
            "running": False
        }
    
    server_info = RUNNING_SERVERS[repo_name]
    pid = server_info['pid']
    
    # Check if process is actually running
    is_running = await _is_process_running(pid)
    
    if not is_running:
        # Clean up stale entry
        del RUNNING_SERVERS[repo_name]
        await _save_server_state()
        
        return {
            "status": "info",
            "message": "Server was running but has stopped",
            "running": False,
            "last_known_info": server_info
        }
    
    # Get process info
    try:
        process = psutil.Process(pid)
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        
        uptime = datetime.now() - datetime.fromisoformat(server_info['started_at'])
        
        return {
            "status": "success",
            "running": True,
            "server_info": {
                "url": f"http://localhost:{server_info['port']}",
                "port": server_info['port'],
                "pid": pid,
                "uptime_seconds": int(uptime.total_seconds()),
                "cpu_percent": cpu_percent,
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "started_at": server_info['started_at']
            }
        }
    except:
        return {
            "status": "success",
            "running": True,
            "server_info": server_info
        }

async def list_servers(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all running development servers."""
    servers = []
    stale_servers = []
    
    for repo_name, server_info in list(RUNNING_SERVERS.items()):
        pid = server_info['pid']
        
        if await _is_process_running(pid):
            servers.append({
                "repository": repo_name,
                "port": server_info['port'],
                "url": f"http://localhost:{server_info['port']}",
                "pid": pid,
                "started_at": server_info['started_at'],
                "project_type": server_info.get('project_type', 'unknown')
            })
        else:
            stale_servers.append(repo_name)
    
    # Clean up stale entries
    for repo_name in stale_servers:
        del RUNNING_SERVERS[repo_name]
    
    if stale_servers:
        await _save_server_state()
    
    return {
        "status": "success",
        "total_running": len(servers),
        "servers": servers,
        "cleaned_stale": len(stale_servers)
    }

async def get_server_logs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get logs from a development server."""
    repo_name = params['repo_name']
    
    return {
        "status": "info",
        "message": "Log retrieval requires process output capture implementation",
        "suggestion": "Check terminal where server was started or use PM2 for process management"
    }

async def _detect_project_type(repo_path: Path) -> Optional[str]:
    """Detect the type of project."""
    if (repo_path / 'artisan').exists():
        return 'laravel'
    
    if (repo_path / 'package.json').exists():
        package_data = json.loads((repo_path / 'package.json').read_text())
        deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
        
        if 'next' in deps:
            return 'nextjs'
        elif 'react-scripts' in deps:
            return 'create-react-app'
        elif 'vite' in deps or '@vitejs/plugin-react' in deps:
            return 'vite'
        elif 'express' in deps:
            return 'express'
        else:
            return 'node'
    
    return None

async def _get_start_command(repo_path: Path, project_type: str, port: int) -> Optional[str]:
    """Get the start command for the project type."""
    # Check package.json for scripts first
    if (repo_path / 'package.json').exists():
        package_data = json.loads((repo_path / 'package.json').read_text())
        scripts = package_data.get('scripts', {})
        
        # Look for common dev scripts
        for script_name in ['dev', 'develop', 'start', 'serve']:
            if script_name in scripts:
                return f"npm run {script_name}"
    
    # Fallback based on project type
    if project_type == 'laravel':
        return f"php artisan serve --port={port}"
    elif project_type == 'nextjs':
        return "npm run dev"
    elif project_type == 'create-react-app':
        return "npm start"
    elif project_type == 'vite':
        return "npm run dev"
    elif project_type == 'express':
        # Check for main file
        if (repo_path / 'server.js').exists():
            return "node server.js"
        elif (repo_path / 'app.js').exists():
            return "node app.js"
        elif (repo_path / 'index.js').exists():
            return "node index.js"
    
    return None

async def _find_available_port() -> int:
    """Find an available port starting from 3000."""
    import socket
    
    # Common dev ports to try
    ports_to_try = [3000, 3001, 3002, 4000, 5000, 5173, 8000, 8080]
    
    # Add ports already in use by our servers
    used_ports = [info['port'] for info in RUNNING_SERVERS.values()]
    
    for port in ports_to_try:
        if port in used_ports:
            continue
            
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except:
            continue
    
    # If all common ports are taken, find a random one
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

async def _is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

async def _load_server_state():
    """Load server state from file."""
    global RUNNING_SERVERS
    
    if SERVERS_STATE_FILE.exists():
        try:
            with open(SERVERS_STATE_FILE, 'r') as f:
                RUNNING_SERVERS = json.load(f)
        except:
            RUNNING_SERVERS = {}
    else:
        RUNNING_SERVERS = {}

async def _save_server_state():
    """Save server state to file."""
    SERVERS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(SERVERS_STATE_FILE, 'w') as f:
        json.dump(RUNNING_SERVERS, f, indent=2)

# Test function
async def main():
    # Test server operations
    result = await development_server_manager(
        operation='list',
        repo_name='all'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())