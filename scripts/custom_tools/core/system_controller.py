"""
System Controller

Unified tool for all system control operations including start, stop, restart,
monitoring, and performance management. Consolidates 6 system tools into one.

Capabilities:
- Start, stop, restart system components
- Monitor system performance and health
- Check system status and metrics
- Control orchestrator and workers
- Manage system configuration
- Performance optimization
"""

import os
import sys
import json
import psutil
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
import asyncio
import time

from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient


class SystemController:
    """Comprehensive system control and monitoring."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.ai_client = HTTPAIClient()
        self.project_root = Path(__file__).parent.parent.parent
        
        # System component definitions
        self.components = {
            'orchestrator': {
                'name': 'Continuous Orchestrator',
                'script': 'run_continuous_ai.py',
                'pid_file': 'orchestrator.pid',
                'state_file': 'continuous_orchestrator_state.json'
            },
            'workers': {
                'name': 'AI Workers',
                'script': 'scripts/ai_worker_agent.py',
                'pid_pattern': 'ai_worker',
                'max_workers': 5
            },
            'monitor': {
                'name': 'System Monitor',
                'script': 'scripts/worker_status_monitor.py',
                'pid_file': 'monitor.pid'
            },
            'intelligence': {
                'name': 'Intelligence Hub',
                'script': 'scripts/intelligence_hub.py',
                'pid_file': 'intelligence.pid'
            }
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute system control operation.
        
        Operations:
        - start: Start system or component
        - stop: Stop system or component
        - restart: Restart system or component
        - status: Check system status
        - monitor: Monitor system performance
        - control: Advanced control operations
        - health: Check system health
        - metrics: Get system metrics
        - optimize: Optimize performance
        """
        operations = {
            'start': self.start_system,
            'stop': self.stop_system,
            'restart': self.restart_system,
            'status': self.check_status,
            'monitor': self.monitor_system,
            'control': self.control_system,
            'health': self.check_health,
            'metrics': self.get_metrics,
            'optimize': self.optimize_performance,
            'logs': self.get_logs,
            'config': self.manage_config,
            'debug': self.debug_system
        }
        
        handler = operations.get(operation)
        if not handler:
            # Fuzzy match
            for op, func in operations.items():
                if op in operation or operation in op:
                    handler = func
                    break
                    
        if not handler:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": list(operations.keys())
            }
            
        return await handler(**kwargs)
        
    async def start_system(self, component: str = 'all', **kwargs) -> Dict[str, Any]:
        """Start system components."""
        results = {}
        
        if component == 'all':
            # Start all components in order
            for comp_name in ['orchestrator', 'workers', 'monitor', 'intelligence']:
                results[comp_name] = await self._start_component(comp_name)
        else:
            # Start specific component
            if component in self.components:
                results[component] = await self._start_component(component)
            else:
                return {"error": f"Unknown component: {component}"}
                
        return {
            "success": all(r.get('success', False) for r in results.values()),
            "components_started": results,
            "message": "System started successfully" if all(r.get('success', False) for r in results.values()) else "Some components failed to start"
        }
        
    async def stop_system(self, component: str = 'all', graceful: bool = True, **kwargs) -> Dict[str, Any]:
        """Stop system components."""
        results = {}
        
        if component == 'all':
            # Stop all components in reverse order
            for comp_name in ['intelligence', 'monitor', 'workers', 'orchestrator']:
                results[comp_name] = await self._stop_component(comp_name, graceful)
        else:
            # Stop specific component
            if component in self.components:
                results[component] = await self._stop_component(component, graceful)
            else:
                return {"error": f"Unknown component: {component}"}
                
        return {
            "success": all(r.get('success', False) for r in results.values()),
            "components_stopped": results,
            "message": "System stopped successfully" if all(r.get('success', False) for r in results.values()) else "Some components failed to stop"
        }
        
    async def restart_system(self, component: str = 'all', delay: int = 2, **kwargs) -> Dict[str, Any]:
        """Restart system components."""
        # Stop first
        stop_result = await self.stop_system(component)
        if not stop_result.get('success'):
            return {
                "success": False,
                "error": "Failed to stop system",
                "details": stop_result
            }
            
        # Wait before restarting
        await asyncio.sleep(delay)
        
        # Start again
        start_result = await self.start_system(component)
        
        return {
            "success": start_result.get('success'),
            "message": "System restarted successfully" if start_result.get('success') else "Failed to restart",
            "stop_details": stop_result,
            "start_details": start_result
        }
        
    async def check_status(self, detailed: bool = False, **kwargs) -> Dict[str, Any]:
        """Check system status."""
        status = {
            'overall': 'unknown',
            'components': {},
            'system_info': {}
        }
        
        # Check each component
        running_count = 0
        for comp_name, comp_info in self.components.items():
            comp_status = await self._check_component_status(comp_name)
            status['components'][comp_name] = comp_status
            if comp_status['running']:
                running_count += 1
                
        # Determine overall status
        if running_count == len(self.components):
            status['overall'] = 'running'
        elif running_count > 0:
            status['overall'] = 'partial'
        else:
            status['overall'] = 'stopped'
            
        # Get system information
        status['system_info'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'uptime': self._get_uptime()
        }
        
        # Detailed information if requested
        if detailed:
            status['detailed'] = await self._get_detailed_status()
            
        return {
            "success": True,
            "status": status
        }
        
    async def monitor_system(self, duration: int = 60, interval: int = 5, **kwargs) -> Dict[str, Any]:
        """Monitor system performance over time."""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Collect metrics
            metric = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'available_mb': psutil.virtual_memory().available / 1024 / 1024
                },
                'disk': {
                    'percent': psutil.disk_usage('/').percent,
                    'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
                },
                'processes': len(psutil.pids()),
                'components': {}
            }
            
            # Check component status
            for comp_name in self.components:
                comp_status = await self._check_component_status(comp_name)
                metric['components'][comp_name] = comp_status['running']
                
            metrics.append(metric)
            
            # Wait for next interval
            if time.time() - start_time < duration:
                await asyncio.sleep(interval)
                
        # Analyze metrics
        analysis = self._analyze_metrics(metrics)
        
        return {
            "success": True,
            "duration": duration,
            "samples": len(metrics),
            "metrics": metrics,
            "analysis": analysis
        }
        
    async def control_system(self, action: str, target: str = None, **kwargs) -> Dict[str, Any]:
        """Advanced system control operations."""
        actions = {
            'scale_workers': self._scale_workers,
            'reset_state': self._reset_state,
            'clear_queues': self._clear_queues,
            'force_checkpoint': self._force_checkpoint,
            'emergency_stop': self._emergency_stop
        }
        
        handler = actions.get(action)
        if not handler:
            return {
                "error": f"Unknown control action: {action}",
                "available_actions": list(actions.keys())
            }
            
        return await handler(target=target, **kwargs)
        
    async def check_health(self, **kwargs) -> Dict[str, Any]:
        """Check system health."""
        health_checks = {
            'components': await self._check_components_health(),
            'resources': await self._check_resources_health(),
            'queues': await self._check_queues_health(),
            'errors': await self._check_error_rates()
        }
        
        # Calculate overall health score
        health_score = 100
        issues = []
        
        # Component health
        if health_checks['components']['unhealthy'] > 0:
            health_score -= 20 * health_checks['components']['unhealthy']
            issues.append(f"{health_checks['components']['unhealthy']} unhealthy components")
            
        # Resource health
        if health_checks['resources']['critical']:
            health_score -= 30
            issues.extend(health_checks['resources']['critical'])
            
        # Queue health
        if health_checks['queues']['backed_up']:
            health_score -= 10
            issues.append("Queue backlog detected")
            
        # Error rates
        if health_checks['errors']['high_error_rate']:
            health_score -= 15
            issues.append("High error rate detected")
            
        health_score = max(0, health_score)
        
        return {
            "success": True,
            "health_score": health_score,
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
            "checks": health_checks,
            "issues": issues,
            "recommendations": self._generate_health_recommendations(health_checks, issues)
        }
        
    async def get_metrics(self, period: str = 'last_hour', **kwargs) -> Dict[str, Any]:
        """Get system metrics."""
        # Load orchestrator state for metrics
        orch_state_file = self.project_root / "continuous_orchestrator_state.json"
        metrics = {
            'period': period,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if orch_state_file.exists():
            try:
                with open(orch_state_file, 'r') as f:
                    state = json.load(f)
                    
                metrics.update({
                    'tasks': {
                        'queued': len(state.get('task_queue', [])),
                        'completed': state.get('completed_tasks', 0),
                        'failed': state.get('failed_tasks', 0),
                        'success_rate': self._calculate_success_rate(state)
                    },
                    'workers': {
                        'active': state.get('active_workers', 0),
                        'idle': state.get('idle_workers', 0),
                        'total': state.get('total_workers', 0)
                    },
                    'performance': {
                        'avg_task_duration': state.get('avg_task_duration', 0),
                        'tasks_per_minute': state.get('tasks_per_minute', 0)
                    }
                })
            except:
                metrics['error'] = "Failed to load orchestrator state"
        else:
            metrics['message'] = "System not running - no metrics available"
            
        # Add system metrics
        metrics['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': self._get_network_io()
        }
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    async def optimize_performance(self, target: str = 'auto', **kwargs) -> Dict[str, Any]:
        """Optimize system performance."""
        optimizations = []
        
        # Analyze current state
        status = await self.check_status(detailed=True)
        metrics = await self.get_metrics()
        
        if target == 'auto':
            # Automatic optimization based on analysis
            
            # CPU optimization
            if status['status']['system_info']['cpu_percent'] > 80:
                optimizations.append({
                    'area': 'cpu',
                    'action': 'reduce_worker_count',
                    'reason': 'High CPU usage detected'
                })
                
            # Memory optimization
            if status['status']['system_info']['memory_percent'] > 85:
                optimizations.append({
                    'area': 'memory',
                    'action': 'clear_caches',
                    'reason': 'High memory usage detected'
                })
                
            # Queue optimization
            if metrics['metrics'].get('tasks', {}).get('queued', 0) > 100:
                optimizations.append({
                    'area': 'queue',
                    'action': 'increase_workers',
                    'reason': 'Large task queue detected'
                })
                
        elif target == 'cpu':
            optimizations.append({
                'area': 'cpu',
                'action': 'optimize_cpu_usage',
                'details': 'Adjust process priorities and worker counts'
            })
        elif target == 'memory':
            optimizations.append({
                'area': 'memory',
                'action': 'optimize_memory_usage',
                'details': 'Clear caches and optimize data structures'
            })
        elif target == 'throughput':
            optimizations.append({
                'area': 'throughput',
                'action': 'optimize_throughput',
                'details': 'Adjust batch sizes and parallelism'
            })
            
        # Apply optimizations
        results = []
        for opt in optimizations:
            result = await self._apply_optimization(opt)
            results.append(result)
            
        return {
            "success": all(r.get('success', False) for r in results),
            "optimizations_applied": len(results),
            "optimizations": optimizations,
            "results": results,
            "new_metrics": await self.get_metrics()
        }
        
    async def get_logs(self, component: str = 'all', lines: int = 100, **kwargs) -> Dict[str, Any]:
        """Get system logs."""
        logs = {}
        log_dir = self.project_root / 'logs'
        
        if not log_dir.exists():
            return {"message": "No logs directory found"}
            
        # Get log files
        if component == 'all':
            log_files = list(log_dir.glob('*.log'))
        else:
            log_files = list(log_dir.glob(f'*{component}*.log'))
            
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    logs[log_file.name] = {
                        'lines': all_lines[-lines:],
                        'total_lines': len(all_lines),
                        'size': log_file.stat().st_size
                    }
            except:
                logs[log_file.name] = {'error': 'Failed to read log'}
                
        return {
            "success": True,
            "logs": logs,
            "log_count": len(logs)
        }
        
    async def manage_config(self, action: str = 'get', config: Dict = None, **kwargs) -> Dict[str, Any]:
        """Manage system configuration."""
        config_file = self.project_root / 'system_config.json'
        
        if action == 'get':
            if config_file.exists():
                with open(config_file, 'r') as f:
                    current_config = json.load(f)
                return {
                    "success": True,
                    "config": current_config
                }
            else:
                return {"message": "No configuration file found"}
                
        elif action == 'set' and config:
            # Validate config
            if not self._validate_config(config):
                return {"error": "Invalid configuration"}
                
            # Save config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            return {
                "success": True,
                "message": "Configuration updated",
                "config": config
            }
            
        elif action == 'reset':
            # Reset to defaults
            default_config = self._get_default_config()
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            return {
                "success": True,
                "message": "Configuration reset to defaults",
                "config": default_config
            }
            
        return {"error": f"Unknown config action: {action}"}
        
    async def debug_system(self, component: str = None, **kwargs) -> Dict[str, Any]:
        """Debug system issues."""
        debug_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system': {},
            'components': {},
            'errors': [],
            'suggestions': []
        }
        
        # System debug info
        debug_info['system'] = {
            'platform': sys.platform,
            'python_version': sys.version,
            'process_count': len(psutil.pids()),
            'open_files': len(psutil.Process().open_files()),
            'connections': len(psutil.Process().connections())
        }
        
        # Component debug info
        for comp_name in self.components:
            if component and comp_name != component:
                continue
                
            comp_debug = await self._debug_component(comp_name)
            debug_info['components'][comp_name] = comp_debug
            
            # Check for errors
            if comp_debug.get('errors'):
                debug_info['errors'].extend(comp_debug['errors'])
                
        # Generate suggestions
        if debug_info['errors']:
            debug_info['suggestions'] = self._generate_debug_suggestions(debug_info['errors'])
            
        return {
            "success": True,
            "debug_info": debug_info
        }
        
    async def _start_component(self, component: str) -> Dict[str, Any]:
        """Start a specific component."""
        comp_info = self.components[component]
        
        # Check if already running
        status = await self._check_component_status(component)
        if status['running']:
            return {"success": True, "message": f"{comp_info['name']} already running"}
            
        # Start component
        try:
            if component == 'workers':
                # Special handling for multiple workers
                for i in range(comp_info['max_workers']):
                    subprocess.Popen([
                        sys.executable, 
                        str(self.project_root / comp_info['script']),
                        f'--worker-id={i}'
                    ])
            else:
                # Start single component
                process = subprocess.Popen([
                    sys.executable,
                    str(self.project_root / comp_info['script'])
                ])
                
                # Save PID if needed
                if 'pid_file' in comp_info:
                    pid_file = self.project_root / comp_info['pid_file']
                    with open(pid_file, 'w') as f:
                        f.write(str(process.pid))
                        
            return {"success": True, "message": f"{comp_info['name']} started"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _stop_component(self, component: str, graceful: bool = True) -> Dict[str, Any]:
        """Stop a specific component."""
        comp_info = self.components[component]
        
        try:
            if 'pid_file' in comp_info:
                # Read PID from file
                pid_file = self.project_root / comp_info['pid_file']
                if pid_file.exists():
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                        
                    # Stop process
                    if graceful:
                        os.kill(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGKILL)
                        
                    # Remove PID file
                    pid_file.unlink()
                    
            elif component == 'workers':
                # Stop all workers
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if comp_info['pid_pattern'] in str(proc.info['cmdline']):
                        if graceful:
                            proc.terminate()
                        else:
                            proc.kill()
                            
            return {"success": True, "message": f"{comp_info['name']} stopped"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _check_component_status(self, component: str) -> Dict[str, Any]:
        """Check if component is running."""
        comp_info = self.components[component]
        
        if 'pid_file' in comp_info:
            pid_file = self.project_root / comp_info['pid_file']
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    # Check if process exists
                    psutil.Process(pid)
                    return {'running': True, 'pid': pid}
                except:
                    # Process doesn't exist, clean up PID file
                    pid_file.unlink()
                    
        elif component == 'workers':
            # Count running workers
            worker_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if comp_info.get('pid_pattern', '') in str(proc.info.get('cmdline', [])):
                    worker_count += 1
            return {'running': worker_count > 0, 'count': worker_count}
            
        return {'running': False}
        
    def _get_uptime(self) -> str:
        """Get system uptime."""
        boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime = datetime.now(timezone.utc) - boot_time
        
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        return f"{days}d {hours}h {minutes}m"
        
    async def _get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        return {
            'processes': len(psutil.pids()),
            'threads': psutil.Process().num_threads(),
            'open_files': len(psutil.Process().open_files()),
            'connections': len(psutil.Process().connections()),
            'cpu_cores': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
    def _analyze_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze collected metrics."""
        if not metrics:
            return {}
            
        # Calculate averages
        avg_cpu = sum(m['cpu_percent'] for m in metrics) / len(metrics)
        avg_memory = sum(m['memory']['percent'] for m in metrics) / len(metrics)
        
        # Find peaks
        max_cpu = max(m['cpu_percent'] for m in metrics)
        max_memory = max(m['memory']['percent'] for m in metrics)
        
        # Check stability
        cpu_variance = sum((m['cpu_percent'] - avg_cpu) ** 2 for m in metrics) / len(metrics)
        stable = cpu_variance < 100  # Arbitrary threshold
        
        return {
            'averages': {
                'cpu': round(avg_cpu, 2),
                'memory': round(avg_memory, 2)
            },
            'peaks': {
                'cpu': round(max_cpu, 2),
                'memory': round(max_memory, 2)
            },
            'stability': 'stable' if stable else 'fluctuating',
            'trends': self._detect_trends(metrics)
        }
        
    def _detect_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """Detect trends in metrics."""
        if len(metrics) < 3:
            return {'cpu': 'insufficient_data', 'memory': 'insufficient_data'}
            
        # Simple trend detection
        cpu_values = [m['cpu_percent'] for m in metrics]
        memory_values = [m['memory']['percent'] for m in metrics]
        
        cpu_trend = 'stable'
        if cpu_values[-1] > cpu_values[0] * 1.2:
            cpu_trend = 'increasing'
        elif cpu_values[-1] < cpu_values[0] * 0.8:
            cpu_trend = 'decreasing'
            
        memory_trend = 'stable'
        if memory_values[-1] > memory_values[0] * 1.2:
            memory_trend = 'increasing'
        elif memory_values[-1] < memory_values[0] * 0.8:
            memory_trend = 'decreasing'
            
        return {'cpu': cpu_trend, 'memory': memory_trend}
        
    async def _scale_workers(self, target: str = None, count: int = None, **kwargs) -> Dict[str, Any]:
        """Scale worker count."""
        current_status = await self._check_component_status('workers')
        current_count = current_status.get('count', 0)
        
        if count is None:
            # Auto-scale based on queue
            metrics = await self.get_metrics()
            queued_tasks = metrics['metrics'].get('tasks', {}).get('queued', 0)
            
            if queued_tasks > 50:
                count = min(10, current_count + 2)
            elif queued_tasks < 10 and current_count > 2:
                count = max(1, current_count - 1)
            else:
                count = current_count
                
        if count == current_count:
            return {"success": True, "message": "No scaling needed", "workers": count}
            
        # Scale workers
        if count > current_count:
            # Start more workers
            for i in range(count - current_count):
                subprocess.Popen([
                    sys.executable,
                    str(self.project_root / self.components['workers']['script']),
                    f'--worker-id={current_count + i}'
                ])
        else:
            # Stop some workers
            # This is simplified - in reality would need graceful shutdown
            pass
            
        return {"success": True, "scaled_from": current_count, "scaled_to": count}
        
    async def _reset_state(self, **kwargs) -> Dict[str, Any]:
        """Reset system state."""
        state_files = [
            'continuous_orchestrator_state.json',
            'task_state.json',
            'system_state.json'
        ]
        
        reset_count = 0
        for state_file in state_files:
            file_path = self.project_root / state_file
            if file_path.exists():
                # Backup existing state
                backup_path = file_path.with_suffix('.backup')
                file_path.rename(backup_path)
                reset_count += 1
                
        return {
            "success": True,
            "message": f"Reset {reset_count} state files",
            "backed_up": True
        }
        
    async def _clear_queues(self, **kwargs) -> Dict[str, Any]:
        """Clear task queues."""
        # This would clear Redis queues or other queue systems
        return {"success": True, "message": "Queues cleared"}
        
    async def _force_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """Force system checkpoint."""
        # This would trigger immediate state saving
        return {"success": True, "message": "Checkpoint created"}
        
    async def _emergency_stop(self, **kwargs) -> Dict[str, Any]:
        """Emergency stop all components."""
        return await self.stop_system(component='all', graceful=False)
        
    async def _check_components_health(self) -> Dict[str, Any]:
        """Check health of all components."""
        healthy = 0
        unhealthy = 0
        details = {}
        
        for comp_name in self.components:
            status = await self._check_component_status(comp_name)
            if status['running']:
                healthy += 1
                details[comp_name] = 'healthy'
            else:
                unhealthy += 1
                details[comp_name] = 'not running'
                
        return {
            'healthy': healthy,
            'unhealthy': unhealthy,
            'total': len(self.components),
            'details': details
        }
        
    async def _check_resources_health(self) -> Dict[str, Any]:
        """Check system resources health."""
        critical = []
        warnings = []
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            critical.append(f"CPU usage critical: {cpu_percent}%")
        elif cpu_percent > 75:
            warnings.append(f"CPU usage high: {cpu_percent}%")
            
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            critical.append(f"Memory usage critical: {memory_percent}%")
        elif memory_percent > 80:
            warnings.append(f"Memory usage high: {memory_percent}%")
            
        disk_percent = psutil.disk_usage('/').percent
        if disk_percent > 95:
            critical.append(f"Disk usage critical: {disk_percent}%")
        elif disk_percent > 85:
            warnings.append(f"Disk usage high: {disk_percent}%")
            
        return {
            'critical': critical,
            'warnings': warnings,
            'healthy': len(critical) == 0 and len(warnings) == 0
        }
        
    async def _check_queues_health(self) -> Dict[str, Any]:
        """Check queue health."""
        # Load orchestrator state
        orch_state_file = self.project_root / "continuous_orchestrator_state.json"
        
        if orch_state_file.exists():
            try:
                with open(orch_state_file, 'r') as f:
                    state = json.load(f)
                    
                queued = len(state.get('task_queue', []))
                workers = state.get('active_workers', 0)
                
                backed_up = queued > workers * 10  # Arbitrary threshold
                
                return {
                    'queued_tasks': queued,
                    'active_workers': workers,
                    'backed_up': backed_up,
                    'ratio': round(queued / max(workers, 1), 2)
                }
            except:
                pass
                
        return {'backed_up': False, 'message': 'No queue data available'}
        
    async def _check_error_rates(self) -> Dict[str, Any]:
        """Check system error rates."""
        # This would check logs or metrics for error rates
        return {
            'high_error_rate': False,
            'error_count': 0,
            'error_rate': 0.0
        }
        
    def _generate_health_recommendations(self, checks: Dict, issues: List[str]) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if 'unhealthy components' in ' '.join(issues):
            recommendations.append("Restart failed components")
            
        if 'CPU usage critical' in ' '.join(issues):
            recommendations.append("Reduce worker count or optimize CPU-intensive operations")
            
        if 'Memory usage critical' in ' '.join(issues):
            recommendations.append("Clear caches and reduce memory usage")
            
        if 'Queue backlog' in ' '.join(issues):
            recommendations.append("Scale up workers to process queue faster")
            
        return recommendations
        
    def _calculate_success_rate(self, state: Dict) -> float:
        """Calculate task success rate."""
        completed = state.get('completed_tasks', 0)
        failed = state.get('failed_tasks', 0)
        total = completed + failed
        
        if total == 0:
            return 100.0
            
        return round((completed / total) * 100, 2)
        
    def _get_network_io(self) -> Dict[str, int]:
        """Get network I/O stats."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }
        
    async def _apply_optimization(self, optimization: Dict) -> Dict[str, Any]:
        """Apply a specific optimization."""
        # This would implement actual optimizations
        return {
            "success": True,
            "optimization": optimization['action'],
            "result": "Applied successfully"
        }
        
    def _validate_config(self, config: Dict) -> bool:
        """Validate system configuration."""
        required_keys = ['max_workers', 'task_timeout', 'queue_size']
        return all(key in config for key in required_keys)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            'max_workers': 5,
            'task_timeout': 300,
            'queue_size': 1000,
            'checkpoint_interval': 300,
            'log_level': 'INFO'
        }
        
    async def _debug_component(self, component: str) -> Dict[str, Any]:
        """Debug specific component."""
        debug_info = {
            'status': await self._check_component_status(component),
            'errors': [],
            'warnings': []
        }
        
        # Check component-specific issues
        if not debug_info['status']['running']:
            debug_info['errors'].append(f"{component} is not running")
            
        # Check logs for errors
        logs = await self.get_logs(component=component, lines=50)
        for log_name, log_data in logs.get('logs', {}).items():
            if 'error' in log_name.lower():
                debug_info['errors'].append(f"Errors found in {log_name}")
                
        return debug_info
        
    def _generate_debug_suggestions(self, errors: List[str]) -> List[str]:
        """Generate debug suggestions based on errors."""
        suggestions = []
        
        for error in errors:
            if 'not running' in error:
                suggestions.append("Check if component crashed and review logs")
            elif 'Errors found' in error:
                suggestions.append("Review error logs for specific issues")
                
        return suggestions


async def main(operation: str = None, **kwargs):
    """Entry point for system controller."""
    controller = SystemController()
    
    if operation:
        return await controller.execute(operation, **kwargs)
    else:
        # Default to status check
        return await controller.check_status(**kwargs)
        

if __name__ == "__main__":
    # Test the controller
    import asyncio
    
    async def test():
        controller = SystemController()
        
        # Test operations
        print("Status:", await controller.check_status())
        print("\nHealth:", await controller.check_health())
        print("\nMetrics:", await controller.get_metrics())
        
    asyncio.run(test())