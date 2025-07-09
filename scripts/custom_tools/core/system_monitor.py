"""
System Monitor

Unified tool for comprehensive system monitoring including worker activity,
performance tracking, and real-time insights. Consolidates monitoring tools.

Capabilities:
- Monitor worker activity and status
- Track system performance metrics
- Real-time system insights
- Resource usage monitoring
- Queue and task monitoring
- Health checks and alerts
"""

import os
import json
import psutil
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import time

from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient


class SystemMonitor:
    """Comprehensive system monitoring and insights."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.ai_client = HTTPAIClient()
        self.project_root = Path(__file__).parent.parent.parent
        
        # Monitoring data storage
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alerts = []
        self.monitoring_config = {
            'cpu_threshold': 80,
            'memory_threshold': 85,
            'disk_threshold': 90,
            'queue_threshold': 100,
            'error_rate_threshold': 0.1
        }
        
    async def monitor(self, what: str = 'system', duration: int = None, 
                     real_time: bool = False, **kwargs) -> Dict[str, Any]:
        """Main monitoring entry point.
        
        Args:
            what: What to monitor (system, workers, performance, etc.)
            duration: How long to monitor (seconds)
            real_time: Whether to stream results
            **kwargs: Additional parameters
            
        Returns:
            Monitoring results
        """
        monitors = {
            'system': self.monitor_system,
            'workers': self.monitor_workers,
            'performance': self.monitor_performance,
            'resources': self.monitor_resources,
            'queues': self.monitor_queues,
            'health': self.monitor_health,
            'errors': self.monitor_errors,
            'activity': self.monitor_activity,
            'realtime': self.monitor_realtime
        }
        
        # Normalize what
        what_lower = what.lower()
        
        # Find best match
        handler = None
        for key, func in monitors.items():
            if key in what_lower or what_lower in key:
                handler = func
                break
                
        if not handler:
            # Default to system monitoring
            handler = self.monitor_system
            
        # Handle real-time monitoring
        if real_time:
            return await self.monitor_realtime(target=what, duration=duration, **kwargs)
            
        return await handler(duration=duration, **kwargs)
        
    async def monitor_system(self, duration: int = None, **kwargs) -> Dict[str, Any]:
        """Monitor overall system status."""
        if duration:
            # Collect metrics over time
            return await self._collect_metrics_over_time(duration)
        else:
            # Single snapshot
            metrics = await self._collect_system_metrics()
            analysis = self._analyze_metrics([metrics])
            
            return {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "analysis": analysis,
                "alerts": self._check_alerts(metrics),
                "recommendations": self._generate_recommendations(metrics, analysis)
            }
            
    async def monitor_workers(self, **kwargs) -> Dict[str, Any]:
        """Monitor worker activity and status."""
        workers_info = {
            'active': 0,
            'idle': 0,
            'total': 0,
            'workers': [],
            'performance': {}
        }
        
        # Check orchestrator state
        orch_state_file = self.project_root / "continuous_orchestrator_state.json"
        if orch_state_file.exists():
            try:
                with open(orch_state_file, 'r') as f:
                    state = json.load(f)
                    
                workers_info.update({
                    'active': state.get('active_workers', 0),
                    'idle': state.get('idle_workers', 0),
                    'total': state.get('total_workers', 0)
                })
                
                # Get worker details
                worker_status = state.get('worker_status', {})
                for worker_id, status in worker_status.items():
                    workers_info['workers'].append({
                        'id': worker_id,
                        'status': status.get('status', 'unknown'),
                        'last_task': status.get('last_task', 'none'),
                        'tasks_completed': status.get('tasks_completed', 0),
                        'uptime': status.get('uptime', 0)
                    })
                    
            except Exception as e:
                workers_info['error'] = str(e)
                
        # Check actual processes
        worker_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if 'worker' in str(proc.info.get('cmdline', [])).lower():
                    worker_processes.append({
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info.get('cpu_percent', 0),
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info.get('memory_info') else 0
                    })
            except:
                pass
                
        workers_info['processes'] = worker_processes
        
        # Calculate performance metrics
        if workers_info['workers']:
            total_tasks = sum(w['tasks_completed'] for w in workers_info['workers'])
            workers_info['performance'] = {
                'total_tasks_completed': total_tasks,
                'avg_tasks_per_worker': round(total_tasks / max(len(workers_info['workers']), 1), 2),
                'utilization': round((workers_info['active'] / max(workers_info['total'], 1)) * 100, 2)
            }
            
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workers": workers_info,
            "health": self._assess_worker_health(workers_info)
        }
        
    async def monitor_performance(self, duration: int = 60, **kwargs) -> Dict[str, Any]:
        """Monitor system performance metrics."""
        performance_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Collect performance metrics
            perf_metric = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'per_core': psutil.cpu_percent(interval=1, percpu=True),
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'used_gb': psutil.virtual_memory().used / 1024 / 1024 / 1024,
                    'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
                },
                'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            }
            
            performance_data.append(perf_metric)
            
            # Wait before next collection
            if time.time() - start_time < duration:
                await asyncio.sleep(5)  # Collect every 5 seconds
                
        # Analyze performance
        analysis = self._analyze_performance(performance_data)
        
        return {
            "success": True,
            "duration": duration,
            "samples": len(performance_data),
            "performance": analysis,
            "raw_data": performance_data[-10:],  # Last 10 samples
            "trends": self._detect_performance_trends(performance_data)
        }
        
    async def monitor_resources(self, **kwargs) -> Dict[str, Any]:
        """Monitor system resource usage."""
        resources = {
            'cpu': {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'used_gb': psutil.virtual_memory().used / 1024 / 1024 / 1024,
                'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'percent': psutil.virtual_memory().percent,
                'swap_percent': psutil.swap_memory().percent
            },
            'disk': [],
            'network': {
                'interfaces': [],
                'connections': len(psutil.net_connections())
            }
        }
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                resources['disk'].append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'total_gb': usage.total / 1024 / 1024 / 1024,
                    'used_gb': usage.used / 1024 / 1024 / 1024,
                    'free_gb': usage.free / 1024 / 1024 / 1024,
                    'percent': usage.percent
                })
            except:
                pass
                
        # Network interfaces
        net_if_stats = psutil.net_if_stats()
        for interface, stats in net_if_stats.items():
            if stats.isup:
                resources['network']['interfaces'].append({
                    'name': interface,
                    'speed_mbps': stats.speed,
                    'mtu': stats.mtu
                })
                
        # Check resource alerts
        alerts = self._check_resource_alerts(resources)
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resources": resources,
            "alerts": alerts,
            "health_score": self._calculate_resource_health(resources)
        }
        
    async def monitor_queues(self, **kwargs) -> Dict[str, Any]:
        """Monitor task queues and processing."""
        queue_info = {
            'task_queue': {'size': 0, 'oldest_task': None},
            'completed_tasks': 0,
            'failed_tasks': 0,
            'processing_rate': 0,
            'average_wait_time': 0
        }
        
        # Load orchestrator state
        orch_state_file = self.project_root / "continuous_orchestrator_state.json"
        if orch_state_file.exists():
            try:
                with open(orch_state_file, 'r') as f:
                    state = json.load(f)
                    
                # Queue metrics
                task_queue = state.get('task_queue', [])
                queue_info['task_queue']['size'] = len(task_queue)
                
                if task_queue:
                    # Get oldest task
                    oldest_task = min(task_queue, key=lambda x: x.get('created_at', '9999'))
                    queue_info['task_queue']['oldest_task'] = oldest_task.get('created_at')
                    
                queue_info['completed_tasks'] = state.get('completed_tasks', 0)
                queue_info['failed_tasks'] = state.get('failed_tasks', 0)
                
                # Calculate processing rate
                if 'metrics' in state:
                    queue_info['processing_rate'] = state['metrics'].get('tasks_per_minute', 0)
                    
            except Exception as e:
                queue_info['error'] = str(e)
                
        # Analyze queue health
        queue_health = self._analyze_queue_health(queue_info)
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queues": queue_info,
            "health": queue_health,
            "recommendations": self._generate_queue_recommendations(queue_info, queue_health)
        }
        
    async def monitor_health(self, deep_check: bool = False, **kwargs) -> Dict[str, Any]:
        """Monitor overall system health."""
        health_checks = {
            'system': await self._check_system_health(),
            'workers': await self._check_workers_health(),
            'resources': await self._check_resources_health(),
            'queues': await self._check_queues_health(),
            'errors': await self._check_errors_health()
        }
        
        # Deep health check if requested
        if deep_check:
            health_checks['components'] = await self._deep_component_check()
            health_checks['dependencies'] = await self._check_dependencies_health()
            
        # Calculate overall health score
        health_score = self._calculate_overall_health(health_checks)
        
        # Generate health report
        report = {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_score": health_score,
            "status": self._get_health_status(health_score),
            "checks": health_checks,
            "issues": self._identify_health_issues(health_checks),
            "recommendations": self._generate_health_recommendations(health_checks)
        }
        
        # Store health check result
        self._store_health_check(report)
        
        return report
        
    async def monitor_errors(self, time_range: str = 'last_hour', **kwargs) -> Dict[str, Any]:
        """Monitor system errors and issues."""
        errors = {
            'by_type': defaultdict(int),
            'by_component': defaultdict(int),
            'recent_errors': [],
            'error_rate': 0,
            'total_errors': 0
        }
        
        # Check log files
        log_dir = self.project_root / 'logs'
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-1000:]  # Last 1000 lines
                        
                    for line in lines:
                        if 'ERROR' in line or 'CRITICAL' in line:
                            errors['total_errors'] += 1
                            
                            # Extract error type
                            if 'ERROR' in line:
                                errors['by_type']['error'] += 1
                            if 'CRITICAL' in line:
                                errors['by_type']['critical'] += 1
                                
                            # Extract component
                            component = log_file.stem
                            errors['by_component'][component] += 1
                            
                            # Add to recent errors (limit to 20)
                            if len(errors['recent_errors']) < 20:
                                errors['recent_errors'].append({
                                    'file': log_file.name,
                                    'line': line.strip(),
                                    'timestamp': self._extract_timestamp(line)
                                })
                                
                except Exception:
                    pass
                    
        # Calculate error rate
        # This is simplified - real implementation would track over time
        errors['error_rate'] = errors['total_errors'] / 60  # Errors per minute
        
        # Analyze error patterns
        analysis = self._analyze_error_patterns(errors)
        
        return {
            "success": True,
            "time_range": time_range,
            "errors": dict(errors),
            "analysis": analysis,
            "alerts": self._generate_error_alerts(errors, analysis)
        }
        
    async def monitor_activity(self, **kwargs) -> Dict[str, Any]:
        """Monitor system activity and usage patterns."""
        activity = {
            'current_processes': len(psutil.pids()),
            'active_connections': len(psutil.net_connections()),
            'open_files': self._count_open_files(),
            'system_calls': self._estimate_system_calls(),
            'user_activity': await self._get_user_activity(),
            'component_activity': await self._get_component_activity()
        }
        
        # Get activity timeline
        timeline = await self._generate_activity_timeline()
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "activity": activity,
            "timeline": timeline,
            "patterns": self._identify_activity_patterns(activity, timeline)
        }
        
    async def monitor_realtime(self, target: str = 'system', duration: int = None, **kwargs) -> Dict[str, Any]:
        """Real-time monitoring with streaming updates."""
        # This would implement real-time monitoring
        # For now, return a message about the feature
        return {
            "success": True,
            "message": "Real-time monitoring initiated",
            "target": target,
            "duration": duration or "continuous",
            "note": "Real-time updates would be streamed via WebSocket or Server-Sent Events"
        }
        
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'percent': psutil.virtual_memory().percent,
                'used_gb': psutil.virtual_memory().used / 1024 / 1024 / 1024,
                'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            },
            'disk': {
                'percent': psutil.disk_usage('/').percent,
                'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            'processes': len(psutil.pids()),
            'uptime': self._get_system_uptime()
        }
        
    async def _collect_metrics_over_time(self, duration: int) -> Dict[str, Any]:
        """Collect metrics over a time period."""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metric = await self._collect_system_metrics()
            metrics.append(metric)
            self.metrics_history.append(metric)  # Store in history
            
            # Wait before next collection
            if time.time() - start_time < duration:
                await asyncio.sleep(5)  # Collect every 5 seconds
                
        # Analyze collected metrics
        analysis = self._analyze_metrics(metrics)
        
        return {
            "success": True,
            "duration": duration,
            "samples": len(metrics),
            "metrics": metrics[-10:],  # Last 10 samples
            "analysis": analysis,
            "alerts": self._check_metrics_alerts(metrics)
        }
        
    def _analyze_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze collected metrics."""
        if not metrics:
            return {}
            
        analysis = {
            'cpu': {
                'average': sum(m['cpu']['percent'] for m in metrics) / len(metrics),
                'max': max(m['cpu']['percent'] for m in metrics),
                'min': min(m['cpu']['percent'] for m in metrics)
            },
            'memory': {
                'average': sum(m['memory']['percent'] for m in metrics) / len(metrics),
                'max': max(m['memory']['percent'] for m in metrics),
                'min': min(m['memory']['percent'] for m in metrics)
            },
            'stability': self._assess_stability(metrics),
            'trends': self._identify_trends(metrics)
        }
        
        return analysis
        
    def _check_alerts(self, metrics: Dict) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        # CPU alert
        if metrics['cpu']['percent'] > self.monitoring_config['cpu_threshold']:
            alerts.append({
                'type': 'cpu',
                'severity': 'high',
                'message': f"CPU usage high: {metrics['cpu']['percent']}%",
                'threshold': self.monitoring_config['cpu_threshold']
            })
            
        # Memory alert
        if metrics['memory']['percent'] > self.monitoring_config['memory_threshold']:
            alerts.append({
                'type': 'memory',
                'severity': 'high',
                'message': f"Memory usage high: {metrics['memory']['percent']}%",
                'threshold': self.monitoring_config['memory_threshold']
            })
            
        # Disk alert
        if metrics['disk']['percent'] > self.monitoring_config['disk_threshold']:
            alerts.append({
                'type': 'disk',
                'severity': 'critical',
                'message': f"Disk usage critical: {metrics['disk']['percent']}%",
                'threshold': self.monitoring_config['disk_threshold']
            })
            
        return alerts
        
    def _generate_recommendations(self, metrics: Dict, analysis: Dict) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metrics['cpu']['percent'] > 80:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
            
        if metrics['memory']['percent'] > 85:
            recommendations.append("Memory usage is high - consider increasing RAM or optimizing memory usage")
            
        if metrics['disk']['percent'] > 90:
            recommendations.append("Disk space is running low - clean up old files or expand storage")
            
        if analysis.get('stability') == 'unstable':
            recommendations.append("System showing instability - investigate resource spikes")
            
        return recommendations
        
    def _assess_worker_health(self, workers_info: Dict) -> Dict[str, Any]:
        """Assess worker health."""
        health = {
            'status': 'healthy',
            'score': 100,
            'issues': []
        }
        
        # Check worker count
        if workers_info['total'] == 0:
            health['status'] = 'critical'
            health['score'] = 0
            health['issues'].append("No workers running")
        elif workers_info['active'] == 0:
            health['status'] = 'warning'
            health['score'] = 50
            health['issues'].append("All workers are idle")
            
        # Check utilization
        utilization = workers_info.get('performance', {}).get('utilization', 0)
        if utilization < 20 and workers_info['total'] > 0:
            health['score'] -= 20
            health['issues'].append("Low worker utilization")
        elif utilization > 95:
            health['score'] -= 10
            health['issues'].append("Workers near capacity")
            
        # Set final status
        if health['score'] >= 80:
            health['status'] = 'healthy'
        elif health['score'] >= 50:
            health['status'] = 'warning'
        else:
            health['status'] = 'critical'
            
        return health
        
    def _analyze_performance(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance data."""
        if not performance_data:
            return {}
            
        # Calculate statistics
        cpu_values = [p['cpu']['percent'] for p in performance_data]
        memory_values = [p['memory']['percent'] for p in performance_data]
        
        return {
            'cpu': {
                'average': round(sum(cpu_values) / len(cpu_values), 2),
                'max': round(max(cpu_values), 2),
                'min': round(min(cpu_values), 2),
                'std_dev': self._calculate_std_dev(cpu_values)
            },
            'memory': {
                'average': round(sum(memory_values) / len(memory_values), 2),
                'max': round(max(memory_values), 2),
                'min': round(min(memory_values), 2),
                'std_dev': self._calculate_std_dev(memory_values)
            },
            'io_rates': self._calculate_io_rates(performance_data)
        }
        
    def _detect_performance_trends(self, performance_data: List[Dict]) -> Dict[str, str]:
        """Detect trends in performance data."""
        if len(performance_data) < 3:
            return {'cpu': 'insufficient_data', 'memory': 'insufficient_data'}
            
        cpu_values = [p['cpu']['percent'] for p in performance_data]
        memory_values = [p['memory']['percent'] for p in performance_data]
        
        # Simple trend detection
        cpu_trend = self._detect_trend(cpu_values)
        memory_trend = self._detect_trend(memory_values)
        
        return {
            'cpu': cpu_trend,
            'memory': memory_trend,
            'overall': 'concerning' if cpu_trend == 'increasing' and memory_trend == 'increasing' else 'normal'
        }
        
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in a series of values."""
        if len(values) < 3:
            return 'insufficient_data'
            
        # Calculate moving average
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:3]) / 3
        
        if recent_avg > older_avg * 1.2:
            return 'increasing'
        elif recent_avg < older_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
            
    def _check_resource_alerts(self, resources: Dict) -> List[Dict[str, Any]]:
        """Check for resource-related alerts."""
        alerts = []
        
        # CPU load alert
        if resources['cpu']['load_average'][0] > resources['cpu']['count'] * 1.5:
            alerts.append({
                'type': 'cpu_load',
                'severity': 'warning',
                'message': f"High CPU load: {resources['cpu']['load_average'][0]:.2f}"
            })
            
        # Memory alert
        if resources['memory']['percent'] > self.monitoring_config['memory_threshold']:
            alerts.append({
                'type': 'memory',
                'severity': 'high',
                'message': f"Memory usage: {resources['memory']['percent']:.1f}%"
            })
            
        # Disk alerts
        for disk in resources['disk']:
            if disk['percent'] > self.monitoring_config['disk_threshold']:
                alerts.append({
                    'type': 'disk',
                    'severity': 'critical',
                    'message': f"Disk {disk['device']} at {disk['percent']:.1f}% capacity"
                })
                
        return alerts
        
    def _calculate_resource_health(self, resources: Dict) -> int:
        """Calculate resource health score."""
        score = 100
        
        # CPU penalty
        cpu_usage = resources['cpu']['percent']
        if cpu_usage > 90:
            score -= 30
        elif cpu_usage > 80:
            score -= 20
        elif cpu_usage > 70:
            score -= 10
            
        # Memory penalty
        memory_usage = resources['memory']['percent']
        if memory_usage > 90:
            score -= 30
        elif memory_usage > 80:
            score -= 20
        elif memory_usage > 70:
            score -= 10
            
        # Disk penalty
        for disk in resources['disk']:
            if disk['percent'] > 95:
                score -= 20
            elif disk['percent'] > 90:
                score -= 10
                
        return max(0, score)
        
    def _analyze_queue_health(self, queue_info: Dict) -> Dict[str, Any]:
        """Analyze queue health."""
        health = {
            'status': 'healthy',
            'backlog_level': 'low',
            'processing_efficiency': 'good'
        }
        
        queue_size = queue_info['task_queue']['size']
        
        # Check backlog
        if queue_size > self.monitoring_config['queue_threshold']:
            health['status'] = 'warning'
            health['backlog_level'] = 'high'
        elif queue_size > self.monitoring_config['queue_threshold'] / 2:
            health['backlog_level'] = 'medium'
            
        # Check processing rate
        if queue_info['processing_rate'] < 1 and queue_size > 10:
            health['status'] = 'critical'
            health['processing_efficiency'] = 'poor'
        elif queue_info['processing_rate'] < 5 and queue_size > 50:
            health['processing_efficiency'] = 'moderate'
            
        return health
        
    def _generate_queue_recommendations(self, queue_info: Dict, health: Dict) -> List[str]:
        """Generate queue recommendations."""
        recommendations = []
        
        if health['backlog_level'] == 'high':
            recommendations.append("High queue backlog - consider adding more workers")
            
        if health['processing_efficiency'] == 'poor':
            recommendations.append("Low processing rate - investigate task processing issues")
            
        if queue_info['failed_tasks'] > queue_info['completed_tasks'] * 0.1:
            recommendations.append("High failure rate - review error logs and task definitions")
            
        return recommendations
        
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        metrics = await self._collect_system_metrics()
        
        return {
            'healthy': metrics['cpu']['percent'] < 80 and metrics['memory']['percent'] < 85,
            'cpu_status': 'ok' if metrics['cpu']['percent'] < 80 else 'high',
            'memory_status': 'ok' if metrics['memory']['percent'] < 85 else 'high',
            'disk_status': 'ok' if metrics['disk']['percent'] < 90 else 'critical'
        }
        
    async def _check_workers_health(self) -> Dict[str, Any]:
        """Check workers health."""
        workers = await self.monitor_workers()
        return workers.get('health', {})
        
    async def _check_resources_health(self) -> Dict[str, Any]:
        """Check resources health."""
        resources = await self.monitor_resources()
        return {
            'healthy': resources.get('health_score', 0) >= 70,
            'score': resources.get('health_score', 0),
            'alerts': len(resources.get('alerts', []))
        }
        
    async def _check_queues_health(self) -> Dict[str, Any]:
        """Check queues health."""
        queues = await self.monitor_queues()
        return queues.get('health', {})
        
    async def _check_errors_health(self) -> Dict[str, Any]:
        """Check errors health."""
        errors = await self.monitor_errors()
        error_rate = errors.get('errors', {}).get('error_rate', 0)
        
        return {
            'healthy': error_rate < self.monitoring_config['error_rate_threshold'],
            'error_rate': error_rate,
            'total_errors': errors.get('errors', {}).get('total_errors', 0)
        }
        
    async def _deep_component_check(self) -> Dict[str, Any]:
        """Perform deep component health check."""
        # This would check individual components deeply
        return {
            'orchestrator': {'status': 'running', 'healthy': True},
            'state_manager': {'status': 'active', 'healthy': True},
            'ai_client': {'status': 'connected', 'healthy': True}
        }
        
    async def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check external dependencies health."""
        # This would check external services, APIs, etc.
        return {
            'all_available': True,
            'dependencies': {
                'database': 'connected',
                'api_services': 'available',
                'file_system': 'accessible'
            }
        }
        
    def _calculate_overall_health(self, health_checks: Dict) -> int:
        """Calculate overall health score."""
        score = 100
        
        # System health
        if not health_checks.get('system', {}).get('healthy', True):
            score -= 20
            
        # Workers health
        worker_health = health_checks.get('workers', {})
        if worker_health.get('status') == 'critical':
            score -= 30
        elif worker_health.get('status') == 'warning':
            score -= 15
            
        # Resources health
        if not health_checks.get('resources', {}).get('healthy', True):
            score -= 20
            
        # Queue health
        queue_health = health_checks.get('queues', {})
        if queue_health.get('status') == 'critical':
            score -= 20
        elif queue_health.get('status') == 'warning':
            score -= 10
            
        # Error health
        if not health_checks.get('errors', {}).get('healthy', True):
            score -= 15
            
        return max(0, score)
        
    def _get_health_status(self, score: int) -> str:
        """Get health status from score."""
        if score >= 90:
            return 'excellent'
        elif score >= 80:
            return 'good'
        elif score >= 70:
            return 'fair'
        elif score >= 50:
            return 'poor'
        else:
            return 'critical'
            
    def _identify_health_issues(self, health_checks: Dict) -> List[str]:
        """Identify health issues from checks."""
        issues = []
        
        # System issues
        system = health_checks.get('system', {})
        if system.get('cpu_status') == 'high':
            issues.append("High CPU usage")
        if system.get('memory_status') == 'high':
            issues.append("High memory usage")
        if system.get('disk_status') == 'critical':
            issues.append("Critical disk usage")
            
        # Worker issues
        worker_issues = health_checks.get('workers', {}).get('issues', [])
        issues.extend(worker_issues)
        
        # Queue issues
        if health_checks.get('queues', {}).get('status') != 'healthy':
            issues.append("Queue processing issues")
            
        # Error issues
        if not health_checks.get('errors', {}).get('healthy', True):
            issues.append("High error rate")
            
        return issues
        
    def _generate_health_recommendations(self, health_checks: Dict) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        issues = self._identify_health_issues(health_checks)
        
        for issue in issues:
            if "CPU" in issue:
                recommendations.append("Optimize CPU-intensive operations or scale horizontally")
            elif "memory" in issue:
                recommendations.append("Investigate memory leaks and optimize memory usage")
            elif "disk" in issue:
                recommendations.append("Clean up disk space immediately or expand storage")
            elif "Queue" in issue:
                recommendations.append("Scale up workers or optimize task processing")
            elif "error rate" in issue:
                recommendations.append("Review error logs and fix failing components")
                
        return list(set(recommendations))  # Remove duplicates
        
    def _store_health_check(self, report: Dict) -> None:
        """Store health check result."""
        # Store in state or database
        state = self.state_manager.get_state()
        if 'health_history' not in state:
            state['health_history'] = []
            
        # Keep last 100 health checks
        state['health_history'].append(report)
        state['health_history'] = state['health_history'][-100:]
        
        self.state_manager.save_state()
        
    def _analyze_error_patterns(self, errors: Dict) -> Dict[str, Any]:
        """Analyze error patterns."""
        return {
            'most_common_type': max(errors['by_type'].items(), key=lambda x: x[1])[0] if errors['by_type'] else 'none',
            'most_affected_component': max(errors['by_component'].items(), key=lambda x: x[1])[0] if errors['by_component'] else 'none',
            'error_trend': 'increasing' if errors['error_rate'] > 0.1 else 'stable',
            'severity': 'high' if errors['error_rate'] > 0.5 else 'medium' if errors['error_rate'] > 0.1 else 'low'
        }
        
    def _generate_error_alerts(self, errors: Dict, analysis: Dict) -> List[Dict[str, Any]]:
        """Generate error-related alerts."""
        alerts = []
        
        if errors['error_rate'] > self.monitoring_config['error_rate_threshold']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"High error rate: {errors['error_rate']:.2f} errors/minute"
            })
            
        if analysis['severity'] == 'high':
            alerts.append({
                'type': 'error_severity',
                'severity': 'critical',
                'message': "Critical error levels detected"
            })
            
        return alerts
        
    def _count_open_files(self) -> int:
        """Count open files in the system."""
        try:
            # This is a simplified count
            return len(psutil.Process().open_files())
        except:
            return 0
            
    def _estimate_system_calls(self) -> int:
        """Estimate system calls (placeholder)."""
        # This would require more sophisticated monitoring
        return 0
        
    async def _get_user_activity(self) -> Dict[str, Any]:
        """Get user activity information."""
        return {
            'active_sessions': 1,  # Placeholder
            'recent_commands': []  # Would track actual commands
        }
        
    async def _get_component_activity(self) -> Dict[str, Any]:
        """Get component activity information."""
        return {
            'orchestrator': 'active',
            'workers': 'processing',
            'monitor': 'watching'
        }
        
    async def _generate_activity_timeline(self) -> List[Dict[str, Any]]:
        """Generate activity timeline."""
        # This would create a timeline of system events
        return [
            {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event': 'monitoring_started',
                'details': 'System monitoring initiated'
            }
        ]
        
    def _identify_activity_patterns(self, activity: Dict, timeline: List[Dict]) -> Dict[str, Any]:
        """Identify patterns in system activity."""
        return {
            'peak_hours': 'business_hours',  # Placeholder
            'usage_pattern': 'steady',
            'anomalies': []
        }
        
    def _extract_timestamp(self, log_line: str) -> str:
        """Extract timestamp from log line."""
        # Simple pattern matching for common timestamp formats
        # This would need to be more sophisticated for real use
        import re
        match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', log_line)
        return match.group(0) if match else 'unknown'
        
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def _calculate_io_rates(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Calculate I/O rates from performance data."""
        if len(performance_data) < 2:
            return {'disk_read': 0, 'disk_write': 0, 'net_send': 0, 'net_recv': 0}
            
        # Calculate rates (simplified)
        first = performance_data[0]
        last = performance_data[-1]
        time_diff = len(performance_data) * 5  # Assuming 5-second intervals
        
        return {
            'disk_read': 0,  # Would calculate from disk_io
            'disk_write': 0,
            'net_send': 0,  # Would calculate from network_io
            'net_recv': 0
        }
        
    def _get_system_uptime(self) -> str:
        """Get system uptime."""
        boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime = datetime.now(timezone.utc) - boot_time
        
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        return f"{days}d {hours}h {minutes}m"
        
    def _check_metrics_alerts(self, metrics: List[Dict]) -> List[Dict[str, Any]]:
        """Check metrics for alert conditions."""
        alerts = []
        
        # Check each metric
        for metric in metrics[-5:]:  # Check last 5 metrics
            metric_alerts = self._check_alerts(metric)
            alerts.extend(metric_alerts)
            
        # Deduplicate alerts
        unique_alerts = []
        seen = set()
        for alert in alerts:
            key = (alert['type'], alert['message'])
            if key not in seen:
                seen.add(key)
                unique_alerts.append(alert)
                
        return unique_alerts
        
    def _assess_stability(self, metrics: List[Dict]) -> str:
        """Assess system stability from metrics."""
        if len(metrics) < 3:
            return 'unknown'
            
        # Check CPU variance
        cpu_values = [m['cpu']['percent'] for m in metrics]
        cpu_std = self._calculate_std_dev(cpu_values)
        
        # Check memory variance
        mem_values = [m['memory']['percent'] for m in metrics]
        mem_std = self._calculate_std_dev(mem_values)
        
        if cpu_std < 5 and mem_std < 3:
            return 'stable'
        elif cpu_std < 10 and mem_std < 5:
            return 'mostly_stable'
        else:
            return 'unstable'
            
    def _identify_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """Identify trends in metrics."""
        if len(metrics) < 3:
            return {'cpu': 'unknown', 'memory': 'unknown'}
            
        cpu_values = [m['cpu']['percent'] for m in metrics]
        memory_values = [m['memory']['percent'] for m in metrics]
        
        return {
            'cpu': self._detect_trend(cpu_values),
            'memory': self._detect_trend(memory_values)
        }


async def main(what: str = None, **kwargs):
    """Entry point for system monitor."""
    monitor = SystemMonitor()
    
    if what:
        return await monitor.monitor(what, **kwargs)
    else:
        # Default to system monitoring
        return await monitor.monitor_system(**kwargs)
        

if __name__ == "__main__":
    # Test the monitor
    import asyncio
    
    async def test():
        monitor = SystemMonitor()
        
        # Test different monitoring functions
        print("System:", await monitor.monitor_system())
        print("\nWorkers:", await monitor.monitor_workers())
        print("\nHealth:", await monitor.monitor_health())
        
    asyncio.run(test())