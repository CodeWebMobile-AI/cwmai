"""
System Performance Monitor
Real-time monitoring of system performance metrics.
"""

import psutil
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List


class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'percent': psutil.virtual_memory().percent,
                'available': psutil.virtual_memory().available,
                'total': psutil.virtual_memory().total
            },
            'disk': {
                'percent': psutil.disk_usage('/').percent,
                'free': psutil.disk_usage('/').free,
                'total': psutil.disk_usage('/').total
            },
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
        
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics and generate alerts."""
        if metrics['cpu']['percent'] > 90:
            self.alerts.append({
                'type': 'high_cpu',
                'value': metrics['cpu']['percent'],
                'timestamp': metrics['timestamp']
            })
            
        if metrics['memory']['percent'] > 85:
            self.alerts.append({
                'type': 'high_memory',
                'value': metrics['memory']['percent'],
                'timestamp': metrics['timestamp']
            })
            
        if metrics['disk']['percent'] > 90:
            self.alerts.append({
                'type': 'low_disk',
                'value': metrics['disk']['percent'],
                'timestamp': metrics['timestamp']
            })
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with trends."""
        if not self.metrics_history:
            await self.collect_metrics()
            
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        # Calculate averages
        cpu_avg = sum(m['cpu']['percent'] for m in recent_metrics) / len(recent_metrics)
        memory_avg = sum(m['memory']['percent'] for m in recent_metrics) / len(recent_metrics)
        
        # Detect trends
        cpu_trend = 'stable'
        if len(recent_metrics) >= 5:
            first_half = recent_metrics[:len(recent_metrics)//2]
            second_half = recent_metrics[len(recent_metrics)//2:]
            
            first_cpu = sum(m['cpu']['percent'] for m in first_half) / len(first_half)
            second_cpu = sum(m['cpu']['percent'] for m in second_half) / len(second_half)
            
            if second_cpu > first_cpu * 1.2:
                cpu_trend = 'increasing'
            elif second_cpu < first_cpu * 0.8:
                cpu_trend = 'decreasing'
                
        return {
            'current': recent_metrics[-1] if recent_metrics else None,
            'averages': {
                'cpu': cpu_avg,
                'memory': memory_avg
            },
            'trends': {
                'cpu': cpu_trend
            },
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'health_score': 100 - max(cpu_avg, memory_avg)
        }


# Global monitor instance
monitor = PerformanceMonitor()


async def monitor_performance(duration_seconds: int = 60) -> Dict[str, Any]:
    """Monitor performance for specified duration."""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < duration_seconds:
        await monitor.collect_metrics()
        await asyncio.sleep(5)  # Collect every 5 seconds
        
    return await monitor.get_performance_summary()


async def get_current_performance() -> Dict[str, Any]:
    """Get current performance metrics."""
    return await monitor.collect_metrics()


# Export functions
__all__ = ['monitor_performance', 'get_current_performance', 'PerformanceMonitor']
