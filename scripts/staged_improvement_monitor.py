"""
Staged Improvement Monitor

Monitors and tracks metrics for staged improvements.
Provides before/after comparisons and real-time monitoring.
"""

import os
import json
import time
import psutil
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import traceback
import threading
from collections import defaultdict, deque


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    error_count: int
    response_time_ms: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ImprovementMetrics:
    """Metrics for a specific improvement."""
    staging_id: str
    file_path: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    runtime_metrics: List[PerformanceSnapshot]
    start_time: datetime
    end_time: Optional[datetime] = None
    verdict: Optional[str] = None  # 'improved', 'degraded', 'neutral'


class StagedImprovementMonitor:
    """Monitor performance and behavior of staged improvements."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize monitor with repository path."""
        self.repo_path = os.path.abspath(repo_path)
        self.metrics_dir = os.path.join(repo_path, '.self_improver', 'metrics')
        self.active_monitors: Dict[str, ImprovementMetrics] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.error_counts = defaultdict(int)
        
        # Ensure metrics directories exist
        os.makedirs(os.path.join(self.metrics_dir, 'before'), exist_ok=True)
        os.makedirs(os.path.join(self.metrics_dir, 'after'), exist_ok=True)
        os.makedirs(os.path.join(self.metrics_dir, 'runtime'), exist_ok=True)
    
    async def start_monitoring(self, staging_id: str, file_path: str) -> ImprovementMetrics:
        """Start monitoring an improvement.
        
        Args:
            staging_id: Unique staging identifier
            file_path: Path to the file being improved
            
        Returns:
            ImprovementMetrics object for tracking
        """
        print(f"📊 Starting monitoring for {staging_id}")
        
        # Collect before metrics
        before_metrics = await self._collect_static_metrics(file_path)
        
        # Initialize metrics object
        metrics = ImprovementMetrics(
            staging_id=staging_id,
            file_path=file_path,
            before_metrics=before_metrics,
            after_metrics={},
            runtime_metrics=[],
            start_time=datetime.now(timezone.utc)
        )
        
        # Store in active monitors
        self.active_monitors[staging_id] = metrics
        
        # Save before metrics
        self._save_metrics(staging_id, 'before', before_metrics)
        
        # Start runtime monitoring if not already running
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self._start_runtime_monitoring()
        
        return metrics
    
    async def stop_monitoring(self, staging_id: str, 
                            improvement_applied: bool = False) -> ImprovementMetrics:
        """Stop monitoring an improvement.
        
        Args:
            staging_id: Unique staging identifier
            improvement_applied: Whether the improvement was applied
            
        Returns:
            Final ImprovementMetrics with analysis
        """
        if staging_id not in self.active_monitors:
            raise ValueError(f"No active monitoring for {staging_id}")
        
        metrics = self.active_monitors[staging_id]
        metrics.end_time = datetime.now(timezone.utc)
        
        print(f"📊 Stopping monitoring for {staging_id}")
        
        # Collect after metrics if improvement was applied
        if improvement_applied:
            metrics.after_metrics = await self._collect_static_metrics(
                metrics.file_path
            )
            self._save_metrics(staging_id, 'after', metrics.after_metrics)
        
        # Analyze and determine verdict
        metrics.verdict = self._analyze_improvement(metrics)
        
        # Save runtime metrics
        self._save_runtime_metrics(staging_id, metrics.runtime_metrics)
        
        # Generate monitoring report
        self._generate_monitoring_report(metrics)
        
        # Remove from active monitors
        del self.active_monitors[staging_id]
        
        # Stop runtime monitoring if no active monitors
        if not self.active_monitors:
            self.stop_monitoring.set()
        
        return metrics
    
    async def _collect_static_metrics(self, file_path: str) -> Dict[str, Any]:
        """Collect static metrics for a file."""
        metrics = {
            'file_size': 0,
            'line_count': 0,
            'function_count': 0,
            'class_count': 0,
            'complexity': 0,
            'import_count': 0,
            'docstring_count': 0,
            'test_coverage': None
        }
        
        try:
            # Basic file metrics
            if os.path.exists(file_path):
                metrics['file_size'] = os.path.getsize(file_path)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    metrics['line_count'] = len(lines)
                
                # Parse AST for code metrics
                import ast
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        metrics['function_count'] += 1
                        if ast.get_docstring(node):
                            metrics['docstring_count'] += 1
                    elif isinstance(node, ast.ClassDef):
                        metrics['class_count'] += 1
                        if ast.get_docstring(node):
                            metrics['docstring_count'] += 1
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        metrics['import_count'] += 1
                    elif isinstance(node, (ast.If, ast.While, ast.For)):
                        metrics['complexity'] += 1
            
            # Try to get test coverage if available
            coverage_file = os.path.join(self.repo_path, '.coverage')
            if os.path.exists(coverage_file):
                # Simplified - would need proper coverage parsing
                metrics['test_coverage'] = 'available'
                
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _start_runtime_monitoring(self):
        """Start background thread for runtime monitoring."""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._runtime_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _runtime_monitoring_loop(self):
        """Background loop for collecting runtime metrics."""
        while not self.stop_monitoring.is_set():
            try:
                # Collect snapshot for all active monitors
                snapshot = self._collect_performance_snapshot()
                
                # Add to each active monitor
                for staging_id, metrics in self.active_monitors.items():
                    metrics.runtime_metrics.append(snapshot)
                
                # Sleep for monitoring interval
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics."""
        try:
            # System metrics
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Error count (simplified - would integrate with logging)
            error_count = sum(self.error_counts.values())
            
            # Response time (simplified - would measure actual operations)
            response_time_ms = 10.0  # Placeholder
            
            # Custom metrics could be added here
            custom_metrics = {}
            
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                error_count=error_count,
                response_time_ms=response_time_ms,
                custom_metrics=custom_metrics
            )
            
        except Exception as e:
            # Return default snapshot on error
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_mb=0.0,
                error_count=0,
                response_time_ms=0.0
            )
    
    def _analyze_improvement(self, metrics: ImprovementMetrics) -> str:
        """Analyze metrics to determine if improvement was successful."""
        if not metrics.after_metrics:
            return 'unknown'
        
        before = metrics.before_metrics
        after = metrics.after_metrics
        
        # Calculate improvements/degradations
        improvements = 0
        degradations = 0
        
        # File size comparison
        if after['file_size'] < before['file_size']:
            improvements += 1
        elif after['file_size'] > before['file_size'] * 1.2:
            degradations += 1
        
        # Complexity comparison
        if after['complexity'] < before['complexity']:
            improvements += 1
        elif after['complexity'] > before['complexity'] * 1.3:
            degradations += 1
        
        # Documentation improvement
        if after['docstring_count'] > before['docstring_count']:
            improvements += 1
        
        # Line count (prefer fewer lines if functionality is same)
        if after['line_count'] < before['line_count'] * 0.9:
            improvements += 1
        elif after['line_count'] > before['line_count'] * 1.1:
            degradations += 1
        
        # Runtime metrics analysis
        if metrics.runtime_metrics:
            avg_cpu_before = sum(s.cpu_percent for s in metrics.runtime_metrics[:5]) / 5
            avg_cpu_after = sum(s.cpu_percent for s in metrics.runtime_metrics[-5:]) / 5
            
            if avg_cpu_after < avg_cpu_before * 0.9:
                improvements += 1
            elif avg_cpu_after > avg_cpu_before * 1.1:
                degradations += 1
        
        # Determine verdict
        if improvements > degradations + 1:
            return 'improved'
        elif degradations > improvements + 1:
            return 'degraded'
        else:
            return 'neutral'
    
    def _save_metrics(self, staging_id: str, phase: str, metrics: Dict[str, Any]):
        """Save metrics to file."""
        filename = f"{staging_id}_{phase}.json"
        filepath = os.path.join(self.metrics_dir, phase, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'staging_id': staging_id,
                'phase': phase,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics
            }, f, indent=2)
    
    def _save_runtime_metrics(self, staging_id: str, 
                             snapshots: List[PerformanceSnapshot]):
        """Save runtime metrics to file."""
        filename = f"{staging_id}_runtime.json"
        filepath = os.path.join(self.metrics_dir, 'runtime', filename)
        
        runtime_data = []
        for snapshot in snapshots:
            runtime_data.append({
                'timestamp': snapshot.timestamp.isoformat(),
                'cpu_percent': snapshot.cpu_percent,
                'memory_mb': snapshot.memory_mb,
                'error_count': snapshot.error_count,
                'response_time_ms': snapshot.response_time_ms,
                'custom_metrics': snapshot.custom_metrics
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                'staging_id': staging_id,
                'runtime_metrics': runtime_data
            }, f, indent=2)
    
    def _generate_monitoring_report(self, metrics: ImprovementMetrics):
        """Generate a monitoring report for the improvement."""
        report = {
            'staging_id': metrics.staging_id,
            'file_path': metrics.file_path,
            'monitoring_duration': str(metrics.end_time - metrics.start_time),
            'verdict': metrics.verdict,
            'before_metrics': metrics.before_metrics,
            'after_metrics': metrics.after_metrics,
            'improvements': [],
            'degradations': [],
            'runtime_summary': {}
        }
        
        # Compare before/after metrics
        if metrics.after_metrics:
            for key in metrics.before_metrics:
                before_val = metrics.before_metrics[key]
                after_val = metrics.after_metrics.get(key)
                
                if before_val is not None and after_val is not None:
                    if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                        change_pct = ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
                        
                        if abs(change_pct) > 5:  # Significant change
                            change_info = {
                                'metric': key,
                                'before': before_val,
                                'after': after_val,
                                'change_percent': round(change_pct, 2)
                            }
                            
                            if change_pct < 0 and key in ['complexity', 'file_size', 'line_count']:
                                report['improvements'].append(change_info)
                            elif change_pct > 0 and key in ['docstring_count', 'test_coverage']:
                                report['improvements'].append(change_info)
                            elif change_pct > 0 and key in ['complexity', 'file_size']:
                                report['degradations'].append(change_info)
        
        # Runtime summary
        if metrics.runtime_metrics:
            report['runtime_summary'] = {
                'samples_collected': len(metrics.runtime_metrics),
                'avg_cpu_percent': sum(s.cpu_percent for s in metrics.runtime_metrics) / len(metrics.runtime_metrics),
                'max_cpu_percent': max(s.cpu_percent for s in metrics.runtime_metrics),
                'avg_memory_mb': sum(s.memory_mb for s in metrics.runtime_metrics) / len(metrics.runtime_metrics),
                'max_memory_mb': max(s.memory_mb for s in metrics.runtime_metrics),
                'total_errors': max(s.error_count for s in metrics.runtime_metrics)
            }
        
        # Save report
        report_dir = os.path.join(self.repo_path, '.self_improver', 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(
            report_dir,
            f"monitoring_{metrics.staging_id}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📊 Monitoring Report for {metrics.staging_id}")
        print(f"Verdict: {metrics.verdict.upper()}")
        print(f"Improvements: {len(report['improvements'])}")
        print(f"Degradations: {len(report['degradations'])}")
    
    async def get_historical_performance(self, file_path: str) -> Dict[str, Any]:
        """Get historical performance data for a file."""
        history = {
            'file_path': file_path,
            'total_improvements': 0,
            'successful_improvements': 0,
            'average_improvement': 0.0,
            'metrics_history': []
        }
        
        # Look for historical metrics for this file
        for phase_dir in ['before', 'after']:
            phase_path = os.path.join(self.metrics_dir, phase_dir)
            if not os.path.exists(phase_path):
                continue
            
            for filename in os.listdir(phase_path):
                if filename.endswith('.json'):
                    with open(os.path.join(phase_path, filename), 'r') as f:
                        data = json.load(f)
                    
                    # Check if metrics are for this file
                    # (Would need to store file path in metrics)
                    history['metrics_history'].append(data)
        
        return history
    
    def record_error(self, staging_id: str, error_type: str):
        """Record an error during improvement execution."""
        self.error_counts[f"{staging_id}:{error_type}"] += 1
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'active_monitors': len(self.active_monitors),
            'monitoring_files': [m.file_path for m in self.active_monitors.values()],
            'total_errors': sum(self.error_counts.values()),
            'monitoring_thread_active': self.monitoring_thread.is_alive() if self.monitoring_thread else False
        }


class ABTestMonitor:
    """Monitor for A/B testing improvements."""
    
    def __init__(self, monitor: StagedImprovementMonitor):
        """Initialize A/B test monitor."""
        self.monitor = monitor
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
    
    async def start_ab_test(self, staging_id: str, 
                           original_path: str,
                           improved_path: str,
                           duration_seconds: int = 3600) -> str:
        """Start an A/B test for an improvement."""
        test_id = f"ab_test_{staging_id}"
        
        self.ab_tests[test_id] = {
            'staging_id': staging_id,
            'original_path': original_path,
            'improved_path': improved_path,
            'start_time': datetime.now(timezone.utc),
            'duration': duration_seconds,
            'original_metrics': [],
            'improved_metrics': [],
            'verdict': None
        }
        
        print(f"🔬 Starting A/B test {test_id} for {duration_seconds} seconds")
        
        # Start monitoring both versions
        asyncio.create_task(self._run_ab_test(test_id))
        
        return test_id
    
    async def _run_ab_test(self, test_id: str):
        """Run the A/B test."""
        test = self.ab_tests[test_id]
        end_time = test['start_time'] + timedelta(seconds=test['duration'])
        
        while datetime.now(timezone.utc) < end_time:
            # Collect metrics for both versions
            # In real implementation, would route traffic between versions
            
            # Simulate metric collection
            original_metric = await self._collect_version_metrics('original')
            improved_metric = await self._collect_version_metrics('improved')
            
            test['original_metrics'].append(original_metric)
            test['improved_metrics'].append(improved_metric)
            
            await asyncio.sleep(60)  # Collect every minute
        
        # Analyze results
        test['verdict'] = self._analyze_ab_test(test)
        self._generate_ab_test_report(test_id)
    
    async def _collect_version_metrics(self, version: str) -> Dict[str, float]:
        """Collect metrics for a version (simplified)."""
        # In real implementation, would measure actual performance
        import random
        
        if version == 'improved':
            # Simulate improved performance
            return {
                'response_time': random.uniform(8, 12),
                'error_rate': random.uniform(0, 0.02),
                'cpu_usage': random.uniform(10, 20)
            }
        else:
            # Original performance
            return {
                'response_time': random.uniform(10, 15),
                'error_rate': random.uniform(0, 0.03),
                'cpu_usage': random.uniform(15, 25)
            }
    
    def _analyze_ab_test(self, test: Dict[str, Any]) -> str:
        """Analyze A/B test results."""
        if not test['original_metrics'] or not test['improved_metrics']:
            return 'insufficient_data'
        
        # Calculate averages
        orig_avg_response = sum(m['response_time'] for m in test['original_metrics']) / len(test['original_metrics'])
        imp_avg_response = sum(m['response_time'] for m in test['improved_metrics']) / len(test['improved_metrics'])
        
        orig_avg_error = sum(m['error_rate'] for m in test['original_metrics']) / len(test['original_metrics'])
        imp_avg_error = sum(m['error_rate'] for m in test['improved_metrics']) / len(test['improved_metrics'])
        
        # Compare
        response_improved = imp_avg_response < orig_avg_response * 0.95
        errors_improved = imp_avg_error < orig_avg_error * 0.95
        
        if response_improved and errors_improved:
            return 'improved'
        elif response_improved or errors_improved:
            return 'partially_improved'
        else:
            return 'no_improvement'
    
    def _generate_ab_test_report(self, test_id: str):
        """Generate A/B test report."""
        test = self.ab_tests[test_id]
        
        report = {
            'test_id': test_id,
            'staging_id': test['staging_id'],
            'duration': test['duration'],
            'samples_collected': len(test['original_metrics']),
            'verdict': test['verdict'],
            'metrics_comparison': {}
        }
        
        # Calculate comparisons
        if test['original_metrics'] and test['improved_metrics']:
            for metric in ['response_time', 'error_rate', 'cpu_usage']:
                orig_avg = sum(m[metric] for m in test['original_metrics']) / len(test['original_metrics'])
                imp_avg = sum(m[metric] for m in test['improved_metrics']) / len(test['improved_metrics'])
                
                report['metrics_comparison'][metric] = {
                    'original': round(orig_avg, 3),
                    'improved': round(imp_avg, 3),
                    'change_percent': round((imp_avg - orig_avg) / orig_avg * 100, 2)
                }
        
        # Save report
        report_dir = os.path.join(self.monitor.repo_path, '.self_improver', 'reports')
        report_path = os.path.join(report_dir, f"ab_test_{test['staging_id']}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"🔬 A/B Test Complete: {test['verdict']}")


# Standalone monitoring example
async def monitor_improvement_example():
    """Example of monitoring an improvement."""
    monitor = StagedImprovementMonitor()
    
    # Start monitoring
    metrics = await monitor.start_monitoring(
        staging_id="example_001",
        file_path="/path/to/file.py"
    )
    
    # Simulate some work
    await asyncio.sleep(10)
    
    # Stop monitoring
    final_metrics = await monitor.stop_monitoring(
        staging_id="example_001",
        improvement_applied=True
    )
    
    print(f"Monitoring complete. Verdict: {final_metrics.verdict}")


if __name__ == "__main__":
    asyncio.run(monitor_improvement_example())