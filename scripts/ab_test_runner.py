#!/usr/bin/env python3
"""
A/B Test Runner for Staged Improvements

Enables running A/B tests between original and improved code versions.
"""

import os
import sys
import asyncio
import time
import json
import importlib.util
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import random
import statistics
import argparse

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from staged_self_improver import StagedSelfImprover, StagedImprovement
from staged_improvement_monitor import StagedImprovementMonitor, ABTestMonitor


@dataclass
class ABTestConfig:
    """Configuration for A/B tests."""
    duration_seconds: int = 3600  # 1 hour default
    traffic_split: float = 0.5  # 50/50 split
    min_samples: int = 100
    confidence_level: float = 0.95
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'execution_time', 'memory_usage', 'error_rate', 'throughput'
    ])
    auto_promote: bool = True
    promotion_threshold: float = 0.1  # 10% improvement required


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    test_id: str
    staging_id: str
    start_time: datetime
    end_time: datetime
    original_metrics: Dict[str, List[float]]
    improved_metrics: Dict[str, List[float]]
    statistical_analysis: Dict[str, Any]
    winner: str  # 'original', 'improved', or 'no_difference'
    confidence: float
    recommendation: str


class ABTestRunner:
    """Runs A/B tests for staged improvements."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize A/B test runner."""
        self.repo_path = os.path.abspath(repo_path)
        self.improver = StagedSelfImprover(repo_path)
        self.monitor = StagedImprovementMonitor(repo_path)
        self.ab_monitor = ABTestMonitor(self.monitor)
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: List[ABTestResult] = []
    
    async def start_ab_test(self, staging_id: str, 
                           config: Optional[ABTestConfig] = None) -> str:
        """Start an A/B test for a staged improvement.
        
        Args:
            staging_id: ID of the staged improvement
            config: A/B test configuration
            
        Returns:
            Test ID
        """
        if config is None:
            config = ABTestConfig()
        
        # Get staged improvement
        if staging_id not in self.improver.staged_improvements:
            raise ValueError(f"Staged improvement {staging_id} not found")
        
        improvement = self.improver.staged_improvements[staging_id]
        
        # Validate improvement is ready
        if not improvement.validation_status or not improvement.validation_status.get('ready_to_apply'):
            raise ValueError(f"Improvement {staging_id} not validated or ready")
        
        # Create test ID
        test_id = f"ab_{staging_id}_{int(time.time())}"
        
        # Initialize test
        self.active_tests[test_id] = {
            'config': config,
            'improvement': improvement,
            'start_time': datetime.now(timezone.utc),
            'metrics': {
                'original': {metric: [] for metric in config.metrics_to_track},
                'improved': {metric: [] for metric in config.metrics_to_track}
            },
            'request_count': {'original': 0, 'improved': 0}
        }
        
        print(f"🔬 Starting A/B test {test_id}")
        print(f"  File: {improvement.modification.target_file}")
        print(f"  Duration: {config.duration_seconds}s")
        print(f"  Traffic split: {config.traffic_split:.0%} to improved version")
        
        # Start test execution
        asyncio.create_task(self._run_ab_test(test_id))
        
        return test_id
    
    async def _run_ab_test(self, test_id: str):
        """Run the A/B test."""
        test = self.active_tests[test_id]
        config = test['config']
        improvement = test['improvement']
        
        # Create version switcher
        version_switcher = self._create_version_switcher(improvement)
        
        # Run test for configured duration
        end_time = test['start_time'] + timedelta(seconds=config.duration_seconds)
        
        while datetime.now(timezone.utc) < end_time:
            # Simulate requests/executions
            await self._simulate_execution_batch(
                test_id, version_switcher, batch_size=10
            )
            
            # Check early stopping criteria
            if self._should_stop_early(test_id):
                print(f"⚠️  Stopping test {test_id} early due to significant difference")
                break
            
            await asyncio.sleep(1)  # 1 second between batches
        
        # Finalize test
        await self._finalize_test(test_id)
    
    def _create_version_switcher(self, improvement: StagedImprovement) -> Callable:
        """Create a function that switches between versions."""
        original_path = improvement.original_path
        staged_path = improvement.staged_path
        
        def switch_version(use_improved: bool) -> Any:
            """Dynamically load the specified version."""
            if use_improved:
                spec = importlib.util.spec_from_file_location(
                    "improved_module", staged_path
                )
            else:
                spec = importlib.util.spec_from_file_location(
                    "original_module", original_path
                )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            return None
        
        return switch_version
    
    async def _simulate_execution_batch(self, test_id: str, 
                                      version_switcher: Callable,
                                      batch_size: int = 10):
        """Simulate a batch of executions."""
        test = self.active_tests[test_id]
        config = test['config']
        
        for _ in range(batch_size):
            # Decide which version to use
            use_improved = random.random() < config.traffic_split
            version = 'improved' if use_improved else 'original'
            
            # Measure execution
            metrics = await self._measure_execution(version_switcher, use_improved)
            
            # Record metrics
            for metric_name, value in metrics.items():
                if metric_name in test['metrics'][version]:
                    test['metrics'][version][metric_name].append(value)
            
            test['request_count'][version] += 1
    
    async def _measure_execution(self, version_switcher: Callable, 
                               use_improved: bool) -> Dict[str, float]:
        """Measure execution metrics for a version."""
        metrics = {}
        
        # Start measurements
        start_time = time.time()
        start_memory = self._get_memory_usage()
        error_occurred = False
        
        try:
            # Load and execute module
            module = version_switcher(use_improved)
            
            if module:
                # Execute some standard operations
                # In real implementation, would execute actual workload
                await self._execute_workload(module)
            
        except Exception as e:
            error_occurred = True
        
        # End measurements
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        metrics['execution_time'] = (end_time - start_time) * 1000  # ms
        metrics['memory_usage'] = end_memory - start_memory
        metrics['error_rate'] = 1.0 if error_occurred else 0.0
        metrics['throughput'] = 1.0 / (end_time - start_time) if not error_occurred else 0.0
        
        return metrics
    
    async def _execute_workload(self, module: Any):
        """Execute a standard workload on the module."""
        # This is a simplified workload
        # In real implementation, would execute actual operations
        
        # Try to find and execute common functions
        for func_name in ['main', 'process', 'run', 'execute']:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    # Simple execution
                    if asyncio.iscoroutinefunction(func):
                        await func()
                    else:
                        func()
                    break
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _should_stop_early(self, test_id: str) -> bool:
        """Check if test should stop early due to significant results."""
        test = self.active_tests[test_id]
        config = test['config']
        
        # Need minimum samples
        min_samples = min(
            len(test['metrics']['original']['execution_time']),
            len(test['metrics']['improved']['execution_time'])
        )
        
        if min_samples < config.min_samples:
            return False
        
        # Check for significant difference in error rates
        orig_errors = test['metrics']['original']['error_rate']
        imp_errors = test['metrics']['improved']['error_rate']
        
        if orig_errors and imp_errors:
            orig_error_rate = sum(orig_errors) / len(orig_errors)
            imp_error_rate = sum(imp_errors) / len(imp_errors)
            
            # Stop if improved version has significantly higher error rate
            if imp_error_rate > orig_error_rate * 2 and imp_error_rate > 0.1:
                return True
        
        return False
    
    async def _finalize_test(self, test_id: str):
        """Finalize the A/B test and generate results."""
        test = self.active_tests[test_id]
        config = test['config']
        improvement = test['improvement']
        
        # Perform statistical analysis
        analysis = self._perform_statistical_analysis(test['metrics'])
        
        # Determine winner
        winner = self._determine_winner(analysis, config)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(winner, analysis, config)
        
        # Create result
        result = ABTestResult(
            test_id=test_id,
            staging_id=improvement.metadata['staging_id'],
            start_time=test['start_time'],
            end_time=datetime.now(timezone.utc),
            original_metrics=test['metrics']['original'],
            improved_metrics=test['metrics']['improved'],
            statistical_analysis=analysis,
            winner=winner,
            confidence=analysis.get('confidence', 0.0),
            recommendation=recommendation
        )
        
        self.test_results.append(result)
        
        # Save results
        self._save_test_results(result)
        
        # Auto-promote if configured
        if config.auto_promote and winner == 'improved':
            await self._auto_promote_improvement(improvement)
        
        # Clean up
        del self.active_tests[test_id]
        
        # Print summary
        self._print_test_summary(result)
    
    def _perform_statistical_analysis(self, metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform statistical analysis on test metrics."""
        analysis = {}
        
        for metric_name in metrics['original']:
            orig_data = metrics['original'][metric_name]
            imp_data = metrics['improved'][metric_name]
            
            if not orig_data or not imp_data:
                continue
            
            # Calculate statistics
            orig_mean = statistics.mean(orig_data)
            imp_mean = statistics.mean(imp_data)
            
            orig_stdev = statistics.stdev(orig_data) if len(orig_data) > 1 else 0
            imp_stdev = statistics.stdev(imp_data) if len(imp_data) > 1 else 0
            
            # Calculate improvement
            if orig_mean != 0:
                improvement = (imp_mean - orig_mean) / orig_mean
            else:
                improvement = 0
            
            # Perform t-test (simplified)
            if orig_stdev > 0 and imp_stdev > 0:
                # Welch's t-test
                t_stat = (imp_mean - orig_mean) / (
                    (orig_stdev**2/len(orig_data) + imp_stdev**2/len(imp_data))**0.5
                )
                # Simplified p-value calculation
                p_value = min(1.0, abs(t_stat) / 10)  # Very simplified
            else:
                t_stat = 0
                p_value = 1.0
            
            analysis[metric_name] = {
                'original_mean': orig_mean,
                'improved_mean': imp_mean,
                'original_stdev': orig_stdev,
                'improved_stdev': imp_stdev,
                'improvement_percent': improvement * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Overall confidence
        significant_improvements = sum(
            1 for m in analysis.values() 
            if m.get('significant', False) and m.get('improvement_percent', 0) > 0
        )
        analysis['confidence'] = significant_improvements / len(analysis) if analysis else 0
        
        return analysis
    
    def _determine_winner(self, analysis: Dict[str, Any], 
                         config: ABTestConfig) -> str:
        """Determine the winner of the A/B test."""
        improvements = 0
        regressions = 0
        
        for metric_name, stats in analysis.items():
            if metric_name == 'confidence':
                continue
            
            if stats.get('significant', False):
                improvement_pct = stats.get('improvement_percent', 0)
                
                # For metrics where lower is better (execution time, error rate)
                if metric_name in ['execution_time', 'error_rate', 'memory_usage']:
                    improvement_pct = -improvement_pct
                
                if improvement_pct > config.promotion_threshold * 100:
                    improvements += 1
                elif improvement_pct < -config.promotion_threshold * 100:
                    regressions += 1
        
        if improvements > regressions and improvements > 0:
            return 'improved'
        elif regressions > improvements:
            return 'original'
        else:
            return 'no_difference'
    
    def _generate_recommendation(self, winner: str, analysis: Dict[str, Any],
                               config: ABTestConfig) -> str:
        """Generate recommendation based on test results."""
        if winner == 'improved':
            improvements = []
            for metric, stats in analysis.items():
                if metric != 'confidence' and stats.get('improvement_percent', 0) != 0:
                    improvements.append(
                        f"{metric}: {stats['improvement_percent']:.1f}%"
                    )
            
            return f"APPLY improvement. Significant improvements in: {', '.join(improvements)}"
        
        elif winner == 'original':
            return "REJECT improvement. Original version performs better."
        
        else:
            return "NEEDS MORE DATA. No significant difference detected."
    
    async def _auto_promote_improvement(self, improvement: StagedImprovement):
        """Auto-promote improvement to production."""
        print(f"🚀 Auto-promoting improvement {improvement.metadata['staging_id']}")
        
        success = await self.improver.apply_staged_improvement(
            improvement.metadata['staging_id']
        )
        
        if success:
            print("✅ Improvement successfully promoted to production")
        else:
            print("❌ Failed to promote improvement")
    
    def _save_test_results(self, result: ABTestResult):
        """Save test results to file."""
        results_dir = os.path.join(self.repo_path, '.self_improver', 'ab_tests')
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(
            results_dir,
            f"{result.test_id}_results.json"
        )
        
        result_data = {
            'test_id': result.test_id,
            'staging_id': result.staging_id,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'winner': result.winner,
            'confidence': result.confidence,
            'recommendation': result.recommendation,
            'statistical_analysis': result.statistical_analysis,
            'sample_sizes': {
                'original': len(result.original_metrics.get('execution_time', [])),
                'improved': len(result.improved_metrics.get('execution_time', []))
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def _print_test_summary(self, result: ABTestResult):
        """Print test result summary."""
        print(f"\n{'='*60}")
        print(f"A/B Test Complete: {result.test_id}")
        print(f"{'='*60}")
        print(f"Winner: {result.winner.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Recommendation: {result.recommendation}")
        
        print("\nMetric Analysis:")
        for metric, stats in result.statistical_analysis.items():
            if metric == 'confidence':
                continue
            
            print(f"\n{metric}:")
            print(f"  Original: {stats['original_mean']:.2f} (±{stats['original_stdev']:.2f})")
            print(f"  Improved: {stats['improved_mean']:.2f} (±{stats['improved_stdev']:.2f})")
            print(f"  Change: {stats['improvement_percent']:+.1f}%")
            print(f"  Significant: {'Yes' if stats['significant'] else 'No'}")
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an A/B test."""
        if test_id not in self.active_tests:
            # Check completed tests
            for result in self.test_results:
                if result.test_id == test_id:
                    return {
                        'status': 'completed',
                        'winner': result.winner,
                        'confidence': result.confidence,
                        'recommendation': result.recommendation
                    }
            return None
        
        test = self.active_tests[test_id]
        elapsed = (datetime.now(timezone.utc) - test['start_time']).total_seconds()
        remaining = test['config'].duration_seconds - elapsed
        
        return {
            'status': 'running',
            'elapsed_seconds': elapsed,
            'remaining_seconds': max(0, remaining),
            'samples': {
                'original': test['request_count']['original'],
                'improved': test['request_count']['improved']
            }
        }


async def main():
    """Main entry point for A/B test runner."""
    parser = argparse.ArgumentParser(
        description="Run A/B tests for staged improvements"
    )
    parser.add_argument(
        'staging_id',
        help='Staging ID of improvement to test'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Test duration in seconds (default: 3600)'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.5,
        help='Traffic split to improved version (default: 0.5)'
    )
    parser.add_argument(
        '--auto-promote',
        action='store_true',
        help='Automatically promote if improved version wins'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ABTestConfig(
        duration_seconds=args.duration,
        traffic_split=args.split,
        auto_promote=args.auto_promote
    )
    
    # Run test
    runner = ABTestRunner()
    test_id = await runner.start_ab_test(args.staging_id, config)
    
    print(f"\nTest ID: {test_id}")
    print("Monitoring test progress...")
    
    # Monitor until complete
    while True:
        status = runner.get_test_status(test_id)
        if not status or status['status'] == 'completed':
            break
        
        print(f"\rProgress: {status['elapsed_seconds']:.0f}s elapsed, "
              f"{status['samples']['original']} original, "
              f"{status['samples']['improved']} improved samples", end='')
        
        await asyncio.sleep(5)
    
    print("\n\nTest completed!")


if __name__ == "__main__":
    asyncio.run(main())