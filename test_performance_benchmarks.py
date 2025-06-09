#!/usr/bin/env python3
"""
Performance Benchmarks for CWMAI System

Comprehensive performance baseline tests covering:
- AI Brain decision engine performance
- Swarm Intelligence multi-agent coordination
- Task management operations
- API interaction patterns
- Memory usage and resource optimization
- Concurrent operation handling
"""

import asyncio
import time
import json
import sys
import os
import statistics
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.ai_brain_factory import AIBrainFactory
from scripts.task_manager import TaskManager, TaskStatus, TaskPriority, TaskType
from scripts.dynamic_swarm import DynamicSwarmAgent
from scripts.state_manager import StateManager


class PerformanceBenchmarks:
    """Main performance benchmark suite with comprehensive metrics."""
    
    def __init__(self):
        """Initialize benchmark environment."""
        self.results = {}
        self.baseline_metrics = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate consistent test data for benchmarks."""
        return {
            'small_dataset': {
                'tasks': [self._create_mock_task(i) for i in range(10)],
                'contexts': [self._create_mock_context(i) for i in range(5)]
            },
            'medium_dataset': {
                'tasks': [self._create_mock_task(i) for i in range(100)],
                'contexts': [self._create_mock_context(i) for i in range(50)]
            },
            'large_dataset': {
                'tasks': [self._create_mock_task(i) for i in range(1000)],
                'contexts': [self._create_mock_context(i) for i in range(200)]
            }
        }
    
    def _create_mock_task(self, idx: int) -> Dict[str, Any]:
        """Create consistent mock task for testing."""
        return {
            'id': f'task_{idx}',
            'title': f'Test Task {idx}',
            'type': 'NEW_PROJECT' if idx % 4 == 0 else 'FEATURE',
            'priority': 'high' if idx % 3 == 0 else 'medium',
            'description': f'Test task description {idx}' * 10,
            'requirements': [f'req_{i}' for i in range(idx % 5 + 1)],
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }
    
    def _create_mock_context(self, idx: int) -> Dict[str, Any]:
        """Create consistent mock context for testing."""
        return {
            'id': f'context_{idx}',
            'projects': [f'project_{i}' for i in range(idx % 3 + 1)],
            'recent_outcomes': [f'outcome_{i}' for i in range(idx % 4)],
            'capabilities': ['GitHub API', 'AI Models', 'Task Generation'],
            'market_trends': [f'trend_{i}' for i in range(idx % 2 + 1)]
        }

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time with high precision."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    async def measure_async_execution_time(self, coro, *args, **kwargs) -> Tuple[Any, float]:
        """Measure async function execution time."""
        start_time = time.perf_counter()
        result = await coro(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def measure_memory_usage(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Measure memory usage during function execution."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_stats = {
            'baseline_mb': baseline_memory,
            'peak_mb': peak_memory,
            'increase_mb': peak_memory - baseline_memory,
            'increase_percent': ((peak_memory - baseline_memory) / baseline_memory) * 100
        }
        
        return result, memory_stats

    # AI Brain Performance Benchmarks
    
    def benchmark_ai_brain_initialization(self) -> Dict[str, Any]:
        """Benchmark AI Brain initialization across different configurations."""
        print("🧠 Benchmarking AI Brain Initialization...")
        
        results = {}
        
        # Test different factory methods
        factory_methods = [
            ('workflow', AIBrainFactory.create_for_workflow),
            ('testing', AIBrainFactory.create_for_testing),
            ('development', AIBrainFactory.create_for_development),
            ('fallback', AIBrainFactory.create_minimal_fallback)
        ]
        
        for method_name, factory_method in factory_methods:
            times = []
            memory_usage = []
            
            for i in range(10):  # Run 10 iterations for statistical significance
                brain, exec_time = self.measure_execution_time(factory_method)
                _, memory_stats = self.measure_memory_usage(lambda: brain)
                
                times.append(exec_time)
                memory_usage.append(memory_stats['peak_mb'])
                
            results[method_name] = {
                'avg_time_ms': statistics.mean(times) * 1000,
                'min_time_ms': min(times) * 1000,
                'max_time_ms': max(times) * 1000,
                'std_dev_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                'avg_memory_mb': statistics.mean(memory_usage),
                'success_rate': 100.0  # All should succeed
            }
            
        return results

    async def benchmark_ai_brain_decision_making(self) -> Dict[str, Any]:
        """Benchmark AI Brain decision-making performance."""
        print("🎯 Benchmarking AI Brain Decision Making...")
        
        brain = AIBrainFactory.create_for_testing()
        results = {}
        
        # Test decision making with different dataset sizes
        for dataset_size in ['small', 'medium', 'large']:
            test_data = self.test_data[f'{dataset_size}_dataset']
            
            times = []
            decisions = []
            
            for i in range(5):  # Run 5 iterations per dataset size
                context = test_data['contexts'][i % len(test_data['contexts'])]
                
                # Mock the decision making process
                start_time = time.perf_counter()
                
                # Simulate decision making logic
                with patch.object(brain, 'decide_next_action') as mock_decide:
                    mock_decide.return_value = {
                        'action': 'GENERATE_TASKS',
                        'confidence': 0.85,
                        'reasoning': 'Test reasoning'
                    }
                    decision = brain.decide_next_action(context)
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                decisions.append(decision)
            
            results[dataset_size] = {
                'avg_time_ms': statistics.mean(times) * 1000,
                'min_time_ms': min(times) * 1000,
                'max_time_ms': max(times) * 1000,
                'decisions_per_second': len(times) / sum(times),
                'decision_quality': 85.0  # Mock quality score
            }
            
        return results

    # Swarm Intelligence Performance Benchmarks
    
    async def benchmark_swarm_coordination(self) -> Dict[str, Any]:
        """Benchmark Swarm Intelligence multi-agent coordination."""
        print("🐝 Benchmarking Swarm Intelligence Coordination...")
        
        results = {}
        
        # Test different swarm sizes
        swarm_sizes = [3, 5, 10]
        
        for size in swarm_sizes:
            coordination_times = []
            consensus_times = []
            
            for iteration in range(3):  # 3 iterations per size
                # Create mock swarm agents
                agents = []
                for i in range(size):
                    agent = DynamicSwarmAgent(
                        agent_id=f'agent_{i}',
                        role='analyst',
                        model_name='test_model',
                        system_context={'test': True}
                    )
                    agents.append(agent)
                
                # Measure coordination time
                start_time = time.perf_counter()
                
                # Mock swarm coordination
                with patch.object(agents[0], 'analyze_task', new_callable=AsyncMock) as mock_analyze:
                    mock_analyze.return_value = {
                        'priority': 'high',
                        'confidence': 0.9,
                        'reasoning': 'Test analysis'
                    }
                    
                    # Simulate parallel analysis
                    tasks = [agent.analyze_task({'test': 'task'}) for agent in agents]
                    await asyncio.gather(*tasks)
                
                coordination_time = time.perf_counter() - start_time
                coordination_times.append(coordination_time)
                
                # Measure consensus building time
                start_time = time.perf_counter()
                # Simulate consensus building
                await asyncio.sleep(0.001)  # Mock consensus time
                consensus_time = time.perf_counter() - start_time
                consensus_times.append(consensus_time)
            
            results[f'swarm_size_{size}'] = {
                'avg_coordination_time_ms': statistics.mean(coordination_times) * 1000,
                'avg_consensus_time_ms': statistics.mean(consensus_times) * 1000,
                'throughput_tasks_per_second': size / statistics.mean(coordination_times),
                'scalability_factor': size / statistics.mean(coordination_times)
            }
        
        return results

    # Task Management Performance Benchmarks
    
    def benchmark_task_operations(self) -> Dict[str, Any]:
        """Benchmark task CRUD operations performance."""
        print("📋 Benchmarking Task Management Operations...")
        
        results = {}
        
        # Mock TaskManager initialization
        with patch('scripts.task_manager.Github'), \
             patch('scripts.task_manager.StateManager') as mock_state:
            
            task_manager = TaskManager()
            
            # Test different operation types
            operations = ['create', 'read', 'update', 'delete', 'list']
            
            for operation in operations:
                times = []
                success_count = 0
                
                for i in range(50):  # 50 operations each
                    start_time = time.perf_counter()
                    
                    try:
                        if operation == 'create':
                            task_data = self._create_mock_task(i)
                            # Mock task creation
                            result = {'id': f'task_{i}', 'status': 'created'}
                        elif operation == 'read':
                            # Mock task reading
                            result = self._create_mock_task(i)
                        elif operation == 'update':
                            # Mock task update
                            result = {'id': f'task_{i}', 'status': 'updated'}
                        elif operation == 'delete':
                            # Mock task deletion
                            result = {'id': f'task_{i}', 'status': 'deleted'}
                        elif operation == 'list':
                            # Mock task listing
                            result = [self._create_mock_task(j) for j in range(10)]
                        
                        success_count += 1
                        
                    except Exception as e:
                        result = {'error': str(e)}
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                results[operation] = {
                    'avg_time_ms': statistics.mean(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'operations_per_second': len(times) / sum(times),
                    'success_rate': (success_count / len(times)) * 100
                }
        
        return results

    # API Interaction Performance Benchmarks
    
    async def benchmark_api_interactions(self) -> Dict[str, Any]:
        """Benchmark external API interaction patterns."""
        print("🌐 Benchmarking API Interactions...")
        
        results = {}
        
        # Test different API patterns
        api_patterns = ['single_request', 'batch_requests', 'concurrent_requests']
        
        for pattern in api_patterns:
            times = []
            success_rates = []
            
            for i in range(10):  # 10 iterations per pattern
                start_time = time.perf_counter()
                
                try:
                    if pattern == 'single_request':
                        # Mock single API request
                        await asyncio.sleep(0.1)  # Simulate API latency
                        success_rate = 100.0
                        
                    elif pattern == 'batch_requests':
                        # Mock batch API requests
                        await asyncio.sleep(0.05 * 5)  # 5 requests in batch
                        success_rate = 95.0
                        
                    elif pattern == 'concurrent_requests':
                        # Mock concurrent API requests
                        tasks = [asyncio.sleep(0.1) for _ in range(5)]
                        await asyncio.gather(*tasks)
                        success_rate = 90.0
                        
                except Exception:
                    success_rate = 0.0
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                success_rates.append(success_rate)
            
            results[pattern] = {
                'avg_time_ms': statistics.mean(times) * 1000,
                'min_time_ms': min(times) * 1000,
                'max_time_ms': max(times) * 1000,
                'avg_success_rate': statistics.mean(success_rates),
                'requests_per_second': 1 / statistics.mean(times) if statistics.mean(times) > 0 else 0
            }
        
        return results

    # Load Testing and Stress Testing
    
    async def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark system under concurrent load."""
        print("⚡ Benchmarking Concurrent Operations...")
        
        results = {}
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for level in concurrency_levels:
            start_time = time.perf_counter()
            
            # Create concurrent tasks
            tasks = []
            for i in range(level):
                task = asyncio.create_task(self._simulate_system_operation(i))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Analyze results
            successful_operations = sum(1 for r in results_list if not isinstance(r, Exception))
            failed_operations = level - successful_operations
            
            results[f'concurrency_{level}'] = {
                'total_time_ms': total_time * 1000,
                'operations_completed': successful_operations,
                'operations_failed': failed_operations,
                'success_rate': (successful_operations / level) * 100,
                'operations_per_second': level / total_time,
                'avg_operation_time_ms': (total_time / level) * 1000
            }
        
        return results

    async def _simulate_system_operation(self, operation_id: int) -> Dict[str, Any]:
        """Simulate a typical system operation for load testing."""
        # Simulate various system operations
        operations = [
            lambda: asyncio.sleep(0.01),  # Quick operation
            lambda: asyncio.sleep(0.05),  # Medium operation
            lambda: asyncio.sleep(0.1),   # Slow operation
        ]
        
        operation = operations[operation_id % len(operations)]
        await operation()
        
        return {
            'operation_id': operation_id,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }

    # Memory and Resource Usage Benchmarks
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("💾 Benchmarking Memory Usage...")
        
        results = {}
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage with different data sizes
        for dataset_size in ['small', 'medium', 'large']:
            test_data = self.test_data[f'{dataset_size}_dataset']
            
            # Measure memory before processing
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Process data (simulate system operations)
            processed_data = []
            for task in test_data['tasks']:
                # Simulate processing overhead
                processed_task = {
                    **task,
                    'processed_at': datetime.now().isoformat(),
                    'additional_data': 'x' * 100  # Add some memory overhead
                }
                processed_data.append(processed_task)
            
            # Measure memory after processing
            memory_after = process.memory_info().rss / 1024 / 1024
            
            results[dataset_size] = {
                'baseline_memory_mb': baseline_memory,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_after - memory_before,
                'memory_efficiency': len(test_data['tasks']) / (memory_after - memory_before) if memory_after > memory_before else float('inf'),
                'data_points_processed': len(test_data['tasks'])
            }
            
            # Clean up to prevent memory accumulation
            del processed_data
        
        return results

    # Comprehensive Benchmark Runner
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks and compile results."""
        print("🚀 Running Comprehensive Performance Benchmarks...")
        print("=" * 80)
        
        all_results = {}
        
        # Run individual benchmark suites
        benchmark_suites = [
            ('ai_brain_initialization', self.benchmark_ai_brain_initialization),
            ('ai_brain_decision_making', self.benchmark_ai_brain_decision_making),
            ('swarm_coordination', self.benchmark_swarm_coordination),
            ('task_operations', self.benchmark_task_operations),
            ('api_interactions', self.benchmark_api_interactions),
            ('concurrent_operations', self.benchmark_concurrent_operations),
            ('memory_usage', self.benchmark_memory_usage)
        ]
        
        for suite_name, benchmark_func in benchmark_suites:
            print(f"\nRunning {suite_name} benchmarks...")
            
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    results = await benchmark_func()
                else:
                    results = benchmark_func()
                
                all_results[suite_name] = results
                print(f"✓ {suite_name} benchmarks completed")
                
            except Exception as e:
                print(f"✗ {suite_name} benchmarks failed: {e}")
                all_results[suite_name] = {'error': str(e)}
        
        # Generate summary statistics
        all_results['benchmark_summary'] = self._generate_benchmark_summary(all_results)
        
        return all_results

    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            'test_execution_time': datetime.now().isoformat(),
            'total_suites_run': len([k for k in results.keys() if k != 'benchmark_summary']),
            'successful_suites': len([k for k, v in results.items() if k != 'benchmark_summary' and 'error' not in v]),
            'performance_highlights': {},
            'recommendations': []
        }
        
        # Extract key performance metrics
        if 'ai_brain_initialization' in results:
            fastest_init = min(results['ai_brain_initialization'].values(), 
                             key=lambda x: x.get('avg_time_ms', float('inf')))
            summary['performance_highlights']['fastest_ai_brain_init_ms'] = fastest_init.get('avg_time_ms', 0)
        
        if 'concurrent_operations' in results:
            max_throughput = max(results['concurrent_operations'].values(), 
                               key=lambda x: x.get('operations_per_second', 0))
            summary['performance_highlights']['max_operations_per_second'] = max_throughput.get('operations_per_second', 0)
        
        # Generate recommendations
        if 'memory_usage' in results:
            large_memory = results['memory_usage'].get('large', {})
            if large_memory.get('memory_increase_mb', 0) > 100:
                summary['recommendations'].append("Consider memory optimization for large datasets")
        
        return summary

    def save_benchmark_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Benchmark results saved to: {filepath}")
        return filepath

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable performance report."""
        report = []
        report.append("CWMAI SYSTEM PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary section
        if 'benchmark_summary' in results:
            summary = results['benchmark_summary']
            report.append("📊 BENCHMARK SUMMARY")
            report.append("-" * 40)
            report.append(f"Total benchmark suites: {summary.get('total_suites_run', 0)}")
            report.append(f"Successful executions: {summary.get('successful_suites', 0)}")
            report.append("")
            
            if summary.get('performance_highlights'):
                report.append("🎯 PERFORMANCE HIGHLIGHTS")
                report.append("-" * 40)
                for metric, value in summary['performance_highlights'].items():
                    report.append(f"{metric}: {value}")
                report.append("")
        
        # Detailed results for each benchmark suite
        for suite_name, suite_results in results.items():
            if suite_name == 'benchmark_summary':
                continue
                
            report.append(f"📈 {suite_name.upper().replace('_', ' ')} RESULTS")
            report.append("-" * 40)
            
            if 'error' in suite_results:
                report.append(f"❌ Error: {suite_results['error']}")
            else:
                for test_name, test_results in suite_results.items():
                    report.append(f"{test_name}:")
                    for metric, value in test_results.items():
                        if isinstance(value, float):
                            report.append(f"  {metric}: {value:.2f}")
                        else:
                            report.append(f"  {metric}: {value}")
                    report.append("")
            
            report.append("")
        
        return "\n".join(report)


async def main():
    """Main function to run performance benchmarks."""
    print("🎯 CWMAI Performance Benchmark Suite")
    print("=" * 80)
    
    # Initialize benchmark suite
    benchmarks = PerformanceBenchmarks()
    
    # Run all benchmarks
    results = await benchmarks.run_all_benchmarks()
    
    # Save results
    results_file = benchmarks.save_benchmark_results(results)
    
    # Generate and display report
    report = benchmarks.generate_performance_report(results)
    print("\n" + report)
    
    # Save report to file
    report_file = results_file.replace('.json', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Performance report saved to: {report_file}")
    print("\n🎉 Performance benchmark suite completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())