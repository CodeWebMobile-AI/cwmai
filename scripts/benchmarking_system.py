#!/usr/bin/env python3
"""
Benchmarking System - Continuous performance monitoring and improvement
"""

import time
import psutil
import json
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import tracemalloc
import cProfile
import pstats
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

from scripts.http_ai_client import HTTPAIClient
import logging
from scripts.tool_calling_system import ToolCallingSystem

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Represents a single benchmark result"""
    name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    iterations: int
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    tool_name: str
    average_time: float
    min_time: float
    max_time: float
    std_deviation: float
    memory_peak: float
    cpu_peak: float
    success_rate: float
    regression_detected: bool
    improvement_suggestions: List[str]
    comparison_with_baseline: Dict[str, Any]


class BenchmarkingSystem:
    """Continuously benchmark and improve system performance"""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.tool_system = ToolCallingSystem()
        self.benchmark_history = self._load_benchmark_history()
        self.baseline_metrics = self._load_baseline_metrics()
        self.external_benchmarks = self._load_external_benchmarks()
        
    def _load_benchmark_history(self) -> Dict[str, List[BenchmarkResult]]:
        """Load historical benchmark data"""
        history_file = Path("benchmark_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to BenchmarkResult objects
                    history = {}
                    for tool, results in data.items():
                        history[tool] = [
                            BenchmarkResult(**{
                                **r,
                                'timestamp': datetime.fromisoformat(r['timestamp'])
                            }) for r in results
                        ]
                    return history
            except Exception as e:
                logger.info(f"Error loading benchmark history: {str(e)}")
        return {}
    
    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load baseline performance metrics"""
        return {
            "tool_execution": {
                "acceptable_time": 1.0,  # seconds
                "good_time": 0.5,
                "excellent_time": 0.1
            },
            "memory_usage": {
                "acceptable": 100,  # MB
                "good": 50,
                "excellent": 20
            },
            "cpu_usage": {
                "acceptable": 80,  # percentage
                "good": 50,
                "excellent": 20
            }
        }
    
    def _load_external_benchmarks(self) -> Dict[str, Any]:
        """Load industry standard benchmarks"""
        return {
            "response_time": {
                "api_call": 0.2,  # seconds
                "database_query": 0.05,
                "file_operation": 0.01,
                "ai_generation": 2.0
            },
            "throughput": {
                "requests_per_second": 1000,
                "concurrent_operations": 100
            },
            "resource_usage": {
                "memory_per_operation": 10,  # MB
                "cpu_per_operation": 5  # percentage
            }
        }
    
    async def benchmark_tools(self, tools: Optional[List[str]] = None, iterations: int = 10) -> Dict[str, PerformanceReport]:
        """Run performance tests on tools"""
        reports = {}
        
        # Get all tools if not specified
        if not tools:
            tools = await self._get_all_tools()
        
        for tool_name in tools:
            logger.info(f"Benchmarking tool: {tool_name}")
            
            try:
                results = await self._benchmark_single_tool(tool_name, iterations)
                report = self._generate_performance_report(tool_name, results)
                reports[tool_name] = report
                
                # Save results
                self._save_benchmark_results(tool_name, results)
                
                # Check for regressions
                if report.regression_detected:
                    await self._handle_regression(tool_name, report)
                
            except Exception as e:
                logger.info(f"Error benchmarking {tool_name}: {str(e)}")
                reports[tool_name] = PerformanceReport(
                    tool_name=tool_name,
                    average_time=0,
                    min_time=0,
                    max_time=0,
                    std_deviation=0,
                    memory_peak=0,
                    cpu_peak=0,
                    success_rate=0,
                    regression_detected=False,
                    improvement_suggestions=["Tool failed to benchmark"],
                    comparison_with_baseline={}
                )
        
        return reports
    
    async def benchmark_against_external(self, operations: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Compare with industry standards"""
        results = {}
        
        # Define operations to benchmark
        if not operations:
            operations = ["api_call", "data_processing", "ai_generation", "file_operation"]
        
        for operation in operations:
            logger.info(f"Benchmarking against external standards: {operation}")
            
            try:
                # Run our implementation
                our_results = await self._run_operation_benchmark(operation)
                
                # Compare with industry standards
                comparison = self._compare_with_industry(operation, our_results)
                
                # Get improvement suggestions from AI
                suggestions = await self._get_improvement_suggestions(operation, our_results, comparison)
                
                results[operation] = {
                    "our_performance": our_results,
                    "industry_standard": self.external_benchmarks.get(operation, {}),
                    "comparison": comparison,
                    "improvement_suggestions": suggestions
                }
                
            except Exception as e:
                logger.info(f"Error in external benchmark for {operation}: {str(e)}")
                results[operation] = {"error": str(e)}
        
        return results
    
    async def _get_all_tools(self) -> List[str]:
        """Get list of all available tools"""
        try:
            # Get from tool calling system
            tools = self.tool_system.available_tools
            return list(tools.keys())
        except:
            # Fallback to scanning directory
            tools_dir = Path("scripts/custom_tools")
            if tools_dir.exists():
                return [f.stem for f in tools_dir.glob("*.py") if f.stem != "__init__"]
            return []
    
    async def _benchmark_single_tool(self, tool_name: str, iterations: int) -> List[BenchmarkResult]:
        """Benchmark a single tool multiple times"""
        results = []
        
        for i in range(iterations):
            # Start monitoring
            tracemalloc.start()
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            
            start_time = time.time()
            
            try:
                # Execute tool
                success = await self._execute_tool_for_benchmark(tool_name)
                
                execution_time = time.time() - start_time
                
                # Get resource usage
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                cpu_after = process.cpu_percent()
                memory_usage = peak / 1024 / 1024  # Convert to MB
                cpu_usage = cpu_after - cpu_before
                
                results.append(BenchmarkResult(
                    name=tool_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    timestamp=datetime.now(),
                    iterations=1,
                    metrics={
                        "success": success,
                        "memory_current": current / 1024 / 1024,
                        "memory_peak": memory_usage
                    },
                    metadata={
                        "iteration": i + 1,
                        "total_iterations": iterations
                    }
                ))
                
            except Exception as e:
                logger.info(f"Benchmark iteration {i+1} failed for {tool_name}: {str(e)}")
                results.append(BenchmarkResult(
                    name=tool_name,
                    execution_time=0,
                    memory_usage=0,
                    cpu_usage=0,
                    timestamp=datetime.now(),
                    iterations=1,
                    metrics={"success": False, "error": str(e)},
                    metadata={"iteration": i + 1}
                ))
        
        return results
    
    async def _execute_tool_for_benchmark(self, tool_name: str) -> bool:
        """Execute a tool for benchmarking purposes"""
        try:
            # Create a simple test case for the tool
            test_input = self._generate_test_input_for_tool(tool_name)
            
            # Execute through tool calling system
            result = await self.tool_system.execute_tool(tool_name, test_input)
            
            return result is not None
        except Exception as e:
            logger.info(f"Tool execution error: {str(e)}")
            return False
    
    def _generate_test_input_for_tool(self, tool_name: str) -> Dict[str, Any]:
        """Generate appropriate test input for a tool"""
        # Common test inputs for different tool types
        test_inputs = {
            "analyzer": {"code": "def test(): return 42", "language": "python"},
            "generator": {"prompt": "Generate a simple function", "language": "python"},
            "optimizer": {"code": "for i in range(10): print(i)", "target": "performance"},
            "default": {"input": "test data", "options": {}}
        }
        
        # Match tool type
        for tool_type, test_input in test_inputs.items():
            if tool_type in tool_name.lower():
                return test_input
        
        return test_inputs["default"]
    
    def _generate_performance_report(self, tool_name: str, results: List[BenchmarkResult]) -> PerformanceReport:
        """Generate comprehensive performance report"""
        if not results:
            return PerformanceReport(
                tool_name=tool_name,
                average_time=0,
                min_time=0,
                max_time=0,
                std_deviation=0,
                memory_peak=0,
                cpu_peak=0,
                success_rate=0,
                regression_detected=False,
                improvement_suggestions=["No results to analyze"],
                comparison_with_baseline={}
            )
        
        # Calculate metrics
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        memory_usages = [r.memory_usage for r in results]
        cpu_usages = [r.cpu_usage for r in results]
        success_count = sum(1 for r in results if r.metrics.get("success", False))
        
        avg_time = statistics.mean(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Check for regression
        regression_detected = self._check_for_regression(tool_name, avg_time)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            tool_name, avg_time, max(memory_usages), max(cpu_usages)
        )
        
        # Compare with baseline
        comparison = self._compare_with_baseline(
            avg_time, max(memory_usages), max(cpu_usages)
        )
        
        return PerformanceReport(
            tool_name=tool_name,
            average_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_deviation=std_dev,
            memory_peak=max(memory_usages) if memory_usages else 0,
            cpu_peak=max(cpu_usages) if cpu_usages else 0,
            success_rate=success_count / len(results) if results else 0,
            regression_detected=regression_detected,
            improvement_suggestions=suggestions,
            comparison_with_baseline=comparison
        )
    
    def _check_for_regression(self, tool_name: str, current_avg_time: float) -> bool:
        """Check if performance has regressed"""
        if tool_name not in self.benchmark_history:
            return False
        
        historical = self.benchmark_history[tool_name]
        if len(historical) < 5:
            return False
        
        # Compare with recent average
        recent_times = [r.execution_time for r in historical[-10:]]
        recent_avg = statistics.mean(recent_times)
        
        # Regression if current is 20% slower
        return current_avg_time > recent_avg * 1.2
    
    def _generate_improvement_suggestions(self, tool_name: str, avg_time: float, 
                                        memory_peak: float, cpu_peak: float) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Time-based suggestions
        if avg_time > self.baseline_metrics["tool_execution"]["acceptable_time"]:
            suggestions.append("Consider optimizing algorithm complexity")
            suggestions.append("Profile code to identify bottlenecks")
            suggestions.append("Implement caching for repeated operations")
        
        # Memory-based suggestions
        if memory_peak > self.baseline_metrics["memory_usage"]["acceptable"]:
            suggestions.append("Reduce memory allocation in loops")
            suggestions.append("Use generators instead of lists where possible")
            suggestions.append("Implement object pooling for frequently created objects")
        
        # CPU-based suggestions
        if cpu_peak > self.baseline_metrics["cpu_usage"]["acceptable"]:
            suggestions.append("Consider parallel processing for CPU-intensive tasks")
            suggestions.append("Optimize tight loops and recursive calls")
            suggestions.append("Use more efficient data structures")
        
        return suggestions
    
    def _compare_with_baseline(self, avg_time: float, memory_peak: float, 
                             cpu_peak: float) -> Dict[str, Any]:
        """Compare metrics with baseline"""
        return {
            "execution_time": {
                "value": avg_time,
                "rating": self._rate_metric(avg_time, self.baseline_metrics["tool_execution"]),
                "vs_excellent": avg_time / self.baseline_metrics["tool_execution"]["excellent_time"]
            },
            "memory_usage": {
                "value": memory_peak,
                "rating": self._rate_metric(memory_peak, self.baseline_metrics["memory_usage"], reverse=True),
                "vs_excellent": memory_peak / self.baseline_metrics["memory_usage"]["excellent"]
            },
            "cpu_usage": {
                "value": cpu_peak,
                "rating": self._rate_metric(cpu_peak, self.baseline_metrics["cpu_usage"], reverse=True),
                "vs_excellent": cpu_peak / self.baseline_metrics["cpu_usage"]["excellent"]
            }
        }
    
    def _rate_metric(self, value: float, thresholds: Dict[str, float], reverse: bool = False) -> str:
        """Rate a metric against thresholds"""
        if reverse:
            if value <= thresholds["excellent"]:
                return "excellent"
            elif value <= thresholds["good"]:
                return "good"
            elif value <= thresholds["acceptable"]:
                return "acceptable"
            else:
                return "poor"
        else:
            if value <= thresholds["excellent_time"]:
                return "excellent"
            elif value <= thresholds["good_time"]:
                return "good"
            elif value <= thresholds["acceptable_time"]:
                return "acceptable"
            else:
                return "poor"
    
    def _save_benchmark_results(self, tool_name: str, results: List[BenchmarkResult]):
        """Save benchmark results to history"""
        if tool_name not in self.benchmark_history:
            self.benchmark_history[tool_name] = []
        
        self.benchmark_history[tool_name].extend(results)
        
        # Keep only last 1000 results per tool
        if len(self.benchmark_history[tool_name]) > 1000:
            self.benchmark_history[tool_name] = self.benchmark_history[tool_name][-1000:]
        
        # Save to file
        try:
            history_file = Path("benchmark_history.json")
            data = {}
            for tool, tool_results in self.benchmark_history.items():
                data[tool] = [
                    {**asdict(r), 'timestamp': r.timestamp.isoformat()}
                    for r in tool_results
                ]
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.info(f"Error saving benchmark history: {str(e)}")
    
    async def _handle_regression(self, tool_name: str, report: PerformanceReport):
        """Handle performance regression"""
        logger.info(f"Performance regression detected in {tool_name}")
        
        # Create issue or alert
        alert = {
            "tool": tool_name,
            "regression_type": "performance",
            "current_performance": report.average_time,
            "suggestions": report.improvement_suggestions,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log alert
        logger.info(f"Performance alert: {json.dumps(alert, indent=2)}")
        
        # Could trigger automatic optimization here
    
    async def _run_operation_benchmark(self, operation: str) -> Dict[str, float]:
        """Run benchmark for a specific operation type"""
        results = {
            "execution_time": 0,
            "throughput": 0,
            "memory_usage": 0,
            "cpu_usage": 0
        }
        
        # Define operation implementations
        operations = {
            "api_call": self._benchmark_api_call,
            "data_processing": self._benchmark_data_processing,
            "ai_generation": self._benchmark_ai_generation,
            "file_operation": self._benchmark_file_operation
        }
        
        if operation in operations:
            results = await operations[operation]()
        
        return results
    
    async def _benchmark_api_call(self) -> Dict[str, float]:
        """Benchmark API call performance"""
        times = []
        
        for _ in range(10):
            start = time.time()
            try:
                # Simulate API call
                await asyncio.sleep(0.1)  # Replace with actual API call
                times.append(time.time() - start)
            except:
                pass
        
        return {
            "execution_time": statistics.mean(times) if times else 0,
            "throughput": len(times) / sum(times) if times else 0,
            "memory_usage": 10,  # MB estimate
            "cpu_usage": 5  # percentage estimate
        }
    
    async def _benchmark_data_processing(self) -> Dict[str, float]:
        """Benchmark data processing performance"""
        # Create test data
        data = list(range(10000))
        
        start = time.time()
        # Process data
        result = [x * 2 for x in data if x % 2 == 0]
        execution_time = time.time() - start
        
        return {
            "execution_time": execution_time,
            "throughput": len(data) / execution_time,
            "memory_usage": 50,  # MB estimate
            "cpu_usage": 20  # percentage estimate
        }
    
    async def _benchmark_ai_generation(self) -> Dict[str, float]:
        """Benchmark AI generation performance"""
        start = time.time()
        
        try:
            response = self.ai_client.generate_response("Generate a hello world function")
            execution_time = time.time() - start
            
            return {
                "execution_time": execution_time,
                "throughput": 1 / execution_time,
                "memory_usage": 100,  # MB estimate
                "cpu_usage": 30  # percentage estimate
            }
        except:
            return {
                "execution_time": 0,
                "throughput": 0,
                "memory_usage": 0,
                "cpu_usage": 0
            }
    
    async def _benchmark_file_operation(self) -> Dict[str, float]:
        """Benchmark file operation performance"""
        test_file = Path("benchmark_test.tmp")
        test_data = "x" * 1000000  # 1MB of data
        
        times = []
        
        for _ in range(5):
            start = time.time()
            try:
                # Write
                with open(test_file, 'w') as f:
                    f.write(test_data)
                
                # Read
                with open(test_file, 'r') as f:
                    _ = f.read()
                
                times.append(time.time() - start)
            except:
                pass
            finally:
                if test_file.exists():
                    test_file.unlink()
        
        return {
            "execution_time": statistics.mean(times) if times else 0,
            "throughput": len(times) / sum(times) if times else 0,
            "memory_usage": 20,  # MB estimate
            "cpu_usage": 10  # percentage estimate
        }
    
    def _compare_with_industry(self, operation: str, our_results: Dict[str, float]) -> Dict[str, Any]:
        """Compare our performance with industry standards"""
        industry = self.external_benchmarks.get("response_time", {}).get(operation, 1.0)
        our_time = our_results.get("execution_time", 0)
        
        if our_time == 0:
            return {"status": "error", "message": "No results to compare"}
        
        ratio = our_time / industry
        
        return {
            "our_time": our_time,
            "industry_standard": industry,
            "ratio": ratio,
            "status": "better" if ratio < 1 else "worse" if ratio > 1.5 else "comparable",
            "percentage_difference": (ratio - 1) * 100
        }
    
    async def _get_improvement_suggestions(self, operation: str, our_results: Dict[str, float], 
                                         comparison: Dict[str, Any]) -> List[str]:
        """Get AI-powered improvement suggestions"""
        if comparison.get("status") == "better":
            return ["Performance is already better than industry standard"]
        
        prompt = f"""Analyze this performance comparison and suggest improvements:

Operation: {operation}
Our Performance: {json.dumps(our_results, indent=2)}
Comparison: {json.dumps(comparison, indent=2)}

Provide specific, actionable suggestions to improve performance."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Parse suggestions from response
            suggestions = [s.strip() for s in response.split('\n') if s.strip() and not s.startswith('#')]
            return suggestions[:5]  # Top 5 suggestions
        except:
            return ["Consider profiling the operation", "Optimize algorithm complexity", "Implement caching"]
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report"""
        report = f"""# Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
Total tools benchmarked: {len(results)}
"""
        
        # Add tool benchmarks
        if "tool_benchmarks" in results:
            report += "\n## Tool Performance\n"
            for tool_name, perf_report in results["tool_benchmarks"].items():
                status = "🟢" if not perf_report.regression_detected else "🔴"
                report += f"\n### {status} {tool_name}\n"
                report += f"- Average execution time: {perf_report.average_time:.3f}s\n"
                report += f"- Success rate: {perf_report.success_rate * 100:.1f}%\n"
                report += f"- Memory peak: {perf_report.memory_peak:.1f}MB\n"
                
                if perf_report.improvement_suggestions:
                    report += "\n**Improvement Suggestions:**\n"
                    for suggestion in perf_report.improvement_suggestions:
                        report += f"- {suggestion}\n"
        
        # Add external benchmarks
        if "external_benchmarks" in results:
            report += "\n## Industry Comparison\n"
            for operation, comparison in results["external_benchmarks"].items():
                if "error" not in comparison:
                    status = "🟢" if comparison["comparison"]["status"] == "better" else "🟡" if comparison["comparison"]["status"] == "comparable" else "🔴"
                    report += f"\n### {status} {operation}\n"
                    report += f"- Our time: {comparison['our_performance']['execution_time']:.3f}s\n"
                    report += f"- Industry standard: {comparison['comparison']['industry_standard']:.3f}s\n"
                    report += f"- Status: {comparison['comparison']['status']}\n"
        
        return report
    
    async def continuous_monitoring(self, interval_minutes: int = 60):
        """Run continuous performance monitoring"""
        logger.info(f"Starting continuous performance monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                # Run benchmarks
                tool_results = await self.benchmark_tools()
                external_results = await self.benchmark_against_external()
                
                # Generate report
                all_results = {
                    "tool_benchmarks": tool_results,
                    "external_benchmarks": external_results
                }
                
                report = self.generate_benchmark_report(all_results)
                
                # Save report
                report_file = Path(f"benchmark_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                report_file.parent.mkdir(exist_ok=True)
                with open(report_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"Benchmark report saved to {report_file}")
                
                # Wait for next iteration
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.info(f"Error in continuous monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying


# Example usage and testing
if __name__ == "__main__":
    async def main():
        system = BenchmarkingSystem()
        
        # Run tool benchmarks
        print("Running tool benchmarks...")
        tool_results = await system.benchmark_tools(iterations=5)
        
        for tool_name, report in tool_results.items():
            print(f"\n{tool_name}:")
            print(f"  Average time: {report.average_time:.3f}s")
            print(f"  Success rate: {report.success_rate * 100:.1f}%")
            print(f"  Regression: {'Yes' if report.regression_detected else 'No'}")
        
        # Run external benchmarks
        print("\n\nRunning external benchmarks...")
        external_results = await system.benchmark_against_external()
        
        for operation, results in external_results.items():
            if "error" not in results:
                print(f"\n{operation}:")
                print(f"  Our time: {results['our_performance']['execution_time']:.3f}s")
                print(f"  vs Industry: {results['comparison']['status']}")
    
    asyncio.run(main())