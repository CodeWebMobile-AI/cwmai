"""
Performance Benchmarks for API Rate Limiting System

Comprehensive performance testing and benchmarking for rate limiting components.
"""

import asyncio
import json
import os
import statistics
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
import logging

try:
    from rate_limiter import RateLimiter, RateLimitTier, RateLimitStrategy
    from http_ai_client import HTTPAIClient
    from rate_limit_monitor import RateLimitMonitor
except ImportError as e:
    print(f"Import error: {e}")
    print("Rate limiting modules not available")
    exit(1)


class RateLimitBenchmark:
    """Comprehensive benchmarking suite for rate limiting system."""
    
    def __init__(self, redis_url: str = None):
        """Initialize benchmark suite.
        
        Args:
            redis_url: Redis URL for testing (uses test Redis if None)
        """
        self.redis_url = redis_url or "redis://localhost:9999/0"  # Test Redis
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests.
        
        Returns:
            Complete benchmark results
        """
        print("🚀 Starting Rate Limiting Performance Benchmarks")
        print("=" * 60)
        
        self.results = {
            "benchmark_start": datetime.now(timezone.utc).isoformat(),
            "environment": self._get_environment_info(),
            "tests": {}
        }
        
        # Run individual benchmark tests
        test_methods = [
            ("single_client_throughput", self._benchmark_single_client_throughput),
            ("multi_client_concurrent", self._benchmark_multi_client_concurrent),
            ("different_strategies", self._benchmark_different_strategies),
            ("fallback_mode_performance", self._benchmark_fallback_mode),
            ("memory_usage", self._benchmark_memory_usage),
            ("latency_distribution", self._benchmark_latency_distribution),
            ("scalability_test", self._benchmark_scalability),
            ("redis_vs_fallback", self._benchmark_redis_vs_fallback)
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n📊 Running {test_name.replace('_', ' ').title()}...")
            try:
                start_time = time.time()
                result = test_method()
                duration = time.time() - start_time
                
                result["test_duration_seconds"] = duration
                self.results["tests"][test_name] = result
                
                print(f"✅ Completed in {duration:.2f}s")
                self._print_test_summary(test_name, result)
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                self.results["tests"][test_name] = {
                    "error": str(e),
                    "test_duration_seconds": 0
                }
        
        self.results["benchmark_end"] = datetime.now(timezone.utc).isoformat()
        self.results["total_duration"] = time.time() - time.mktime(
            datetime.fromisoformat(self.results["benchmark_start"].replace("Z", "+00:00")).timetuple()
        )
        
        print("\n" + "=" * 60)
        print("🏁 Benchmark Complete")
        self._print_overall_summary()
        
        return self.results
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for benchmark context."""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "redis_url": self.redis_url,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _benchmark_single_client_throughput(self) -> Dict[str, Any]:
        """Benchmark single client throughput."""
        rate_limiter = RateLimiter(self.redis_url)
        client_id = "benchmark_single_client"
        num_requests = 1000
        
        try:
            start_time = time.time()
            allowed_count = 0
            blocked_count = 0
            response_times = []
            
            for i in range(num_requests):
                request_start = time.time()
                result = rate_limiter.check_rate_limit(client_id, RateLimitTier.PREMIUM)
                request_duration = time.time() - request_start
                
                response_times.append(request_duration * 1000)  # Convert to ms
                
                if result.allowed:
                    allowed_count += 1
                else:
                    blocked_count += 1
            
            total_duration = time.time() - start_time
            
            return {
                "total_requests": num_requests,
                "allowed_requests": allowed_count,
                "blocked_requests": blocked_count,
                "total_duration_seconds": total_duration,
                "requests_per_second": num_requests / total_duration,
                "average_response_time_ms": statistics.mean(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "p95_response_time_ms": self._percentile(response_times, 95),
                "p99_response_time_ms": self._percentile(response_times, 99),
                "min_response_time_ms": min(response_times),
                "max_response_time_ms": max(response_times)
            }
        finally:
            rate_limiter.close()
    
    def _benchmark_multi_client_concurrent(self) -> Dict[str, Any]:
        """Benchmark multiple clients making concurrent requests."""
        num_clients = 50
        requests_per_client = 20
        
        def client_worker(client_id: str) -> Dict[str, Any]:
            rate_limiter = RateLimiter(self.redis_url)
            try:
                allowed = 0
                blocked = 0
                response_times = []
                
                for i in range(requests_per_client):
                    start = time.time()
                    result = rate_limiter.check_rate_limit(f"client_{client_id}", RateLimitTier.BASIC)
                    duration = time.time() - start
                    
                    response_times.append(duration * 1000)
                    
                    if result.allowed:
                        allowed += 1
                    else:
                        blocked += 1
                
                return {
                    "client_id": client_id,
                    "allowed": allowed,
                    "blocked": blocked,
                    "avg_response_time_ms": statistics.mean(response_times)
                }
            finally:
                rate_limiter.close()
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = [executor.submit(client_worker, i) for i in range(num_clients)]
            client_results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        total_allowed = sum(r["allowed"] for r in client_results)
        total_blocked = sum(r["blocked"] for r in client_results)
        total_requests = total_allowed + total_blocked
        avg_response_times = [r["avg_response_time_ms"] for r in client_results]
        
        return {
            "num_clients": num_clients,
            "requests_per_client": requests_per_client,
            "total_requests": total_requests,
            "total_allowed": total_allowed,
            "total_blocked": total_blocked,
            "total_duration_seconds": total_duration,
            "requests_per_second": total_requests / total_duration,
            "overall_avg_response_time_ms": statistics.mean(avg_response_times),
            "client_response_time_variance": statistics.variance(avg_response_times) if len(avg_response_times) > 1 else 0
        }
    
    def _benchmark_different_strategies(self) -> Dict[str, Any]:
        """Benchmark different rate limiting strategies."""
        strategies = [
            RateLimitStrategy.TOKEN_BUCKET,
            RateLimitStrategy.SLIDING_WINDOW,
            RateLimitStrategy.FIXED_WINDOW,
            RateLimitStrategy.ADAPTIVE
        ]
        
        results = {}
        num_requests = 200
        
        for strategy in strategies:
            rate_limiter = RateLimiter(self.redis_url)
            
            # Modify rule to use specific strategy
            from rate_limiter import RateLimitRule
            custom_rule = RateLimitRule(
                requests_per_minute=50,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_allowance=10,
                strategy=strategy,
                tier=RateLimitTier.BASIC
            )
            rate_limiter.rules[RateLimitTier.BASIC] = custom_rule
            
            try:
                start_time = time.time()
                allowed_count = 0
                response_times = []
                
                for i in range(num_requests):
                    request_start = time.time()
                    result = rate_limiter.check_rate_limit(f"strategy_test_{strategy.value}", RateLimitTier.BASIC)
                    response_time = time.time() - request_start
                    
                    response_times.append(response_time * 1000)
                    if result.allowed:
                        allowed_count += 1
                
                total_duration = time.time() - start_time
                
                results[strategy.value] = {
                    "total_requests": num_requests,
                    "allowed_requests": allowed_count,
                    "blocked_requests": num_requests - allowed_count,
                    "avg_response_time_ms": statistics.mean(response_times),
                    "total_duration_seconds": total_duration,
                    "requests_per_second": num_requests / total_duration
                }
            finally:
                rate_limiter.close()
        
        return results
    
    def _benchmark_fallback_mode(self) -> Dict[str, Any]:
        """Benchmark performance in fallback mode (without Redis)."""
        # Use non-existent Redis URL to force fallback mode
        rate_limiter = RateLimiter("redis://localhost:9999/0")
        
        try:
            num_requests = 500
            start_time = time.time()
            response_times = []
            allowed_count = 0
            
            for i in range(num_requests):
                request_start = time.time()
                result = rate_limiter.check_rate_limit("fallback_test", RateLimitTier.BASIC)
                response_time = time.time() - request_start
                
                response_times.append(response_time * 1000)
                if result.allowed:
                    allowed_count += 1
            
            total_duration = time.time() - start_time
            
            return {
                "total_requests": num_requests,
                "allowed_requests": allowed_count,
                "blocked_requests": num_requests - allowed_count,
                "total_duration_seconds": total_duration,
                "requests_per_second": num_requests / total_duration,
                "avg_response_time_ms": statistics.mean(response_times),
                "redis_available": rate_limiter.redis_available,
                "mode": "fallback"
            }
        finally:
            rate_limiter.close()
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        rate_limiter = RateLimiter(self.redis_url)
        
        try:
            # Create many clients and make requests
            num_clients = 100
            memory_samples = []
            
            for i in range(num_clients):
                # Make requests for each client
                for j in range(10):
                    rate_limiter.check_rate_limit(f"memory_test_client_{i}", RateLimitTier.BASIC)
                
                # Sample memory every 10 clients
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": final_memory - initial_memory,
                "peak_memory_mb": max(memory_samples) if memory_samples else final_memory,
                "num_clients_tested": num_clients,
                "memory_per_client_kb": ((final_memory - initial_memory) * 1024) / num_clients if num_clients > 0 else 0,
                "memory_samples": memory_samples
            }
        finally:
            rate_limiter.close()
    
    def _benchmark_latency_distribution(self) -> Dict[str, Any]:
        """Benchmark latency distribution under different loads."""
        rate_limiter = RateLimiter(self.redis_url)
        
        try:
            results = {}
            load_levels = [1, 5, 10, 20, 50]  # Concurrent clients
            
            for load in load_levels:
                response_times = []
                
                def worker():
                    for i in range(20):  # 20 requests per worker
                        start = time.time()
                        rate_limiter.check_rate_limit(f"latency_test_{threading.current_thread().ident}", RateLimitTier.BASIC)
                        duration = time.time() - start
                        response_times.append(duration * 1000)
                
                # Run concurrent workers
                threads = []
                start_time = time.time()
                
                for i in range(load):
                    thread = threading.Thread(target=worker)
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                total_duration = time.time() - start_time
                
                if response_times:
                    results[f"load_{load}_clients"] = {
                        "concurrent_clients": load,
                        "total_requests": len(response_times),
                        "total_duration_seconds": total_duration,
                        "avg_latency_ms": statistics.mean(response_times),
                        "median_latency_ms": statistics.median(response_times),
                        "p95_latency_ms": self._percentile(response_times, 95),
                        "p99_latency_ms": self._percentile(response_times, 99),
                        "min_latency_ms": min(response_times),
                        "max_latency_ms": max(response_times),
                        "latency_std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                    }
            
            return results
        finally:
            rate_limiter.close()
    
    def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability with increasing load."""
        results = {}
        client_counts = [10, 25, 50, 100, 200]
        
        for client_count in client_counts:
            rate_limiter = RateLimiter(self.redis_url)
            
            try:
                def client_work(client_id):
                    times = []
                    for i in range(5):  # 5 requests per client
                        start = time.time()
                        rate_limiter.check_rate_limit(f"scale_client_{client_id}", RateLimitTier.BASIC)
                        times.append(time.time() - start)
                    return times
                
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=client_count) as executor:
                    futures = [executor.submit(client_work, i) for i in range(client_count)]
                    all_times = []
                    for future in as_completed(futures):
                        all_times.extend(future.result())
                
                total_duration = time.time() - start_time
                
                results[f"clients_{client_count}"] = {
                    "client_count": client_count,
                    "total_requests": len(all_times),
                    "total_duration_seconds": total_duration,
                    "requests_per_second": len(all_times) / total_duration,
                    "avg_response_time_ms": statistics.mean(all_times) * 1000,
                    "p95_response_time_ms": self._percentile([t * 1000 for t in all_times], 95)
                }
            finally:
                rate_limiter.close()
        
        return results
    
    def _benchmark_redis_vs_fallback(self) -> Dict[str, Any]:
        """Compare Redis mode vs fallback mode performance."""
        num_requests = 300
        
        # Test fallback mode (non-existent Redis)
        fallback_limiter = RateLimiter("redis://localhost:9999/0")
        
        try:
            start_time = time.time()
            fallback_times = []
            
            for i in range(num_requests):
                request_start = time.time()
                fallback_limiter.check_rate_limit("fallback_vs_redis", RateLimitTier.BASIC)
                fallback_times.append(time.time() - request_start)
            
            fallback_duration = time.time() - start_time
        finally:
            fallback_limiter.close()
        
        # For comparison, also test with the test Redis URL
        redis_limiter = RateLimiter(self.redis_url)
        
        try:
            start_time = time.time()
            redis_times = []
            
            for i in range(num_requests):
                request_start = time.time()
                redis_limiter.check_rate_limit("fallback_vs_redis", RateLimitTier.BASIC)
                redis_times.append(time.time() - request_start)
            
            redis_duration = time.time() - start_time
        finally:
            redis_limiter.close()
        
        return {
            "fallback_mode": {
                "total_requests": num_requests,
                "total_duration_seconds": fallback_duration,
                "requests_per_second": num_requests / fallback_duration,
                "avg_response_time_ms": statistics.mean(fallback_times) * 1000,
                "redis_available": fallback_limiter.redis_available
            },
            "redis_mode": {
                "total_requests": num_requests,
                "total_duration_seconds": redis_duration,
                "requests_per_second": num_requests / redis_duration,
                "avg_response_time_ms": statistics.mean(redis_times) * 1000,
                "redis_available": redis_limiter.redis_available
            },
            "performance_ratio": {
                "fallback_vs_redis_speed": (num_requests / fallback_duration) / (num_requests / redis_duration) if redis_duration > 0 else 0,
                "fallback_vs_redis_latency": (statistics.mean(fallback_times) * 1000) / (statistics.mean(redis_times) * 1000) if redis_times else 0
            }
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _print_test_summary(self, test_name: str, result: Dict[str, Any]) -> None:
        """Print a summary of test results."""
        if "error" in result:
            return
        
        if test_name == "single_client_throughput":
            print(f"   📈 {result['requests_per_second']:.0f} req/s, {result['average_response_time_ms']:.2f}ms avg")
        elif test_name == "multi_client_concurrent":
            print(f"   👥 {result['num_clients']} clients, {result['requests_per_second']:.0f} req/s total")
        elif test_name == "memory_usage":
            print(f"   🧠 Memory increase: {result['memory_increase_mb']:.1f}MB for {result['num_clients_tested']} clients")
        elif test_name == "scalability_test":
            max_clients = max([int(k.split('_')[1]) for k in result.keys()])
            max_rps = result[f"clients_{max_clients}"]["requests_per_second"]
            print(f"   🚀 Scaled to {max_clients} clients at {max_rps:.0f} req/s")
    
    def _print_overall_summary(self) -> None:
        """Print overall benchmark summary."""
        print(f"\n📊 Overall Results:")
        print(f"   Total Duration: {self.results['total_duration']:.1f}s")
        print(f"   Tests Completed: {len([t for t in self.results['tests'].values() if 'error' not in t])}/{len(self.results['tests'])}")
        
        # Find best performance metrics
        throughput_test = self.results['tests'].get('single_client_throughput', {})
        if 'requests_per_second' in throughput_test:
            print(f"   Peak Single Client: {throughput_test['requests_per_second']:.0f} req/s")
        
        concurrent_test = self.results['tests'].get('multi_client_concurrent', {})
        if 'requests_per_second' in concurrent_test:
            print(f"   Peak Concurrent: {concurrent_test['requests_per_second']:.0f} req/s")
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Filename where results were saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rate_limit_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename


def main():
    """Main function for running benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rate Limiting Performance Benchmarks")
    parser.add_argument("--redis-url", help="Redis URL for testing")
    parser.add_argument("--output", help="Output filename for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run benchmarks
    benchmark = RateLimitBenchmark(args.redis_url)
    
    if args.quick:
        print("🏃 Running quick benchmarks...")
        # Run subset of tests for quick feedback
        benchmark.results = {
            "benchmark_start": datetime.now(timezone.utc).isoformat(),
            "environment": benchmark._get_environment_info(),
            "tests": {}
        }
        
        quick_tests = [
            ("single_client_throughput", benchmark._benchmark_single_client_throughput),
            ("fallback_mode_performance", benchmark._benchmark_fallback_mode)
        ]
        
        for test_name, test_method in quick_tests:
            print(f"\n📊 Running {test_name.replace('_', ' ').title()}...")
            try:
                result = test_method()
                benchmark.results["tests"][test_name] = result
                benchmark._print_test_summary(test_name, result)
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
    else:
        results = benchmark.run_all_benchmarks()
    
    # Save results
    filename = benchmark.save_results(args.output)
    
    print(f"\n🎯 Benchmark complete! Results in {filename}")


if __name__ == "__main__":
    main()