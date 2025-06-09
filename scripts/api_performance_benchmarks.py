"""
API Performance Benchmarks

Comprehensive benchmarking suite for the CWMAI API rate limiting and caching system.
Tests performance under various load conditions.
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
import redis
import aiohttp
import psutil


class BenchmarkResult:
    """Result of a benchmark test."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.requests_sent = 0
        self.requests_successful = 0
        self.requests_rate_limited = 0
        self.requests_failed = 0
        self.response_times: List[float] = []
        self.errors: List[str] = []
        self.memory_usage_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_usage_end = None
        
    def record_request(self, success: bool, response_time: float, rate_limited: bool = False, error: str = None):
        """Record a request result."""
        self.requests_sent += 1
        self.response_times.append(response_time)
        
        if rate_limited:
            self.requests_rate_limited += 1
        elif success:
            self.requests_successful += 1
        else:
            self.requests_failed += 1
            if error:
                self.errors.append(error)
    
    def finish(self):
        """Mark benchmark as complete."""
        self.end_time = time.time()
        self.memory_usage_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        duration = self.end_time - self.start_time if self.end_time else 0
        
        summary = {
            "name": self.name,
            "duration_seconds": duration,
            "requests": {
                "total": self.requests_sent,
                "successful": self.requests_successful,
                "rate_limited": self.requests_rate_limited,
                "failed": self.requests_failed,
                "success_rate": self.requests_successful / max(1, self.requests_sent),
                "requests_per_second": self.requests_sent / max(0.001, duration)
            },
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "mean": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else 0,
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else 0
            },
            "memory": {
                "start_mb": self.memory_usage_start,
                "end_mb": self.memory_usage_end or self.memory_usage_start,
                "peak_increase_mb": (self.memory_usage_end or self.memory_usage_start) - self.memory_usage_start
            },
            "errors": {
                "count": len(self.errors),
                "unique_errors": len(set(self.errors)),
                "sample_errors": self.errors[:5]  # First 5 errors as sample
            }
        }
        
        return summary


class APIPerformanceBenchmarks:
    """Comprehensive API performance benchmarking suite."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize benchmark suite."""
        self.api_base_url = api_base_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.logger = logging.getLogger(f"{__name__}.APIPerformanceBenchmarks")
        
        # Test Redis connection
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
        except Exception as e:
            self.logger.warning(f"Redis not available for benchmarks: {e}")
            self.redis_client = None
            self.redis_available = False
    
    async def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmark tests."""
        print("🚀 Starting CWMAI API Performance Benchmarks")
        print(f"API URL: {self.api_base_url}")
        print(f"Redis: {'Available' if self.redis_available else 'Not Available'}")
        print("=" * 60)
        
        results = {}
        
        # Basic API benchmarks
        results["basic_endpoints"] = await self.benchmark_basic_endpoints()
        results["rate_limiting"] = await self.benchmark_rate_limiting()
        results["concurrent_requests"] = await self.benchmark_concurrent_requests()
        
        # Cache benchmarks
        if self.redis_available:
            results["cache_performance"] = await self.benchmark_cache_performance()
            results["cache_memory"] = await self.benchmark_cache_memory_usage()
        
        # AI endpoint benchmarks
        results["ai_endpoints"] = await self.benchmark_ai_endpoints()
        
        # WebSocket benchmarks
        results["websocket"] = await self.benchmark_websocket_connections()
        
        # Stress tests
        results["stress_test"] = await self.benchmark_stress_test()
        
        return results
    
    async def benchmark_basic_endpoints(self) -> BenchmarkResult:
        """Benchmark basic API endpoints."""
        result = BenchmarkResult("basic_endpoints")
        
        endpoints = [
            "/",
            "/health",
            "/status",
            "/metrics"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                for _ in range(10):  # 10 requests per endpoint
                    start_time = time.time()
                    try:
                        async with session.get(f"{self.api_base_url}{endpoint}") as response:
                            response_time = time.time() - start_time
                            success = response.status == 200
                            rate_limited = response.status == 429
                            
                            result.record_request(success, response_time, rate_limited)
                            
                    except Exception as e:
                        response_time = time.time() - start_time
                        result.record_request(False, response_time, error=str(e))
        
        result.finish()
        return result
    
    async def benchmark_rate_limiting(self) -> BenchmarkResult:
        """Benchmark rate limiting functionality."""
        result = BenchmarkResult("rate_limiting")
        
        async with aiohttp.ClientSession() as session:
            # Send requests rapidly to trigger rate limiting
            for i in range(100):  # Send 100 requests quickly
                start_time = time.time()
                try:
                    async with session.get(f"{self.api_base_url}/status") as response:
                        response_time = time.time() - start_time
                        success = response.status == 200
                        rate_limited = response.status == 429
                        
                        result.record_request(success, response_time, rate_limited)
                        
                        # Small delay to avoid overwhelming
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
        
        result.finish()
        return result
    
    async def benchmark_concurrent_requests(self) -> BenchmarkResult:
        """Benchmark concurrent request handling."""
        result = BenchmarkResult("concurrent_requests")
        
        async def make_request(session, semaphore):
            async with semaphore:
                start_time = time.time()
                try:
                    async with session.get(f"{self.api_base_url}/health") as response:
                        response_time = time.time() - start_time
                        success = response.status == 200
                        rate_limited = response.status == 429
                        
                        result.record_request(success, response_time, rate_limited)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
        
        # Limit to 20 concurrent connections
        semaphore = asyncio.Semaphore(20)
        
        async with aiohttp.ClientSession() as session:
            # Create 100 concurrent requests
            tasks = [make_request(session, semaphore) for _ in range(100)]
            await asyncio.gather(*tasks)
        
        result.finish()
        return result
    
    async def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache performance."""
        result = BenchmarkResult("cache_performance")
        
        if not self.redis_available:
            result.finish()
            return result
        
        # Test cache operations directly
        from scripts.redis_cache_manager import RedisCacheManager
        
        cache_manager = RedisCacheManager(self.redis_client)
        
        # Test cache set operations
        for i in range(1000):
            start_time = time.time()
            try:
                await cache_manager.set(f"test_key_{i}", f"test_value_{i}", ttl=300)
                response_time = time.time() - start_time
                result.record_request(True, response_time)
            except Exception as e:
                response_time = time.time() - start_time
                result.record_request(False, response_time, error=str(e))
        
        # Test cache get operations
        for i in range(1000):
            start_time = time.time()
            try:
                value = await cache_manager.get(f"test_key_{i}")
                response_time = time.time() - start_time
                success = value is not None
                result.record_request(success, response_time)
            except Exception as e:
                response_time = time.time() - start_time
                result.record_request(False, response_time, error=str(e))
        
        # Cleanup
        await cache_manager.clear_all()
        
        result.finish()
        return result
    
    async def benchmark_cache_memory_usage(self) -> BenchmarkResult:
        """Benchmark cache memory usage."""
        result = BenchmarkResult("cache_memory_usage")
        
        if not self.redis_available:
            result.finish()
            return result
        
        from scripts.redis_cache_manager import RedisCacheManager
        
        cache_manager = RedisCacheManager(self.redis_client, max_memory_items=5000)
        
        # Fill cache with various data sizes
        data_sizes = [100, 1000, 10000]  # bytes
        
        for size in data_sizes:
            test_data = "x" * size
            
            for i in range(100):
                start_time = time.time()
                try:
                    await cache_manager.set(f"memory_test_{size}_{i}", test_data, ttl=300)
                    response_time = time.time() - start_time
                    result.record_request(True, response_time)
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
        
        # Test memory cache fallback
        cache_manager.redis_client = None  # Disable Redis temporarily
        
        for i in range(100):
            start_time = time.time()
            try:
                await cache_manager.set(f"memory_fallback_{i}", "test_data", ttl=300)
                response_time = time.time() - start_time
                result.record_request(True, response_time)
            except Exception as e:
                response_time = time.time() - start_time
                result.record_request(False, response_time, error=str(e))
        
        # Cleanup
        await cache_manager.clear_all()
        
        result.finish()
        return result
    
    async def benchmark_ai_endpoints(self) -> BenchmarkResult:
        """Benchmark AI endpoint performance."""
        result = BenchmarkResult("ai_endpoints")
        
        ai_request_data = {
            "prompt": "Hello, this is a test prompt for benchmarking.",
            "model": "claude",
            "max_tokens": 100
        }
        
        async with aiohttp.ClientSession() as session:
            # Test AI endpoints (fewer requests due to rate limiting)
            for i in range(5):
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.api_base_url}/ai/generate",
                        json=ai_request_data
                    ) as response:
                        response_time = time.time() - start_time
                        success = response.status == 200
                        rate_limited = response.status == 429
                        
                        result.record_request(success, response_time, rate_limited)
                        
                        # Wait between AI requests
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
        
        result.finish()
        return result
    
    async def benchmark_websocket_connections(self) -> BenchmarkResult:
        """Benchmark WebSocket connections."""
        result = BenchmarkResult("websocket_connections")
        
        try:
            import websockets
            
            async def test_websocket():
                start_time = time.time()
                try:
                    uri = f"ws://localhost:8000/ws"
                    async with websockets.connect(uri) as websocket:
                        # Send ping
                        await websocket.send(json.dumps({"type": "ping"}))
                        
                        # Wait for pong
                        response = await websocket.recv()
                        response_data = json.loads(response)
                        
                        response_time = time.time() - start_time
                        success = response_data.get("type") == "pong"
                        
                        result.record_request(success, response_time)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
            
            # Test multiple WebSocket connections
            tasks = [test_websocket() for _ in range(10)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except ImportError:
            self.logger.warning("websockets library not available for WebSocket benchmarks")
        
        result.finish()
        return result
    
    async def benchmark_stress_test(self) -> BenchmarkResult:
        """Stress test the API."""
        result = BenchmarkResult("stress_test")
        
        async def stress_worker(session, worker_id):
            """Individual stress test worker."""
            endpoints = ["/", "/health", "/status", "/metrics"]
            
            for i in range(50):  # 50 requests per worker
                endpoint = endpoints[i % len(endpoints)]
                start_time = time.time()
                
                try:
                    async with session.get(f"{self.api_base_url}{endpoint}") as response:
                        response_time = time.time() - start_time
                        success = response.status == 200
                        rate_limited = response.status == 429
                        
                        result.record_request(success, response_time, rate_limited)
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    result.record_request(False, response_time, error=str(e))
                
                # Small random delay
                await asyncio.sleep(0.05 + (i % 3) * 0.01)
        
        # Run 10 workers concurrently
        async with aiohttp.ClientSession() as session:
            workers = [stress_worker(session, i) for i in range(10)]
            await asyncio.gather(*workers)
        
        result.finish()
        return result
    
    def print_benchmark_report(self, results: Dict[str, BenchmarkResult]):
        """Print comprehensive benchmark report."""
        print("\n" + "=" * 80)
        print("CWMAI API PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        for name, result in results.items():
            summary = result.get_summary()
            
            print(f"\n📊 {summary['name'].upper()}")
            print("-" * 50)
            
            # Duration and throughput
            print(f"Duration: {summary['duration_seconds']:.2f}s")
            print(f"Requests/sec: {summary['requests']['requests_per_second']:.2f}")
            
            # Request statistics
            req = summary['requests']
            print(f"Requests: {req['total']} total, {req['successful']} success, {req['rate_limited']} rate limited, {req['failed']} failed")
            print(f"Success rate: {req['success_rate']:.1%}")
            
            # Response times
            rt = summary['response_times']
            print(f"Response times: min={rt['min']:.3f}s, avg={rt['mean']:.3f}s, max={rt['max']:.3f}s")
            print(f"Percentiles: p95={rt['p95']:.3f}s, p99={rt['p99']:.3f}s")
            
            # Memory usage
            mem = summary['memory']
            print(f"Memory: {mem['start_mb']:.1f}MB → {mem['end_mb']:.1f}MB (Δ{mem['peak_increase_mb']:+.1f}MB)")
            
            # Errors
            errors = summary['errors']
            if errors['count'] > 0:
                print(f"⚠️  Errors: {errors['count']} total, {errors['unique_errors']} unique")
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully! 🎉")


async def main():
    """Run benchmarks."""
    logging.basicConfig(level=logging.INFO)
    
    benchmarks = APIPerformanceBenchmarks()
    results = await benchmarks.run_all_benchmarks()
    benchmarks.print_benchmark_report(results)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            name: result.get_summary()
            for name, result in results.items()
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())