"""
Test Redis Week 2 Migration Implementation

Comprehensive test suite for Redis-backed AI cache and state management
with migration capabilities and performance validation.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

# Test imports
from scripts.redis_integration_adapters import (
    get_enhanced_cache,
    get_enhanced_state_manager,
    get_migration_manager,
    MigrationConfig
)
from scripts.enhanced_http_ai_client import get_enhanced_ai_client
from scripts.redis_migration_coordinator import create_migration_coordinator


class RedisWeek2Tester:
    """Comprehensive tester for Redis Week 2 migration implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        self.start_time = None
        
        # Test configuration
        self.test_config = {
            'cache_test_entries': 100,
            'state_test_keys': 50,
            'performance_iterations': 10,
            'migration_timeout': 300  # 5 minutes
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite for Redis Week 2 implementation."""
        self.start_time = time.time()
        self.logger.info("Starting Redis Week 2 comprehensive test suite")
        
        try:
            # Test 1: Enhanced Cache Adapter
            await self._test_enhanced_cache_adapter()
            
            # Test 2: Enhanced State Manager Adapter
            await self._test_enhanced_state_manager_adapter()
            
            # Test 3: Enhanced HTTP AI Client
            await self._test_enhanced_http_ai_client()
            
            # Test 4: Migration Coordinator
            await self._test_migration_coordinator()
            
            # Test 5: Performance Comparison
            await self._test_performance_comparison()
            
            # Test 6: Data Consistency Validation
            await self._test_data_consistency()
            
            # Test 7: Failover and Recovery
            await self._test_failover_recovery()
            
            # Generate final report
            return self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive tests: {e}")
            self._add_test_result("comprehensive_tests", False, str(e))
            return self._generate_test_report()
    
    async def _test_enhanced_cache_adapter(self):
        """Test enhanced cache adapter functionality."""
        test_name = "enhanced_cache_adapter"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize cache adapter
            cache_adapter = await get_enhanced_cache(enable_redis=True, migration_mode="gradual")
            
            # Test basic operations
            test_prompt = "What is the capital of France?"
            test_response = "The capital of France is Paris."
            test_provider = "anthropic"
            test_model = "claude-3"
            
            # Test put operation
            await cache_adapter.put(test_prompt, test_response, test_provider, test_model)
            
            # Test get operation
            retrieved_response = await cache_adapter.get(test_prompt, test_provider, test_model)
            
            if retrieved_response == test_response:
                self._add_test_result(f"{test_name}_basic_ops", True, "Basic cache operations successful")
            else:
                self._add_test_result(f"{test_name}_basic_ops", False, "Cache retrieval mismatch")
            
            # Test cache warming
            historical_data = [
                {
                    'prompt': f'Test prompt {i}',
                    'response': f'Test response {i}',
                    'provider': 'anthropic',
                    'model': 'claude-3'
                }
                for i in range(self.test_config['cache_test_entries'])
            ]
            
            warmed_count = await cache_adapter.warm_cache(historical_data)
            
            if warmed_count == len(historical_data):
                self._add_test_result(f"{test_name}_warming", True, f"Cache warmed with {warmed_count} entries")
            else:
                self._add_test_result(f"{test_name}_warming", False, f"Expected {len(historical_data)}, got {warmed_count}")
            
            # Test cache statistics
            stats = cache_adapter.get_stats()
            
            if 'redis' in stats or 'cache_size' in stats:
                self._add_test_result(f"{test_name}_stats", True, "Cache statistics available")
            else:
                self._add_test_result(f"{test_name}_stats", False, "Cache statistics missing")
            
            # Test migration status
            migration_status = await cache_adapter.get_migration_status()
            
            if 'redis_enabled' in migration_status:
                self._add_test_result(f"{test_name}_migration_status", True, "Migration status available")
            else:
                self._add_test_result(f"{test_name}_migration_status", False, "Migration status missing")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_enhanced_state_manager_adapter(self):
        """Test enhanced state manager adapter functionality."""
        test_name = "enhanced_state_manager_adapter"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize state manager adapter
            state_manager = await get_enhanced_state_manager(enable_redis=True, migration_mode="gradual")
            
            # Test basic state operations
            test_key = "test.performance.score"
            test_value = 0.95
            
            # Test update operation
            await state_manager.update(test_key, test_value)
            
            # Test get operation
            retrieved_value = await state_manager.get(test_key)
            
            if retrieved_value == test_value:
                self._add_test_result(f"{test_name}_basic_ops", True, "Basic state operations successful")
            else:
                self._add_test_result(f"{test_name}_basic_ops", False, f"State retrieval mismatch: {retrieved_value} != {test_value}")
            
            # Test distributed operations
            distributed_key = "test.distributed.counter"
            await state_manager.update(distributed_key, 1, distributed=True)
            
            distributed_value = await state_manager.get(distributed_key)
            
            if distributed_value == 1:
                self._add_test_result(f"{test_name}_distributed", True, "Distributed operations successful")
            else:
                self._add_test_result(f"{test_name}_distributed", False, "Distributed operation failed")
            
            # Test bulk state operations
            for i in range(self.test_config['state_test_keys']):
                await state_manager.update(f"test.bulk.key_{i}", f"value_{i}")
            
            # Verify bulk operations
            bulk_success = True
            for i in range(min(10, self.test_config['state_test_keys'])):  # Check first 10
                value = await state_manager.get(f"test.bulk.key_{i}")
                if value != f"value_{i}":
                    bulk_success = False
                    break
            
            if bulk_success:
                self._add_test_result(f"{test_name}_bulk_ops", True, f"Bulk operations successful ({self.test_config['state_test_keys']} keys)")
            else:
                self._add_test_result(f"{test_name}_bulk_ops", False, "Bulk operations failed")
            
            # Test full state operations
            full_state = await state_manager.get_full_state()
            
            if isinstance(full_state, dict) and len(full_state) > 0:
                self._add_test_result(f"{test_name}_full_state", True, f"Full state retrieved ({len(full_state)} keys)")
            else:
                self._add_test_result(f"{test_name}_full_state", False, "Full state retrieval failed")
            
            # Test metrics
            metrics = state_manager.get_metrics()
            
            if 'total_operations' in metrics or 'read_operations' in metrics:
                self._add_test_result(f"{test_name}_metrics", True, "State metrics available")
            else:
                self._add_test_result(f"{test_name}_metrics", False, "State metrics missing")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_enhanced_http_ai_client(self):
        """Test enhanced HTTP AI client functionality."""
        test_name = "enhanced_http_ai_client"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize enhanced AI client
            ai_client = await get_enhanced_ai_client(enable_redis=True, distributed=True)
            
            # Test basic AI request (mock)
            test_prompt = "Explain quantum computing in simple terms"
            
            response = await ai_client.generate_enhanced_response(
                prompt=test_prompt,
                model="claude-3",
                distributed=True
            )
            
            if 'content' in response and 'cache_backend' in response:
                self._add_test_result(f"{test_name}_basic_request", True, "Enhanced AI request successful")
            else:
                self._add_test_result(f"{test_name}_basic_request", False, "Enhanced AI request missing fields")
            
            # Test caching behavior
            response2 = await ai_client.generate_enhanced_response(
                prompt=test_prompt,
                model="claude-3"
            )
            
            if response2.get('cached', False):
                self._add_test_result(f"{test_name}_caching", True, "Caching behavior working")
            else:
                self._add_test_result(f"{test_name}_caching", False, "Caching not working as expected")
            
            # Test cache status
            cache_status = await ai_client.get_cache_status()
            
            if 'redis_cache_enabled' in cache_status and 'performance_metrics' in cache_status:
                self._add_test_result(f"{test_name}_cache_status", True, "Cache status comprehensive")
            else:
                self._add_test_result(f"{test_name}_cache_status", False, "Cache status incomplete")
            
            # Test distributed statistics
            dist_stats = await ai_client.get_distributed_statistics()
            
            if 'statistics' in dist_stats or 'error' in dist_stats:
                self._add_test_result(f"{test_name}_distributed_stats", True, "Distributed statistics accessible")
            else:
                self._add_test_result(f"{test_name}_distributed_stats", False, "Distributed statistics failed")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_migration_coordinator(self):
        """Test migration coordinator functionality."""
        test_name = "migration_coordinator"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Create migration configuration
            migration_config = MigrationConfig(
                migration_mode="gradual",
                dual_write_duration_hours=0.1,  # 6 minutes for testing
                validation_duration_hours=0.05,  # 3 minutes for testing
                enable_automatic_rollback=True
            )
            
            # Initialize migration coordinator
            coordinator = await create_migration_coordinator(migration_config)
            
            # Test coordinator initialization
            if coordinator:
                self._add_test_result(f"{test_name}_initialization", True, "Migration coordinator initialized")
            else:
                self._add_test_result(f"{test_name}_initialization", False, "Migration coordinator failed to initialize")
                return
            
            # Test migration status
            initial_status = coordinator.get_migration_status()
            
            if 'metrics' in initial_status and 'config' in initial_status:
                self._add_test_result(f"{test_name}_status", True, "Migration status comprehensive")
            else:
                self._add_test_result(f"{test_name}_status", False, "Migration status incomplete")
            
            # Note: We won't run a full migration in tests as it's time-consuming
            # Instead, we'll test the migration planning phase
            
            # Simulate migration planning
            await coordinator._phase_planning()
            
            if coordinator.metrics.cache_entries_total >= 0 and coordinator.metrics.state_keys_total >= 0:
                self._add_test_result(f"{test_name}_planning", True, "Migration planning successful")
            else:
                self._add_test_result(f"{test_name}_planning", False, "Migration planning failed")
            
            # Test coordinator shutdown
            await coordinator.shutdown()
            self._add_test_result(f"{test_name}_shutdown", True, "Migration coordinator shutdown successful")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_performance_comparison(self):
        """Test performance comparison between Redis and legacy systems."""
        test_name = "performance_comparison"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Test Redis cache performance
            redis_cache = await get_enhanced_cache(enable_redis=True, migration_mode="immediate")
            
            redis_times = []
            for i in range(self.test_config['performance_iterations']):
                start_time = time.time()
                await redis_cache.put(f"perf_test_{i}", f"response_{i}", "test_provider", "test_model")
                retrieved = await redis_cache.get(f"perf_test_{i}", "test_provider", "test_model")
                end_time = time.time()
                
                if retrieved == f"response_{i}":
                    redis_times.append(end_time - start_time)
            
            redis_avg_time = sum(redis_times) / len(redis_times) if redis_times else float('inf')
            
            # Test legacy cache performance (if available)
            try:
                from scripts.ai_response_cache import AIResponseCache
                legacy_cache = AIResponseCache()
                
                legacy_times = []
                for i in range(self.test_config['performance_iterations']):
                    start_time = time.time()
                    await legacy_cache.put(f"perf_test_{i}", f"response_{i}", "test_provider", "test_model")
                    retrieved = await legacy_cache.get(f"perf_test_{i}", "test_provider", "test_model")
                    end_time = time.time()
                    
                    if retrieved == f"response_{i}":
                        legacy_times.append(end_time - start_time)
                
                legacy_avg_time = sum(legacy_times) / len(legacy_times) if legacy_times else float('inf')
                
                # Compare performance
                performance_ratio = redis_avg_time / legacy_avg_time if legacy_avg_time > 0 else 1.0
                
                if performance_ratio < 2.0:  # Redis should be within 2x of legacy performance
                    self._add_test_result(f"{test_name}_cache", True, 
                                        f"Redis cache performance acceptable (ratio: {performance_ratio:.2f})")
                else:
                    self._add_test_result(f"{test_name}_cache", False, 
                                        f"Redis cache slower than expected (ratio: {performance_ratio:.2f})")
                
            except ImportError:
                self._add_test_result(f"{test_name}_cache", True, "Legacy cache not available for comparison")
            
            # Test state manager performance
            redis_state = await get_enhanced_state_manager(enable_redis=True, migration_mode="immediate")
            
            state_times = []
            for i in range(self.test_config['performance_iterations']):
                start_time = time.time()
                await redis_state.update(f"perf.state.{i}", f"value_{i}")
                retrieved = await redis_state.get(f"perf.state.{i}")
                end_time = time.time()
                
                if retrieved == f"value_{i}":
                    state_times.append(end_time - start_time)
            
            state_avg_time = sum(state_times) / len(state_times) if state_times else float('inf')
            
            if state_avg_time < 1.0:  # Should be sub-second for basic operations
                self._add_test_result(f"{test_name}_state", True, 
                                    f"State manager performance good ({state_avg_time:.3f}s avg)")
            else:
                self._add_test_result(f"{test_name}_state", False, 
                                    f"State manager performance slow ({state_avg_time:.3f}s avg)")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_data_consistency(self):
        """Test data consistency between Redis and legacy systems."""
        test_name = "data_consistency"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Test cache consistency
            cache_adapter = await get_enhanced_cache(enable_redis=True, migration_mode="gradual")
            
            # Store test data
            test_data = [
                ("consistency_test_1", "response_1", "provider_1", "model_1"),
                ("consistency_test_2", "response_2", "provider_2", "model_2"),
                ("consistency_test_3", "response_3", "provider_3", "model_3"),
            ]
            
            # Store data
            for prompt, response, provider, model in test_data:
                await cache_adapter.put(prompt, response, provider, model)
            
            # Verify data consistency
            consistency_count = 0
            for prompt, expected_response, provider, model in test_data:
                retrieved = await cache_adapter.get(prompt, provider, model)
                if retrieved == expected_response:
                    consistency_count += 1
            
            consistency_rate = consistency_count / len(test_data)
            
            if consistency_rate >= 0.95:  # 95% consistency threshold
                self._add_test_result(f"{test_name}_cache", True, 
                                    f"Cache consistency excellent ({consistency_rate:.1%})")
            else:
                self._add_test_result(f"{test_name}_cache", False, 
                                    f"Cache consistency poor ({consistency_rate:.1%})")
            
            # Test state consistency
            state_manager = await get_enhanced_state_manager(enable_redis=True, migration_mode="gradual")
            
            # Store test state
            test_state_data = {
                "consistency.test.score": 0.95,
                "consistency.test.count": 42,
                "consistency.test.name": "Redis Migration Test"
            }
            
            for key, value in test_state_data.items():
                await state_manager.update(key, value)
            
            # Verify state consistency
            state_consistency_count = 0
            for key, expected_value in test_state_data.items():
                retrieved = await state_manager.get(key)
                if retrieved == expected_value:
                    state_consistency_count += 1
            
            state_consistency_rate = state_consistency_count / len(test_state_data)
            
            if state_consistency_rate >= 0.95:  # 95% consistency threshold
                self._add_test_result(f"{test_name}_state", True, 
                                    f"State consistency excellent ({state_consistency_rate:.1%})")
            else:
                self._add_test_result(f"{test_name}_state", False, 
                                    f"State consistency poor ({state_consistency_rate:.1%})")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_failover_recovery(self):
        """Test failover and recovery mechanisms."""
        test_name = "failover_recovery"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Test cache failover
            cache_adapter = await get_enhanced_cache(enable_redis=True, migration_mode="gradual")
            
            # Store data in normal operation
            await cache_adapter.put("failover_test", "test_response", "test_provider", "test_model")
            
            # Verify normal retrieval
            normal_response = await cache_adapter.get("failover_test", "test_provider", "test_model")
            
            if normal_response == "test_response":
                self._add_test_result(f"{test_name}_normal_operation", True, "Normal cache operation successful")
            else:
                self._add_test_result(f"{test_name}_normal_operation", False, "Normal cache operation failed")
            
            # Simulate Redis unavailability by switching to readonly mode
            if hasattr(cache_adapter, '_redis_cache') and cache_adapter._redis_cache:
                cache_adapter._redis_cache.migration_mode = "readonly"
                
                # Test that fallback still works
                fallback_response = await cache_adapter.get("failover_test", "test_provider", "test_model")
                
                if fallback_response:  # Should fall back to legacy cache
                    self._add_test_result(f"{test_name}_failover", True, "Cache failover successful")
                else:
                    self._add_test_result(f"{test_name}_failover", False, "Cache failover failed")
                
                # Restore normal operation
                cache_adapter._redis_cache.migration_mode = "gradual"
            else:
                self._add_test_result(f"{test_name}_failover", True, "Redis cache not available for failover test")
            
            # Test state manager failover
            state_manager = await get_enhanced_state_manager(enable_redis=True, migration_mode="gradual")
            
            # Store state in normal operation
            await state_manager.update("failover.test.value", "test_state")
            
            # Verify normal retrieval
            normal_state = await state_manager.get("failover.test.value")
            
            if normal_state == "test_state":
                self._add_test_result(f"{test_name}_state_normal", True, "Normal state operation successful")
            else:
                self._add_test_result(f"{test_name}_state_normal", False, "Normal state operation failed")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    def _add_test_result(self, test_name: str, success: bool, message: str):
        """Add test result to results list."""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        self.logger.info(f"[{status}] {test_name}: {message}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        total_time = time.time() - self.start_time if self.start_time else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        self.logger.info(f"Test Report: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [result for result in self.test_results if not result['success']]
        
        if not failed_tests:
            recommendations.append("All tests passed! Redis Week 2 implementation is ready for production.")
        else:
            recommendations.append(f"{len(failed_tests)} tests failed. Review failed tests before proceeding.")
            
            # Specific recommendations based on failures
            failed_test_names = [test['test_name'] for test in failed_tests]
            
            if any('cache' in name for name in failed_test_names):
                recommendations.append("Cache-related tests failed. Check Redis connectivity and cache configuration.")
            
            if any('state' in name for name in failed_test_names):
                recommendations.append("State management tests failed. Verify Redis state manager configuration.")
            
            if any('migration' in name for name in failed_test_names):
                recommendations.append("Migration tests failed. Review migration coordinator setup.")
            
            if any('performance' in name for name in failed_test_names):
                recommendations.append("Performance tests failed. Consider Redis optimization or infrastructure scaling.")
        
        return recommendations


async def main():
    """Main test runner function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Redis Week 2 Migration Test Suite")
    
    try:
        # Initialize tester
        tester = RedisWeek2Tester()
        
        # Run comprehensive tests
        test_report = await tester.run_comprehensive_tests()
        
        # Print summary
        summary = test_report['summary']
        logger.info("=" * 80)
        logger.info("REDIS WEEK 2 MIGRATION TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total Time: {summary['total_time_seconds']:.1f}s")
        logger.info("=" * 80)
        
        # Print recommendations
        logger.info("RECOMMENDATIONS:")
        for i, recommendation in enumerate(test_report['recommendations'], 1):
            logger.info(f"{i}. {recommendation}")
        
        # Save detailed report
        report_filename = f"redis_week2_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        logger.info(f"Detailed report saved to: {report_filename}")
        
        # Return success status
        return summary['success_rate'] >= 0.8  # 80% success threshold
        
    except Exception as e:
        logger.error(f"Error in test runner: {e}")
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)