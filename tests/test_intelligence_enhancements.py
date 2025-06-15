#!/usr/bin/env python3
"""
Test Intelligence Enhancements

Validates the newly implemented AI response cache, async state manager, 
intelligence hub, and enhanced swarm processing.
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Test imports
test_results = {
    'imports': {},
    'functionality': {},
    'performance': {},
    'integration': {}
}

def log_test(category, test_name, result, details=None):
    """Log test result."""
    test_results[category][test_name] = {
        'result': result,
        'details': details,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} {category}.{test_name}: {details or ''}")

async def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    # Test AI Response Cache
    try:
        from ai_response_cache import AIResponseCache, get_global_cache
        log_test('imports', 'ai_response_cache', True, "Successfully imported")
    except Exception as e:
        log_test('imports', 'ai_response_cache', False, f"Import failed: {e}")
    
    # Test Async State Manager
    try:
        from async_state_manager import AsyncStateManager, get_async_state_manager
        log_test('imports', 'async_state_manager', True, "Successfully imported")
    except Exception as e:
        log_test('imports', 'async_state_manager', False, f"Import failed: {e}")
    
    # Test Intelligence Hub
    try:
        from intelligence_hub import IntelligenceHub, get_intelligence_hub, EventType
        log_test('imports', 'intelligence_hub', True, "Successfully imported")
    except Exception as e:
        log_test('imports', 'intelligence_hub', False, f"Import failed: {e}")
    
    # Test Enhanced Swarm Intelligence
    try:
        from swarm_intelligence import RealSwarmIntelligence
        log_test('imports', 'enhanced_swarm', True, "Successfully imported")
    except Exception as e:
        log_test('imports', 'enhanced_swarm', False, f"Import failed: {e}")
    
    # Test Intelligence Integration
    try:
        from intelligence_integration import IntelligenceIntegrator, get_intelligence_integrator
        log_test('imports', 'intelligence_integration', True, "Successfully imported")
    except Exception as e:
        log_test('imports', 'intelligence_integration', False, f"Import failed: {e}")

async def test_ai_response_cache():
    """Test AI response cache functionality."""
    print("\nTesting AI Response Cache...")
    
    try:
        from ai_response_cache import AIResponseCache
        
        # Create cache instance
        cache = AIResponseCache(max_size=10, default_ttl=60)
        
        # Test basic put/get
        await cache.put("test prompt", "test response", "anthropic", "claude", cost_estimate=0.01)
        result = await cache.get("test prompt", "anthropic", "claude")
        
        if result == "test response":
            log_test('functionality', 'cache_basic_operations', True, "Put/get works correctly")
        else:
            log_test('functionality', 'cache_basic_operations', False, f"Expected 'test response', got '{result}'")
        
        # Test cache miss
        miss_result = await cache.get("nonexistent prompt", "anthropic", "claude")
        if miss_result is None:
            log_test('functionality', 'cache_miss_handling', True, "Cache miss returns None")
        else:
            log_test('functionality', 'cache_miss_handling', False, f"Expected None, got '{miss_result}'")
        
        # Test cache stats
        stats = cache.get_stats()
        if 'cache_size' in stats and 'stats' in stats:
            log_test('functionality', 'cache_stats', True, f"Stats: {stats['cache_size']} entries")
        else:
            log_test('functionality', 'cache_stats', False, "Stats format incorrect")
        
        # Test cleanup
        await cache.shutdown()
        log_test('functionality', 'cache_shutdown', True, "Shutdown completed")
        
    except Exception as e:
        log_test('functionality', 'cache_operations', False, f"Cache test failed: {e}")

async def test_async_state_manager():
    """Test async state manager functionality."""
    print("\nTesting Async State Manager...")
    
    try:
        from async_state_manager import AsyncStateManager
        
        # Create state manager
        state_manager = AsyncStateManager(state_file="test_state.json")
        await state_manager.initialize()
        
        # Test basic operations
        await state_manager.update("test.key", "test_value")
        result = await state_manager.get("test.key")
        
        if result == "test_value":
            log_test('functionality', 'state_basic_operations', True, "Update/get works correctly")
        else:
            log_test('functionality', 'state_basic_operations', False, f"Expected 'test_value', got '{result}'")
        
        # Test nested keys
        await state_manager.update("nested.deep.key", {"complex": "value"})
        nested_result = await state_manager.get("nested.deep.key")
        
        if nested_result == {"complex": "value"}:
            log_test('functionality', 'state_nested_operations', True, "Nested keys work correctly")
        else:
            log_test('functionality', 'state_nested_operations', False, f"Nested operation failed")
        
        # Test transaction
        async with state_manager.transaction():
            await state_manager.update("transaction.test", "transactional_value")
        
        tx_result = await state_manager.get("transaction.test")
        if tx_result == "transactional_value":
            log_test('functionality', 'state_transactions', True, "Transactions work correctly")
        else:
            log_test('functionality', 'state_transactions', False, "Transaction failed")
        
        # Test metrics
        metrics = state_manager.get_metrics()
        if 'total_operations' in metrics:
            log_test('functionality', 'state_metrics', True, f"Metrics: {metrics['total_operations']} ops")
        else:
            log_test('functionality', 'state_metrics', False, "Metrics format incorrect")
        
        # Cleanup
        await state_manager.shutdown()
        
        # Remove test file
        try:
            os.remove("test_state.json")
        except:
            pass
        
        log_test('functionality', 'state_shutdown', True, "Shutdown completed")
        
    except Exception as e:
        log_test('functionality', 'state_operations', False, f"State manager test failed: {e}")

async def test_intelligence_hub():
    """Test intelligence hub functionality."""
    print("\nTesting Intelligence Hub...")
    
    try:
        from intelligence_hub import IntelligenceHub, EventType
        
        # Create hub
        hub = IntelligenceHub(max_events=100)
        await hub.start()
        
        # Test event emission
        event_id = await hub.emit_event(
            event_type=EventType.DECISION_MADE,
            source_component="test_component",
            data={"decision": "test_decision", "priority": 8}
        )
        
        if event_id:
            log_test('functionality', 'hub_event_emission', True, f"Event emitted: {event_id}")
        else:
            log_test('functionality', 'hub_event_emission', False, "Event emission failed")
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        # Test event retrieval
        recent_events = hub.get_recent_events(hours=1)
        if len(recent_events) > 0:
            log_test('functionality', 'hub_event_retrieval', True, f"Found {len(recent_events)} events")
        else:
            log_test('functionality', 'hub_event_retrieval', False, "No events found")
        
        # Test metrics
        metrics = hub.get_metrics()
        if 'total_events' in metrics and metrics['total_events'] > 0:
            log_test('functionality', 'hub_metrics', True, f"Metrics: {metrics['total_events']} events")
        else:
            log_test('functionality', 'hub_metrics', False, "Metrics incorrect")
        
        # Test subscription
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
        
        hub.subscribe_to_events(EventType.TASK_COMPLETED, event_handler)
        
        # Emit test event
        await hub.emit_event(
            event_type=EventType.TASK_COMPLETED,
            source_component="test_component",
            data={"task": "test_task"}
        )
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        if len(received_events) > 0:
            log_test('functionality', 'hub_subscriptions', True, f"Received {len(received_events)} events")
        else:
            log_test('functionality', 'hub_subscriptions', False, "No events received")
        
        # Shutdown
        await hub.stop()
        log_test('functionality', 'hub_shutdown', True, "Shutdown completed")
        
    except Exception as e:
        log_test('functionality', 'hub_operations', False, f"Hub test failed: {e}")

async def test_enhanced_swarm():
    """Test enhanced swarm intelligence functionality."""
    print("\nTesting Enhanced Swarm Intelligence...")
    
    try:
        from swarm_intelligence import RealSwarmIntelligence
        from ai_brain import IntelligentAIBrain
        
        # Create AI brain mock
        class MockAIBrain:
            async def generate_enhanced_response(self, prompt, model=None):
                return {
                    'content': json.dumps({
                        'key_points': ['Mock insight 1', 'Mock insight 2'],
                        'challenges': ['Mock challenge 1'],
                        'recommendations': ['Mock recommendation 1'],
                        'priority': 7,
                        'complexity': 'medium',
                        'confidence': 0.8
                    }),
                    'provider': 'mock'
                }
        
        # Create swarm with mock brain
        mock_brain = MockAIBrain()
        swarm = RealSwarmIntelligence(ai_brain=mock_brain, num_agents=3)
        
        # Test task processing
        test_task = {
            'id': 'test_task_1',
            'type': 'feature_development',
            'title': 'Test Feature',
            'description': 'A test feature for validation',
            'requirements': ['requirement1', 'requirement2']
        }
        
        start_time = time.time()
        result = await swarm.process_task_swarm(test_task)
        duration = time.time() - start_time
        
        if result and 'collective_review' in result:
            log_test('functionality', 'swarm_task_processing', True, f"Processed in {duration:.2f}s")
        else:
            log_test('functionality', 'swarm_task_processing', False, "Task processing failed")
        
        # Test performance metrics
        if 'performance_metrics' in result:
            log_test('functionality', 'swarm_performance_tracking', True, "Performance metrics available")
        else:
            log_test('functionality', 'swarm_performance_tracking', False, "No performance metrics")
        
        # Test cache functionality
        cache_test_task = {
            'id': 'test_task_2',
            'type': 'feature_development',
            'title': 'Test Feature',  # Same title for cache hit
            'description': 'A test feature for validation',
            'requirements': ['requirement1', 'requirement2']
        }
        
        cache_start = time.time()
        cache_result = await swarm.process_task_swarm(cache_test_task)
        cache_duration = time.time() - cache_start
        
        # Should be faster due to caching
        if cache_duration < duration:
            log_test('performance', 'swarm_caching', True, f"Cache speedup: {duration:.2f}s -> {cache_duration:.2f}s")
        else:
            log_test('performance', 'swarm_caching', False, f"No cache speedup detected")
        
        # Test swarm status
        status = swarm.get_swarm_status()
        if 'active_agents' in status and status['active_agents'] > 0:
            log_test('functionality', 'swarm_status', True, f"Status: {status['active_agents']} agents")
        else:
            log_test('functionality', 'swarm_status', False, "Invalid status")
        
    except Exception as e:
        log_test('functionality', 'swarm_operations', False, f"Swarm test failed: {e}")

async def test_integration():
    """Test component integration."""
    print("\nTesting Component Integration...")
    
    try:
        from intelligence_integration import IntelligenceIntegrator
        from intelligence_hub import get_intelligence_hub
        
        # Create integrator
        integrator = IntelligenceIntegrator()
        success = await integrator.initialize()
        
        if success:
            log_test('integration', 'integrator_initialization', True, "Integrator initialized")
        else:
            log_test('integration', 'integrator_initialization', False, "Initialization failed")
        
        # Test status
        status = await integrator.get_integration_status()
        if status['integration_available']:
            log_test('integration', 'integration_status', True, f"Components: {status['total_components']}")
        else:
            log_test('integration', 'integration_status', False, "Integration not available")
        
        # Cleanup
        await integrator.shutdown()
        log_test('integration', 'integrator_shutdown', True, "Shutdown completed")
        
    except Exception as e:
        log_test('integration', 'integration_test', False, f"Integration test failed: {e}")

async def test_performance():
    """Test performance improvements."""
    print("\nTesting Performance Improvements...")
    
    try:
        from ai_response_cache import AIResponseCache
        
        # Test cache performance
        cache = AIResponseCache(max_size=1000)
        
        # Measure cache put performance
        start_time = time.time()
        for i in range(100):
            await cache.put(f"prompt_{i}", f"response_{i}", "anthropic", "claude")
        put_duration = time.time() - start_time
        
        # Measure cache get performance
        start_time = time.time()
        hits = 0
        for i in range(100):
            result = await cache.get(f"prompt_{i}", "anthropic", "claude")
            if result:
                hits += 1
        get_duration = time.time() - start_time
        
        log_test('performance', 'cache_operations', True, 
                f"Put: {put_duration:.3f}s, Get: {get_duration:.3f}s, Hits: {hits}/100")
        
        # Test semantic similarity (if available)
        similar_result = await cache.get("prompt_1_similar", "anthropic", "claude")
        if similar_result:
            log_test('performance', 'semantic_similarity', True, "Semantic matching works")
        else:
            log_test('performance', 'semantic_similarity', False, "No semantic matching")
        
        await cache.shutdown()
        
    except Exception as e:
        log_test('performance', 'performance_test', False, f"Performance test failed: {e}")

async def run_all_tests():
    """Run all tests."""
    print("🧪 Running Intelligence Enhancement Tests")
    print("=" * 50)
    
    await test_imports()
    await test_ai_response_cache()
    await test_async_state_manager()
    await test_intelligence_hub()
    await test_enhanced_swarm()
    await test_integration()
    await test_performance()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in test_results.items():
        category_passed = sum(1 for test in tests.values() if test['result'])
        category_total = len(tests)
        total_tests += category_total
        passed_tests += category_passed
        
        print(f"{category.upper()}: {category_passed}/{category_total} passed")
        
        for test_name, test_data in tests.items():
            status = "✅" if test_data['result'] else "❌"
            print(f"  {status} {test_name}: {test_data['details'] or ''}")
    
    print("-" * 50)
    print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Intelligence enhancements are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the results above for details.")
        return False

def main():
    """Main test function."""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()