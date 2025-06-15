"""
Test Enhanced Worker Intelligence System

Comprehensive tests and validation for the enhanced worker intelligence system.
Tests all components integration, performance, error handling, and intelligence features.
"""

import asyncio
import pytest
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import tempfile
import os

# Import the enhanced intelligence system
from scripts.worker_intelligence_integration import (
    WorkerIntelligenceCoordinator,
    WorkerEnhancementConfig,
    IntelligentWorkerMixin
)
from scripts.worker_logging_config import LogLevel
from scripts.worker_intelligence_hub import WorkerSpecialization
from scripts.enhanced_swarm_intelligence import (
    EnhancedSwarmIntelligence,
    create_enhanced_swarm
)
from scripts.worker_status_reporter import AlertSeverity, AlertType


class MockAIBrain:
    """Mock AI brain for testing."""
    
    def __init__(self, response_delay: float = 0.1):
        self.response_delay = response_delay
        self.call_count = 0
        self.failures_enabled = False
    
    async def generate_enhanced_response(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Generate mock AI response."""
        self.call_count += 1
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        if self.failures_enabled and self.call_count % 3 == 0:
            raise Exception("Simulated AI failure")
        
        # Generate realistic response based on prompt content
        if "analysis" in prompt.lower():
            return {
                'content': json.dumps({
                    'key_points': [f"Key insight {self.call_count}", "Technical consideration"],
                    'challenges': [f"Challenge {self.call_count}", "Implementation complexity"],
                    'recommendations': [f"Recommendation {self.call_count}", "Best practice"],
                    'priority': min(10, max(1, 5 + (self.call_count % 6))),
                    'complexity': 'medium',
                    'confidence': 0.8,
                    'alignment_score': 0.9
                })
            }
        else:
            return {
                'content': json.dumps({
                    'key_insights': [f"Insight {self.call_count}"],
                    'critical_challenges': [f"Critical challenge {self.call_count}"],
                    'top_recommendations': [f"Top recommendation {self.call_count}"],
                    'consensus_priority': 7,
                    'consensus_alignment': 0.8,
                    'success_probability': 0.75
                })
            }


class MockWorker:
    """Mock worker class for testing integration."""
    
    def __init__(self, name: str):
        self.name = name
        self.task_count = 0
        self.should_fail = False
    
    def process_task(self, task_data: str) -> str:
        """Process a task."""
        self.task_count += 1
        
        if self.should_fail:
            raise ValueError(f"Simulated failure in {self.name}")
        
        return f"Processed '{task_data}' by {self.name} (task #{self.task_count})"
    
    async def async_process_task(self, task_data: str) -> str:
        """Process a task asynchronously."""
        await asyncio.sleep(0.05)  # Simulate work
        return self.process_task(task_data)


class TestWorkerIntelligenceComponents:
    """Test individual intelligence components."""
    
    @pytest.mark.asyncio
    async def test_intelligence_coordinator_lifecycle(self):
        """Test intelligence coordinator startup and shutdown."""
        coordinator = WorkerIntelligenceCoordinator()
        
        # Test startup
        await coordinator.start()
        
        # Verify components are running
        assert coordinator.intelligence_hub is not None
        assert coordinator.metrics_collector is not None
        assert coordinator.error_analyzer is not None
        assert coordinator.work_item_tracker is not None
        assert coordinator.status_reporter is not None
        
        # Test shutdown
        await coordinator.stop()
        
        # Verify clean shutdown
        assert coordinator.intelligence_hub._shutdown
        assert coordinator.metrics_collector._shutdown
        assert coordinator.error_analyzer._shutdown
        assert coordinator.work_item_tracker._shutdown
        assert coordinator.status_reporter._shutdown
    
    @pytest.mark.asyncio
    async def test_worker_enhancement(self):
        """Test worker enhancement with intelligence capabilities."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create and enhance a mock worker
            original_worker = MockWorker("test_worker")
            enhanced_worker = coordinator.enhance_worker(
                original_worker, 
                "enhanced_test_worker",
                WorkerEnhancementConfig(
                    worker_specialization=WorkerSpecialization.GENERAL,
                    log_level=LogLevel.DEBUG
                )
            )
            
            # Test that enhancement worked
            assert hasattr(enhanced_worker, 'intelligent_task_execution')
            assert hasattr(enhanced_worker, 'report_performance_metrics')
            assert enhanced_worker.worker_id == "enhanced_test_worker"
            assert enhanced_worker.is_initialized
            
            # Test intelligent task execution
            async with enhanced_worker.intelligent_task_execution(
                "test_task_1", "data_processing", {"test": True}
            ) as context:
                result = enhanced_worker.process_task("test_data")
                assert "test_data" in result
                assert enhanced_worker.task_count > 0
            
            # Test performance metrics
            metrics = enhanced_worker.report_performance_metrics()
            assert metrics['worker_id'] == "enhanced_test_worker"
            assert metrics['tasks_completed'] > 0
            assert metrics['error_rate'] == 0.0  # No errors yet
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_analysis(self):
        """Test error handling and analysis capabilities."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create worker that can fail
            original_worker = MockWorker("failing_worker")
            enhanced_worker = coordinator.enhance_worker(
                original_worker, "failing_enhanced_worker"
            )
            
            # Enable failures
            enhanced_worker.should_fail = True
            
            # Test error capture
            with pytest.raises(ValueError):
                async with enhanced_worker.intelligent_task_execution(
                    "failing_task", "error_test"
                ):
                    enhanced_worker.process_task("will_fail")
            
            # Verify error was registered
            assert enhanced_worker.error_count > 0
            
            # Check error analyzer
            error_summary = coordinator.error_analyzer.get_error_summary(hours=1)
            assert error_summary['total_errors'] > 0
            
            # Get worker-specific error analysis
            worker_analysis = coordinator.error_analyzer.get_worker_error_analysis(
                "failing_enhanced_worker"
            )
            assert worker_analysis['total_errors'] > 0
            assert worker_analysis['worker_id'] == "failing_enhanced_worker"
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection and performance tracking."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create enhanced worker
            original_worker = MockWorker("metrics_worker")
            enhanced_worker = coordinator.enhance_worker(
                original_worker, "metrics_enhanced_worker"
            )
            
            # Execute multiple tasks
            for i in range(5):
                async with enhanced_worker.intelligent_task_execution(
                    f"metrics_task_{i}", "performance_test", {"iteration": i}
                ):
                    result = await enhanced_worker.async_process_task(f"data_{i}")
                    assert f"data_{i}" in result
            
            # Wait for metrics collection
            await asyncio.sleep(1)
            
            # Check system dashboard
            dashboard = coordinator.get_system_dashboard()
            assert dashboard is not None
            assert 'system_health' in dashboard
            assert 'workers' in dashboard
            
            # Check worker performance
            worker_performance = coordinator.get_worker_performance("metrics_enhanced_worker")
            assert worker_performance['tasks_completed'] == 5
            assert worker_performance['error_rate'] == 0.0
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_work_item_tracking(self):
        """Test work item lifecycle tracking."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create work item directly
            work_item_id = coordinator.work_item_tracker.create_work_item(
                title="Test Work Item",
                description="Testing work item tracking",
                work_type="test_task",
                estimated_duration=30.0
            )
            
            # Assign and process work item
            coordinator.work_item_tracker.assign_to_worker(work_item_id, "test_worker")
            coordinator.work_item_tracker.start_work(work_item_id, "test_worker")
            
            # Update progress
            coordinator.work_item_tracker.update_progress(work_item_id, 50, "Halfway done")
            
            # Complete work item
            coordinator.work_item_tracker.complete_work(work_item_id, "test_worker", {
                "result": "success",
                "output": "test_output"
            })
            
            # Verify work item lifecycle
            work_item = coordinator.work_item_tracker.get_work_item(work_item_id)
            assert work_item is None  # Should be moved to completed items
            
            # Check analytics
            analytics = coordinator.work_item_tracker.get_analytics_summary()
            assert analytics['summary']['completed_items'] > 0
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_status_reporting_and_alerts(self):
        """Test status reporting and alerting system."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Wait for initial monitoring cycle
            await asyncio.sleep(2)
            
            # Create custom alert
            alert_id = coordinator.create_alert(
                AlertType.CUSTOM,
                AlertSeverity.WARNING,
                "Test Alert",
                "This is a test alert for validation",
                ["test_worker"]
            )
            
            # Verify alert was created
            alerts = coordinator.status_reporter.get_alerts()
            assert len(alerts) > 0
            assert any(alert.alert_id == alert_id for alert in alerts)
            
            # Acknowledge alert
            success = coordinator.status_reporter.acknowledge_alert(alert_id, "test_user")
            assert success
            
            # Resolve alert
            success = coordinator.status_reporter.resolve_alert(alert_id, "test_user")
            assert success
            
            # Verify alert is resolved
            alerts = coordinator.status_reporter.get_alerts()
            assert not any(alert.alert_id == alert_id for alert in alerts)
            
        finally:
            await coordinator.stop()


class TestEnhancedSwarmIntelligence:
    """Test enhanced swarm intelligence integration."""
    
    @pytest.mark.asyncio
    async def test_enhanced_swarm_creation(self):
        """Test creation of enhanced swarm intelligence."""
        ai_brain = MockAIBrain(response_delay=0.01)
        
        enhanced_swarm, coordinator = create_enhanced_swarm(
            ai_brain, enable_coordinator=True
        )
        
        assert enhanced_swarm is not None
        assert coordinator is not None
        assert len(enhanced_swarm.agents) > 0
        
        # Verify agents are enhanced
        for agent in enhanced_swarm.agents:
            assert hasattr(agent, 'intelligent_task_execution')
            assert hasattr(agent, 'report_performance_metrics')
        
        # Start coordinator
        await coordinator.start()
        
        try:
            # Verify intelligence components are connected
            assert enhanced_swarm.intelligence_coordinator == coordinator
            assert enhanced_swarm.is_initialized
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_enhanced_swarm_task_processing(self):
        """Test enhanced swarm task processing with intelligence tracking."""
        ai_brain = MockAIBrain(response_delay=0.01)
        enhanced_swarm, coordinator = create_enhanced_swarm(ai_brain, enable_coordinator=True)
        
        await coordinator.start()
        
        try:
            # Create test task
            test_task = {
                'id': 'enhanced_test_task',
                'type': 'feature_analysis',
                'title': 'Enhanced Feature Analysis',
                'description': 'Testing enhanced swarm analysis',
                'requirements': ['Performance', 'Scalability', 'Security']
            }
            
            # Process task with enhanced swarm
            result = await enhanced_swarm.process_task_swarm(test_task)
            
            # Verify result structure
            assert result is not None
            assert 'task_id' in result
            assert 'individual_analyses' in result
            assert 'consensus' in result
            assert 'duration_seconds' in result
            
            # Verify intelligence metrics are included
            if 'intelligence_metrics' in result:
                assert 'system_health' in result['intelligence_metrics']
                assert 'swarm_performance' in result['intelligence_metrics']
            
            # Verify all agents participated
            individual_analyses = result.get('individual_analyses', [])
            assert len(individual_analyses) == len(enhanced_swarm.agents)
            
            # Check that work items were created and tracked
            analytics = coordinator.work_item_tracker.get_analytics_summary()
            assert analytics['summary']['completed_items'] > 0
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_enhanced_swarm_error_resilience(self):
        """Test enhanced swarm resilience to errors."""
        ai_brain = MockAIBrain(response_delay=0.01)
        ai_brain.failures_enabled = True  # Enable random failures
        
        enhanced_swarm, coordinator = create_enhanced_swarm(ai_brain, enable_coordinator=True)
        await coordinator.start()
        
        try:
            # Create test task
            test_task = {
                'id': 'error_resilience_test',
                'type': 'error_test',
                'title': 'Error Resilience Test',
                'description': 'Testing swarm resilience to AI failures'
            }
            
            # Process task (some agents may fail)
            result = await enhanced_swarm.process_task_swarm(test_task)
            
            # Verify we still get a result despite failures
            assert result is not None
            assert 'consensus' in result
            
            # Check error analysis
            error_summary = coordinator.error_analyzer.get_error_summary(hours=1)
            if error_summary['total_errors'] > 0:
                print(f"Handled {error_summary['total_errors']} errors gracefully")
            
            # Verify system status is still functional
            dashboard = coordinator.get_system_dashboard()
            assert dashboard is not None
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_enhanced_swarm_analytics(self):
        """Test enhanced swarm analytics and insights."""
        ai_brain = MockAIBrain(response_delay=0.01)
        enhanced_swarm, coordinator = create_enhanced_swarm(ai_brain, enable_coordinator=True)
        
        await coordinator.start()
        
        try:
            # Process multiple tasks
            for i in range(3):
                test_task = {
                    'id': f'analytics_test_{i}',
                    'type': 'analytics_test',
                    'title': f'Analytics Test {i}',
                    'description': f'Testing analytics collection {i}'
                }
                
                await enhanced_swarm.process_task_swarm(test_task)
            
            # Get enhanced analytics
            analytics = enhanced_swarm.get_enhanced_analytics()
            
            # Verify analytics structure
            assert 'intelligence_integration' in analytics
            assert 'components_enabled' in analytics['intelligence_integration']
            
            # Check that intelligence components are properly integrated
            components = analytics['intelligence_integration']['components_enabled']
            assert components['logging'] == True
            assert components['intelligence_hub'] == True
            assert components['metrics'] == True
            
            # Verify additional analytics from intelligence components
            if 'intelligence_hub_status' in analytics:
                assert 'system_metrics' in analytics['intelligence_hub_status']
            
            if 'performance_dashboard' in analytics:
                assert 'system_overview' in analytics['performance_dashboard']
            
        finally:
            await coordinator.stop()


class TestSystemIntegration:
    """Test full system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self):
        """Test complete workflow from worker creation to task completion."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create multiple enhanced workers
            workers = []
            for i in range(3):
                original_worker = MockWorker(f"worker_{i}")
                enhanced_worker = coordinator.enhance_worker(
                    original_worker, 
                    f"enhanced_worker_{i}",
                    WorkerEnhancementConfig(
                        worker_specialization=WorkerSpecialization.GENERAL
                    )
                )
                workers.append(enhanced_worker)
            
            # Simulate concurrent task execution
            tasks = []
            for i in range(10):
                worker = workers[i % len(workers)]
                task = asyncio.create_task(
                    self._execute_worker_task(worker, f"concurrent_task_{i}")
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0
            
            # Wait for metrics collection
            await asyncio.sleep(2)
            
            # Verify system health
            dashboard = coordinator.get_system_dashboard()
            assert dashboard['system_health']['status'] in ['healthy', 'degraded', 'warning']
            
            # Verify all workers are tracked
            assert dashboard['system_health']['total_workers'] == len(workers)
            
            # Check work item tracking
            analytics = coordinator.work_item_tracker.get_analytics_summary()
            assert analytics['summary']['completed_items'] > 0
            
        finally:
            await coordinator.stop()
    
    async def _execute_worker_task(self, worker, task_id: str) -> str:
        """Execute a task on a worker."""
        async with worker.intelligent_task_execution(
            task_id, "integration_test", {"test_type": "concurrent"}
        ):
            return await worker.async_process_task(f"data_for_{task_id}")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        coordinator = WorkerIntelligenceCoordinator()
        await coordinator.start()
        
        try:
            # Create enhanced swarm
            ai_brain = MockAIBrain(response_delay=0.001)  # Very fast responses
            enhanced_swarm, _ = create_enhanced_swarm(ai_brain, enable_coordinator=False)
            enhanced_swarm.intelligence_coordinator = coordinator
            enhanced_swarm.set_intelligence_components(
                coordinator.intelligence_hub,
                coordinator.metrics_collector,
                coordinator.error_analyzer,
                coordinator.work_item_tracker,
                coordinator.status_reporter
            )
            
            # Execute multiple tasks rapidly
            start_time = time.time()
            task_count = 20
            
            tasks = []
            for i in range(task_count):
                test_task = {
                    'id': f'load_test_{i}',
                    'type': 'load_test',
                    'title': f'Load Test {i}',
                    'description': f'Performance test task {i}'
                }
                
                task = asyncio.create_task(
                    enhanced_swarm.process_task_swarm(test_task)
                )
                tasks.append(task)
            
            # Wait for completion
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze performance
            duration = end_time - start_time
            successful_results = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful_results) / len(results)
            throughput = len(successful_results) / duration
            
            print(f"Performance Test Results:")
            print(f"  Tasks: {task_count}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Throughput: {throughput:.1f} tasks/second")
            
            # Verify acceptable performance
            assert success_rate > 0.8  # At least 80% success rate
            assert throughput > 1.0     # At least 1 task per second
            
            # Verify system stability
            dashboard = coordinator.get_system_dashboard()
            assert dashboard['system_health']['status'] != 'critical'
            
        finally:
            await coordinator.stop()


# Main test runner
async def run_all_tests():
    """Run all tests and provide comprehensive validation report."""
    print("🧪 Starting Enhanced Worker Intelligence System Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    # Component tests
    print("\n📋 Testing Individual Components...")
    component_tests = TestWorkerIntelligenceComponents()
    
    try:
        await component_tests.test_intelligence_coordinator_lifecycle()
        print("✅ Intelligence coordinator lifecycle test passed")
        
        await component_tests.test_worker_enhancement()
        print("✅ Worker enhancement test passed")
        
        await component_tests.test_error_handling_and_analysis()
        print("✅ Error handling and analysis test passed")
        
        await component_tests.test_metrics_collection()
        print("✅ Metrics collection test passed")
        
        await component_tests.test_work_item_tracking()
        print("✅ Work item tracking test passed")
        
        await component_tests.test_status_reporting_and_alerts()
        print("✅ Status reporting and alerts test passed")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False
    
    # Enhanced swarm tests
    print("\n🤖 Testing Enhanced Swarm Intelligence...")
    swarm_tests = TestEnhancedSwarmIntelligence()
    
    try:
        await swarm_tests.test_enhanced_swarm_creation()
        print("✅ Enhanced swarm creation test passed")
        
        await swarm_tests.test_enhanced_swarm_task_processing()
        print("✅ Enhanced swarm task processing test passed")
        
        await swarm_tests.test_enhanced_swarm_error_resilience()
        print("✅ Enhanced swarm error resilience test passed")
        
        await swarm_tests.test_enhanced_swarm_analytics()
        print("✅ Enhanced swarm analytics test passed")
        
    except Exception as e:
        print(f"❌ Enhanced swarm test failed: {e}")
        return False
    
    # Integration tests
    print("\n🔗 Testing System Integration...")
    integration_tests = TestSystemIntegration()
    
    try:
        await integration_tests.test_complete_workflow_integration()
        print("✅ Complete workflow integration test passed")
        
        await integration_tests.test_performance_under_load()
        print("✅ Performance under load test passed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print(f"⏱️  Total test duration: {duration:.2f} seconds")
    print("\n📊 Test Summary:")
    print("  ✅ All intelligence components working correctly")
    print("  ✅ Worker enhancement and integration functional")
    print("  ✅ Error handling and recovery operational")
    print("  ✅ Metrics collection and analysis working")
    print("  ✅ Work item tracking and lifecycle management active")
    print("  ✅ Status reporting and alerting system functional")
    print("  ✅ Enhanced swarm intelligence operational")
    print("  ✅ System integration and performance validated")
    
    print("\n🚀 The Enhanced Worker Intelligence System is ready for production!")
    return True


if __name__ == "__main__":
    # Run validation tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)