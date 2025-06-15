"""
Test Redis Week 3 Streams Implementation

Comprehensive test suite for Redis Streams intelligence hub, event sourcing,
distributed workflows, integration, and analytics implementation.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

# Test imports
from scripts.redis_intelligence_hub import (
    RedisIntelligenceHub, IntelligenceEvent, EventType, EventPriority,
    get_intelligence_hub, create_intelligence_hub
)
from scripts.redis_event_sourcing import (
    RedisEventStore, EventSnapshot, SnapshotStrategy,
    get_event_store, create_event_store
)
from scripts.redis_distributed_workflows import (
    RedisWorkflowEngine, WorkflowDefinition, WorkflowTask, WorkflowTrigger,
    get_workflow_engine, create_workflow_engine
)
from scripts.redis_streams_integration import (
    RedisIntelligenceIntegrator, EnhancedWorkerIntelligenceAdapter,
    get_intelligence_integrator, enhance_existing_worker
)
from scripts.redis_event_analytics import (
    RedisEventAnalytics, AnalyticsMetric, EventPattern, AnomalyDetector,
    get_event_analytics, create_event_analytics
)


class RedisWeek3StreamsTester:
    """Comprehensive tester for Redis Week 3 Streams implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        self.start_time = None
        
        # Test configuration
        self.test_config = {
            'test_workers': 5,
            'test_events': 50,
            'test_workflows': 3,
            'test_tasks_per_workflow': 4,
            'analytics_samples': 100,
            'test_duration_seconds': 30
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite for Redis Week 3 Streams implementation."""
        self.start_time = time.time()
        self.logger.info("Starting Redis Week 3 Streams comprehensive test suite")
        
        try:
            # Test 1: Intelligence Hub
            await self._test_intelligence_hub()
            
            # Test 2: Event Sourcing
            await self._test_event_sourcing()
            
            # Test 3: Distributed Workflows
            await self._test_distributed_workflows()
            
            # Test 4: Streams Integration
            await self._test_streams_integration()
            
            # Test 5: Event Analytics
            await self._test_event_analytics()
            
            # Test 6: End-to-End Integration
            await self._test_end_to_end_integration()
            
            # Test 7: Performance and Scalability
            await self._test_performance_scalability()
            
            # Generate final report
            return self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive tests: {e}")
            self._add_test_result("comprehensive_tests", False, str(e))
            return self._generate_test_report()
    
    async def _test_intelligence_hub(self):
        """Test Redis Intelligence Hub functionality."""
        test_name = "intelligence_hub"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize intelligence hub
            hub = await create_intelligence_hub(
                hub_id="test_hub",
                enable_analytics=True,
                enable_event_sourcing=True
            )
            
            # Test worker registration
            registration_id = await hub.register_worker("test_worker_1", ["task_type_a", "task_type_b"])
            
            if registration_id:
                self._add_test_result(f"{test_name}_worker_registration", True, "Worker registration successful")
            else:
                self._add_test_result(f"{test_name}_worker_registration", False, "Worker registration failed")
            
            # Test event publishing
            test_event = IntelligenceEvent(
                event_id="test_event_1",
                event_type=EventType.PERFORMANCE_METRIC,
                worker_id="test_worker_1",
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={'metric_type': 'cpu_usage', 'value': 75.5}
            )
            
            event_id = await hub.publish_event(test_event)
            
            if event_id:
                self._add_test_result(f"{test_name}_event_publishing", True, f"Event published: {event_id}")
            else:
                self._add_test_result(f"{test_name}_event_publishing", False, "Event publishing failed")
            
            # Test heartbeat
            heartbeat_id = await hub.worker_heartbeat("test_worker_1", {'status': 'active', 'load': 0.3})
            
            if heartbeat_id:
                self._add_test_result(f"{test_name}_heartbeat", True, "Heartbeat successful")
            else:
                self._add_test_result(f"{test_name}_heartbeat", False, "Heartbeat failed")
            
            # Test task assignment
            task_id = await hub.assign_task(
                "test_worker_1", 
                "task_123", 
                "data_processing",
                {'input_data': 'test_data'},
                EventPriority.HIGH
            )
            
            if task_id:
                self._add_test_result(f"{test_name}_task_assignment", True, "Task assignment successful")
            else:
                self._add_test_result(f"{test_name}_task_assignment", False, "Task assignment failed")
            
            # Test task completion
            completion_id = await hub.report_task_completion(
                "test_worker_1",
                "task_123",
                {'output': 'processed_data', 'status': 'success'},
                5.5
            )
            
            if completion_id:
                self._add_test_result(f"{test_name}_task_completion", True, "Task completion successful")
            else:
                self._add_test_result(f"{test_name}_task_completion", False, "Task completion failed")
            
            # Test hub statistics
            stats = await hub.get_hub_statistics()
            
            if stats and 'hub_id' in stats:
                self._add_test_result(f"{test_name}_statistics", True, f"Hub statistics available: {stats['active_workers']} workers")
            else:
                self._add_test_result(f"{test_name}_statistics", False, "Hub statistics unavailable")
            
            # Cleanup
            await hub.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_event_sourcing(self):
        """Test Redis Event Sourcing functionality."""
        test_name = "event_sourcing"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize event store
            event_store = await create_event_store(
                store_id="test_store",
                snapshot_strategy=SnapshotStrategy.EVENT_COUNT,
                snapshot_frequency=10,
                enable_projections=True
            )
            
            # Test event appending
            test_events = []
            for i in range(self.test_config['test_events']):
                event = IntelligenceEvent(
                    event_id=f"test_event_{i}",
                    event_type=EventType.TASK_COMPLETION if i % 3 == 0 else EventType.PERFORMANCE_METRIC,
                    worker_id=f"worker_{i % 3}",
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.NORMAL,
                    data={'sequence': i, 'test_data': f'data_{i}'}
                )
                
                sequence = await event_store.append_event(event)
                test_events.append((event, sequence))
            
            if len(test_events) == self.test_config['test_events']:
                self._add_test_result(f"{test_name}_event_appending", True, f"Appended {len(test_events)} events")
            else:
                self._add_test_result(f"{test_name}_event_appending", False, "Event appending incomplete")
            
            # Test event replay
            replayed_state = await event_store.replay_events("worker_0")
            
            if replayed_state and isinstance(replayed_state, dict):
                self._add_test_result(f"{test_name}_event_replay", True, f"State replayed with {len(replayed_state)} keys")
            else:
                self._add_test_result(f"{test_name}_event_replay", False, "Event replay failed")
            
            # Test snapshot creation
            snapshot = await event_store.create_snapshot("worker_0", force=True)
            
            if snapshot and snapshot.entity_id == "worker_0":
                self._add_test_result(f"{test_name}_snapshot_creation", True, f"Snapshot created: {snapshot.snapshot_id}")
            else:
                self._add_test_result(f"{test_name}_snapshot_creation", False, "Snapshot creation failed")
            
            # Test event filtering
            task_events = []
            async for event in event_store.get_events(
                event_types=[EventType.TASK_COMPLETION],
                limit=10
            ):
                task_events.append(event)
            
            if task_events:
                self._add_test_result(f"{test_name}_event_filtering", True, f"Filtered {len(task_events)} task events")
            else:
                self._add_test_result(f"{test_name}_event_filtering", False, "Event filtering failed")
            
            # Test metrics
            metrics = await event_store.get_metrics()
            
            if metrics and 'store_id' in metrics:
                self._add_test_result(f"{test_name}_metrics", True, f"Event store metrics available")
            else:
                self._add_test_result(f"{test_name}_metrics", False, "Event store metrics unavailable")
            
            # Cleanup
            await event_store.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_distributed_workflows(self):
        """Test Redis Distributed Workflows functionality."""
        test_name = "distributed_workflows"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize workflow engine
            workflow_engine = await create_workflow_engine(
                engine_id="test_engine",
                max_concurrent_workflows=10,
                enable_fault_tolerance=True,
                enable_load_balancing=True
            )
            
            # Create test workflow
            workflow_tasks = []
            for i in range(self.test_config['test_tasks_per_workflow']):
                task = WorkflowTask(
                    task_id=f"task_{i}",
                    name=f"Test Task {i}",
                    task_type="test_task",
                    parameters={'step': i, 'data': f'test_data_{i}'},
                    dependencies=[f"task_{i-1}"] if i > 0 else [],
                    max_retries=2,
                    timeout_seconds=300
                )
                workflow_tasks.append(task)
            
            workflow = WorkflowDefinition(
                workflow_id="test_workflow",
                name="Test Workflow",
                description="Test workflow for Redis Streams",
                tasks=workflow_tasks,
                trigger=WorkflowTrigger.MANUAL,
                enable_parallel_execution=True,
                failure_strategy="retry_failed"
            )
            
            # Register workflow
            await workflow_engine.register_workflow(workflow)
            self._add_test_result(f"{test_name}_workflow_registration", True, "Workflow registered successfully")
            
            # Start workflow execution
            execution_id = await workflow_engine.start_workflow("test_workflow", {'test_input': 'data'})
            
            if execution_id:
                self._add_test_result(f"{test_name}_workflow_start", True, f"Workflow started: {execution_id}")
            else:
                self._add_test_result(f"{test_name}_workflow_start", False, "Workflow start failed")
            
            # Wait briefly for processing
            await asyncio.sleep(2)
            
            # Check workflow status
            status = await workflow_engine.get_workflow_status(execution_id)
            
            if status and 'execution_id' in status:
                self._add_test_result(f"{test_name}_workflow_status", True, f"Workflow status: {status['status']}")
            else:
                self._add_test_result(f"{test_name}_workflow_status", False, "Workflow status unavailable")
            
            # Test engine metrics
            metrics = await workflow_engine.get_engine_metrics()
            
            if metrics and 'engine_id' in metrics:
                self._add_test_result(f"{test_name}_engine_metrics", True, f"Engine metrics available")
            else:
                self._add_test_result(f"{test_name}_engine_metrics", False, "Engine metrics unavailable")
            
            # Cleanup
            await workflow_engine.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_streams_integration(self):
        """Test Redis Streams Integration functionality."""
        test_name = "streams_integration"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize integrator
            integrator = await create_intelligence_integrator(
                enable_legacy_bridge=True,
                enable_event_migration=True,
                enable_real_time_sync=True
            )
            
            # Test integration status
            status = await integrator.get_integration_status()
            
            if status and 'integrator_status' in status:
                self._add_test_result(f"{test_name}_integrator_status", True, f"Integrator active with {status['components_integrated']} components")
            else:
                self._add_test_result(f"{test_name}_integrator_status", False, "Integrator status unavailable")
            
            # Test enhanced worker adapter
            worker_adapter = await enhance_existing_worker(
                "integration_test_worker",
                ["integration_task", "analysis_task"]
            )
            
            # Test worker registration
            await worker_adapter.register_worker(["enhanced_capability"])
            self._add_test_result(f"{test_name}_worker_enhancement", True, "Worker enhanced successfully")
            
            # Test enhanced heartbeat
            await worker_adapter.send_heartbeat({'cpu_usage': 45.2, 'memory_usage': 67.8})
            self._add_test_result(f"{test_name}_enhanced_heartbeat", True, "Enhanced heartbeat successful")
            
            # Test intelligence update
            await worker_adapter.update_intelligence({
                'skill_level': 'advanced',
                'specialization': 'stream_processing',
                'efficiency_score': 0.92
            })
            self._add_test_result(f"{test_name}_intelligence_update", True, "Intelligence update successful")
            
            # Test worker analytics
            analytics = await worker_adapter.get_worker_analytics()
            
            if analytics and 'worker_id' in analytics:
                self._add_test_result(f"{test_name}_worker_analytics", True, f"Worker analytics available")
            else:
                self._add_test_result(f"{test_name}_worker_analytics", False, "Worker analytics unavailable")
            
            # Cleanup
            await integrator.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_event_analytics(self):
        """Test Redis Event Analytics functionality."""
        test_name = "event_analytics"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize analytics engine
            analytics = await create_event_analytics(
                analytics_id="test_analytics",
                enable_real_time=True,
                enable_pattern_detection=True,
                enable_anomaly_detection=True,
                enable_predictive_analytics=True
            )
            
            # Generate test data for analytics
            hub = await get_intelligence_hub()
            
            for i in range(self.test_config['analytics_samples']):
                # Generate performance metrics
                performance_event = IntelligenceEvent(
                    event_id=f"perf_event_{i}",
                    event_type=EventType.PERFORMANCE_METRIC,
                    worker_id=f"analytics_worker_{i % 3}",
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.NORMAL,
                    data={
                        'metric_type': 'response_time',
                        'value': 100 + (i % 50),  # Varying response times
                        'metadata': {'test_sample': i}
                    }
                )
                await hub.publish_event(performance_event)
                
                # Generate task events
                if i % 5 == 0:
                    task_event = IntelligenceEvent(
                        event_id=f"task_event_{i}",
                        event_type=EventType.TASK_COMPLETION,
                        worker_id=f"analytics_worker_{i % 3}",
                        timestamp=datetime.now(timezone.utc),
                        priority=EventPriority.NORMAL,
                        data={
                            'task_id': f"analytics_task_{i}",
                            'duration_seconds': 2.5 + (i % 10) * 0.5,
                            'result': {'status': 'success', 'processed_items': i * 10}
                        }
                    )
                    await hub.publish_event(task_event)
            
            # Wait for analytics processing
            await asyncio.sleep(5)
            
            # Test analytics report
            report = await analytics.get_analytics_report()
            
            if report and 'analytics_id' in report:
                self._add_test_result(f"{test_name}_analytics_report", True, f"Analytics report generated")
            else:
                self._add_test_result(f"{test_name}_analytics_report", False, "Analytics report unavailable")
            
            # Check if metrics were processed
            if 'analytics_performance' in report:
                events_processed = report['analytics_performance'].get('events_processed', 0)
                if events_processed > 0:
                    self._add_test_result(f"{test_name}_event_processing", True, f"Processed {events_processed} events")
                else:
                    self._add_test_result(f"{test_name}_event_processing", False, "No events processed")
            
            # Check metrics summary
            if 'metrics_summary' in report and report['metrics_summary']:
                self._add_test_result(f"{test_name}_metrics_summary", True, f"Metrics summary available")
            else:
                self._add_test_result(f"{test_name}_metrics_summary", False, "Metrics summary unavailable")
            
            # Check pattern analysis
            if 'pattern_analysis' in report:
                self._add_test_result(f"{test_name}_pattern_analysis", True, "Pattern analysis available")
            else:
                self._add_test_result(f"{test_name}_pattern_analysis", False, "Pattern analysis unavailable")
            
            # Cleanup
            await analytics.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_end_to_end_integration(self):
        """Test end-to-end integration of all Redis Streams components."""
        test_name = "end_to_end_integration"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize all components
            hub = await get_intelligence_hub()
            event_store = await get_event_store()
            workflow_engine = await get_workflow_engine()
            integrator = await get_intelligence_integrator()
            analytics = await get_event_analytics()
            
            # Create integrated workflow
            integration_task = WorkflowTask(
                task_id="integration_task",
                name="Integration Test Task",
                task_type="integration_test",
                parameters={'test_type': 'end_to_end'},
                dependencies=[],
                max_retries=1,
                timeout_seconds=60
            )
            
            integration_workflow = WorkflowDefinition(
                workflow_id="integration_test_workflow",
                name="End-to-End Integration Test",
                description="Test complete integration",
                tasks=[integration_task],
                trigger=WorkflowTrigger.MANUAL,
                enable_parallel_execution=False
            )
            
            await workflow_engine.register_workflow(integration_workflow)
            
            # Register test worker
            await hub.register_worker("e2e_worker", ["integration_test"])
            
            # Start workflow
            execution_id = await workflow_engine.start_workflow("integration_test_workflow")
            
            # Simulate worker processing
            await asyncio.sleep(1)
            
            # Report task completion
            await hub.report_task_completion(
                "e2e_worker",
                "integration_task",
                {'integration_result': 'success', 'components_tested': 5},
                3.2
            )
            
            # Wait for all systems to process
            await asyncio.sleep(3)
            
            # Check workflow status
            workflow_status = await workflow_engine.get_workflow_status(execution_id)
            
            # Check analytics report
            analytics_report = await analytics.get_analytics_report()
            
            # Check event store metrics
            store_metrics = await event_store.get_metrics()
            
            # Check integration status
            integration_status = await integrator.get_integration_status()
            
            # Validate end-to-end flow
            success_criteria = [
                workflow_status is not None,
                analytics_report.get('analytics_performance', {}).get('events_processed', 0) > 0,
                store_metrics.get('event_sequence', 0) > 0,
                integration_status.get('integrator_status') == 'active'
            ]
            
            if all(success_criteria):
                self._add_test_result(f"{test_name}_complete_flow", True, "End-to-end integration successful")
            else:
                self._add_test_result(f"{test_name}_complete_flow", False, f"Integration criteria not met: {success_criteria}")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_performance_scalability(self):
        """Test performance and scalability of Redis Streams implementation."""
        test_name = "performance_scalability"
        self.logger.info(f"Testing {test_name}")
        
        try:
            hub = await get_intelligence_hub()
            
            # Performance test: Event throughput
            start_time = time.time()
            events_sent = 0
            
            for i in range(200):  # Send 200 events rapidly
                event = IntelligenceEvent(
                    event_id=f"perf_test_{i}",
                    event_type=EventType.PERFORMANCE_METRIC,
                    worker_id=f"perf_worker_{i % 5}",
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.NORMAL,
                    data={'value': i, 'test_type': 'throughput'}
                )
                await hub.publish_event(event)
                events_sent += 1
            
            throughput_time = time.time() - start_time
            events_per_second = events_sent / throughput_time
            
            if events_per_second > 50:  # Expect at least 50 events/second
                self._add_test_result(f"{test_name}_throughput", True, f"Throughput: {events_per_second:.1f} events/sec")
            else:
                self._add_test_result(f"{test_name}_throughput", False, f"Low throughput: {events_per_second:.1f} events/sec")
            
            # Scalability test: Multiple workers
            worker_count = 10
            for i in range(worker_count):
                await hub.register_worker(f"scale_worker_{i}", ["scale_test"])
                await hub.worker_heartbeat(f"scale_worker_{i}", {'load': i * 0.1})
            
            stats = await hub.get_hub_statistics()
            active_workers = stats.get('active_workers', 0)
            
            if active_workers >= worker_count:
                self._add_test_result(f"{test_name}_scalability", True, f"Scaled to {active_workers} workers")
            else:
                self._add_test_result(f"{test_name}_scalability", False, f"Only {active_workers}/{worker_count} workers active")
            
            # Memory usage test
            processing_stats = stats.get('processing_stats', {})
            events_processed = processing_stats.get('events_processed', 0)
            
            if events_processed > 100:
                self._add_test_result(f"{test_name}_processing", True, f"Processed {events_processed} events")
            else:
                self._add_test_result(f"{test_name}_processing", False, f"Only processed {events_processed} events")
            
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
        
        # Categorize results
        categories = {
            'intelligence_hub': [r for r in self.test_results if 'intelligence_hub' in r['test_name']],
            'event_sourcing': [r for r in self.test_results if 'event_sourcing' in r['test_name']],
            'distributed_workflows': [r for r in self.test_results if 'distributed_workflows' in r['test_name']],
            'streams_integration': [r for r in self.test_results if 'streams_integration' in r['test_name']],
            'event_analytics': [r for r in self.test_results if 'event_analytics' in r['test_name']],
            'end_to_end_integration': [r for r in self.test_results if 'end_to_end' in r['test_name']],
            'performance_scalability': [r for r in self.test_results if 'performance_scalability' in r['test_name']]
        }
        
        category_summary = {}
        for category, results in categories.items():
            if results:
                category_passed = sum(1 for r in results if r['success'])
                category_summary[category] = {
                    'total': len(results),
                    'passed': category_passed,
                    'success_rate': category_passed / len(results)
                }
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'category_summary': category_summary,
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
            recommendations.append("All tests passed! Redis Week 3 Streams implementation is ready for production.")
            recommendations.append("Consider proceeding to Week 4: Pub/Sub coordination and distributed locking.")
        else:
            recommendations.append(f"{len(failed_tests)} tests failed. Review failed tests before proceeding.")
            
            # Category-specific recommendations
            failed_categories = set()
            for test in failed_tests:
                if 'intelligence_hub' in test['test_name']:
                    failed_categories.add('intelligence_hub')
                elif 'event_sourcing' in test['test_name']:
                    failed_categories.add('event_sourcing')
                elif 'distributed_workflows' in test['test_name']:
                    failed_categories.add('workflows')
                elif 'streams_integration' in test['test_name']:
                    failed_categories.add('integration')
                elif 'event_analytics' in test['test_name']:
                    failed_categories.add('analytics')
                elif 'performance' in test['test_name']:
                    failed_categories.add('performance')
            
            if 'intelligence_hub' in failed_categories:
                recommendations.append("Intelligence Hub tests failed. Check Redis connectivity and stream configuration.")
            
            if 'event_sourcing' in failed_categories:
                recommendations.append("Event Sourcing tests failed. Verify event store configuration and Redis persistence.")
            
            if 'workflows' in failed_categories:
                recommendations.append("Workflow tests failed. Check distributed workflow engine setup and task coordination.")
            
            if 'integration' in failed_categories:
                recommendations.append("Integration tests failed. Review streams integration with legacy components.")
            
            if 'analytics' in failed_categories:
                recommendations.append("Analytics tests failed. Check event analytics engine configuration and processing.")
            
            if 'performance' in failed_categories:
                recommendations.append("Performance tests failed. Consider Redis optimization and infrastructure scaling.")
        
        return recommendations


async def main():
    """Main test runner function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Redis Week 3 Streams Test Suite")
    
    try:
        # Initialize tester
        tester = RedisWeek3StreamsTester()
        
        # Run comprehensive tests
        test_report = await tester.run_comprehensive_tests()
        
        # Print summary
        summary = test_report['summary']
        logger.info("=" * 80)
        logger.info("REDIS WEEK 3 STREAMS TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total Time: {summary['total_time_seconds']:.1f}s")
        logger.info("=" * 80)
        
        # Print category summary
        logger.info("CATEGORY BREAKDOWN:")
        for category, stats in test_report['category_summary'].items():
            logger.info(f"{category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1%})")
        logger.info("=" * 80)
        
        # Print recommendations
        logger.info("RECOMMENDATIONS:")
        for i, recommendation in enumerate(test_report['recommendations'], 1):
            logger.info(f"{i}. {recommendation}")
        
        # Save detailed report
        report_filename = f"redis_week3_streams_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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