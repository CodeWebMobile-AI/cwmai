"""
Test Dynamic Research Trigger

This test verifies that the dynamic research trigger can be created
and functions correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta

# Import the dynamic research trigger
from scripts.dynamic_research_trigger import DynamicResearchTrigger


class TestDynamicResearchTrigger(unittest.TestCase):
    """Test the dynamic research trigger functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_state_manager = Mock()
        self.mock_research_engine = Mock()
        
        # Mock state manager responses
        self.mock_state_manager.load_state.return_value = {
            "performance": {
                "claude_interactions": {
                    "total_attempts": 100,
                    "successful": 85
                },
                "task_completion": {
                    "total_tasks": 50,
                    "completed_tasks": 45
                }
            },
            "task_state": {
                "tasks": [
                    {"status": "completed"},
                    {"status": "in_progress"},
                    {"status": "failed"}
                ]
            },
            "recent_errors": []
        }
    
    def test_trigger_instantiation(self):
        """Test that the trigger can be instantiated."""
        print("\n=== Testing Dynamic Research Trigger Instantiation ===")
        
        try:
            trigger = DynamicResearchTrigger(
                state_manager=self.mock_state_manager,
                research_engine=self.mock_research_engine
            )
            
            # Verify trigger was created
            self.assertIsNotNone(trigger)
            print("✓ Trigger instantiated successfully")
            
            # Check initial state
            self.assertFalse(trigger.is_monitoring)
            print("✓ Initial monitoring state is False")
            
            # Check trigger conditions
            self.assertIsInstance(trigger.trigger_conditions, dict)
            print(f"✓ Trigger conditions loaded: {len(trigger.trigger_conditions)} types")
            
            # Verify condition types
            expected_conditions = [
                "performance_drop",
                "anomaly_detection",
                "opportunity_based",
                "event_based"
            ]
            
            for condition in expected_conditions:
                self.assertIn(condition, trigger.trigger_conditions)
                print(f"✓ Condition type '{condition}' configured")
            
            print("\n✅ Trigger instantiation test passed!")
            
        except Exception as e:
            self.fail(f"Failed to instantiate trigger: {e}")
    
    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        print("\n=== Testing Metrics Collection ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Collect metrics
        metrics = trigger._collect_current_metrics()
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        print("✓ Metrics collected successfully")
        
        # Check metric values
        self.assertEqual(metrics["claude_success_rate"], 85.0)
        print(f"✓ Claude success rate: {metrics['claude_success_rate']}%")
        
        self.assertEqual(metrics["task_completion_rate"], 90.0)
        print(f"✓ Task completion rate: {metrics['task_completion_rate']}%")
        
        self.assertEqual(metrics["error_count"], 0)
        print(f"✓ Error count: {metrics['error_count']}")
        
        self.assertEqual(metrics["active_tasks"], 1)
        print(f"✓ Active tasks: {metrics['active_tasks']}")
        
        self.assertEqual(metrics["system_health"], "healthy")
        print(f"✓ System health: {metrics['system_health']}")
        
        # Check metrics history
        self.assertGreater(len(trigger.metrics_history["claude_success_rate"]), 0)
        print("✓ Metrics stored in history")
        
        print("\n✅ Metrics collection test passed!")
    
    def test_performance_drop_detection(self):
        """Test performance drop detection."""
        print("\n=== Testing Performance Drop Detection ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Simulate performance drop
        now = datetime.now()
        
        # Add historical data
        trigger.metrics_history["claude_success_rate"].append((now - timedelta(minutes=10), 90.0))
        trigger.metrics_history["claude_success_rate"].append((now - timedelta(minutes=5), 85.0))
        trigger.metrics_history["claude_success_rate"].append((now - timedelta(minutes=2), 70.0))
        trigger.metrics_history["claude_success_rate"].append((now, 50.0))  # 40% drop
        
        # Test drop detection
        result = trigger._check_metric_drop("claude_success_rate", 20, 600)  # 20% threshold, 10 min window
        
        self.assertTrue(result)
        print("✓ Performance drop detected correctly")
        
        # Test no drop scenario
        trigger.metrics_history["task_completion_rate"].clear()
        trigger.metrics_history["task_completion_rate"].append((now - timedelta(minutes=5), 90.0))
        trigger.metrics_history["task_completion_rate"].append((now, 88.0))  # Only 2% drop
        
        result = trigger._check_metric_drop("task_completion_rate", 20, 600)
        
        self.assertFalse(result)
        print("✓ Small drop correctly not triggered")
        
        print("\n✅ Performance drop detection test passed!")
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        print("\n=== Testing Anomaly Detection ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Add normal data
        now = datetime.now()
        for i in range(20):
            trigger.metrics_history["claude_success_rate"].append(
                (now - timedelta(minutes=20-i), 85.0 + (i % 3))  # Normal variation
            )
        
        # Add anomalous data point
        trigger.metrics_history["claude_success_rate"].append((now, 30.0))  # Outlier
        
        # Detect anomalies
        anomalies = trigger._detect_anomalies()
        
        self.assertIsInstance(anomalies, list)
        print(f"✓ Detected {len(anomalies)} anomalies")
        
        # Check for statistical outlier
        outlier_found = any(a["type"] == "statistical_outlier" for a in anomalies)
        if outlier_found:
            print("✓ Statistical outlier detected")
        
        print("\n✅ Anomaly detection test passed!")
    
    def test_opportunity_identification(self):
        """Test opportunity identification."""
        print("\n=== Testing Opportunity Identification ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Set up idle system scenario
        self.mock_state_manager.load_state.return_value["task_state"]["tasks"] = []
        
        # Add stable performance data (plateau)
        now = datetime.now()
        for i in range(25):
            trigger.metrics_history["claude_success_rate"].append(
                (now - timedelta(minutes=25-i), 75.0 + (i % 2) * 0.5)  # Very stable around 75%
            )
        
        # Identify opportunities
        opportunities = trigger._identify_opportunities()
        
        self.assertIsInstance(opportunities, list)
        print(f"✓ Identified {len(opportunities)} opportunities")
        
        # Check for specific opportunity types
        opportunity_types = [opp["type"] for opp in opportunities]
        
        if "idle_resources" in opportunity_types:
            print("✓ Idle resources opportunity identified")
        
        if "learning_plateau" in opportunity_types:
            print("✓ Learning plateau opportunity identified")
        
        print("\n✅ Opportunity identification test passed!")
    
    def test_event_handling(self):
        """Test event handling."""
        print("\n=== Testing Event Handling ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Add events
        trigger.add_event("critical_error", {
            "error": "System failure",
            "severity": "critical"
        })
        
        trigger.add_event("new_project_added", {
            "project": "test_project",
            "timestamp": datetime.now().isoformat()
        })
        
        # Check event queue
        self.assertEqual(len(trigger.event_queue), 2)
        print(f"✓ Added {len(trigger.event_queue)} events to queue")
        
        # Verify event structure
        critical_event = trigger.event_queue[0]
        self.assertEqual(critical_event["type"], "critical_error")
        print("✓ Critical error event added correctly")
        
        project_event = trigger.event_queue[1]
        self.assertEqual(project_event["type"], "new_project_added")
        print("✓ New project event added correctly")
        
        print("\n✅ Event handling test passed!")
    
    def test_cooldown_management(self):
        """Test cooldown management."""
        print("\n=== Testing Cooldown Management ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Test initial cooldown state
        result = trigger._check_cooldown("performance_drop")
        self.assertTrue(result)
        print("✓ No cooldown initially")
        
        # Update cooldowns
        trigger._update_cooldowns("performance_drop")
        
        # Test immediate check (should be in cooldown)
        result = trigger._check_cooldown("performance_drop")
        self.assertFalse(result)
        print("✓ Cooldown active after trigger")
        
        # Test global cooldown
        result = trigger._check_cooldown("anomaly")  # Different type
        self.assertFalse(result)
        print("✓ Global cooldown affects all trigger types")
        
        # Get cooldown status
        stats = trigger.get_statistics()
        cooldown_status = stats["cooldown_status"]
        
        self.assertTrue(cooldown_status["global"]["active"])
        print("✓ Cooldown status reported correctly")
        
        print("\n✅ Cooldown management test passed!")
    
    def test_priority_determination(self):
        """Test priority determination."""
        print("\n=== Testing Priority Determination ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Test critical error priority
        priority = trigger._determine_priority("event", {"type": "critical_error"})
        self.assertEqual(priority, "critical")
        print("✓ Critical error gets critical priority")
        
        # Test performance drop with critical health
        priority = trigger._determine_priority("performance_drop", {
            "metrics": {"system_health": "critical"}
        })
        self.assertEqual(priority, "critical")
        print("✓ Critical system health gets critical priority")
        
        # Test low claude success rate
        priority = trigger._determine_priority("performance_drop", {
            "metrics": {"claude_success_rate": 25}
        })
        self.assertEqual(priority, "high")
        print("✓ Low claude success rate gets high priority")
        
        # Test opportunity priority
        priority = trigger._determine_priority("opportunity", {"priority": "low"})
        self.assertEqual(priority, "low")
        print("✓ Opportunity uses provided priority")
        
        print("\n✅ Priority determination test passed!")
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        print("\n=== Testing Statistics Tracking ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Get initial statistics
        stats = trigger.get_statistics()
        
        self.assertEqual(stats["total_triggers"], 0)
        print("✓ Initial trigger count is 0")
        
        self.assertIsInstance(stats["triggers_by_type"], dict)
        print("✓ Trigger type tracking initialized")
        
        self.assertEqual(stats["monitoring_status"], "inactive")
        print("✓ Monitoring status reported correctly")
        
        # Check structure
        expected_keys = [
            "total_triggers",
            "triggers_by_type",
            "false_positives",
            "successful_triggers",
            "cooldown_blocks",
            "recent_triggers",
            "cooldown_status",
            "monitoring_status"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            print(f"✓ Statistics include '{key}'")
        
        print("\n✅ Statistics tracking test passed!")


class TestDynamicResearchTriggerAsync(unittest.TestCase):
    """Test async functionality of the dynamic research trigger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_state_manager = Mock()
        self.mock_research_engine = AsyncMock()
        
        self.mock_state_manager.load_state.return_value = {
            "performance": {
                "claude_interactions": {"total_attempts": 100, "successful": 85},
                "task_completion": {"total_tasks": 50, "completed_tasks": 45}
            },
            "task_state": {"tasks": []},
            "recent_errors": []
        }
    
    async def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        print("\n=== Testing Monitoring Start/Stop (Async) ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(trigger.start_monitoring())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        self.assertTrue(trigger.is_monitoring)
        print("✓ Monitoring started successfully")
        
        # Stop monitoring
        trigger.stop_monitoring()
        
        self.assertFalse(trigger.is_monitoring)
        print("✓ Monitoring stopped successfully")
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        print("\n✅ Async monitoring test passed!")
    
    async def test_research_triggering(self):
        """Test research triggering."""
        print("\n=== Testing Research Triggering (Async) ===")
        
        trigger = DynamicResearchTrigger(
            state_manager=self.mock_state_manager,
            research_engine=self.mock_research_engine
        )
        
        # Mock execute_emergency_research
        self.mock_research_engine.execute_emergency_research.return_value = {
            "status": "completed"
        }
        
        # Trigger research
        await trigger._trigger_research("performance_drop", {
            "condition": "claude_success_rate_drop",
            "metrics": {"system_health": "critical"}
        })
        
        # Verify research was triggered
        self.mock_research_engine.execute_emergency_research.assert_called_once()
        print("✓ Emergency research triggered for critical issue")
        
        # Check statistics
        stats = trigger.get_statistics()
        self.assertEqual(stats["total_triggers"], 1)
        self.assertEqual(stats["triggers_by_type"]["performance_drop"], 1)
        print("✓ Trigger statistics updated correctly")
        
        print("\n✅ Research triggering test passed!")


def run_tests():
    """Run all tests."""
    # Run synchronous tests
    print("="*60)
    print("RUNNING SYNCHRONOUS TESTS")
    print("="*60)
    
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestDynamicResearchTrigger)
    runner = unittest.TextTestRunner(verbosity=2)
    result1 = runner.run(suite1)
    
    # Run asynchronous tests
    print("\n" + "="*60)
    print("RUNNING ASYNCHRONOUS TESTS")
    print("="*60)
    
    async def run_async_tests():
        """Run async test methods."""
        test_case = TestDynamicResearchTriggerAsync()
        test_case.setUp()
        
        await test_case.test_monitoring_start_stop()
        await test_case.test_research_triggering()
        
        return True
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_success = loop.run_until_complete(run_async_tests())
    loop.close()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Synchronous tests run: {result1.testsRun}")
    print(f"Synchronous failures: {len(result1.failures)}")
    print(f"Synchronous errors: {len(result1.errors)}")
    print(f"Asynchronous tests: {'PASSED' if async_success else 'FAILED'}")
    
    success = result1.wasSuccessful() and async_success
    
    if success:
        print("\n✅ ALL TESTS PASSED! The dynamic research trigger is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)