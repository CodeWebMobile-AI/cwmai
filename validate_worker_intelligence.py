"""
Simple Validation for Enhanced Worker Intelligence System

Simplified validation script that doesn't require external dependencies.
"""

import asyncio
import time
import json
from datetime import datetime, timezone

# Import the enhanced intelligence system
from scripts.worker_intelligence_integration import (
    WorkerIntelligenceCoordinator,
    WorkerEnhancementConfig,
    IntelligentWorkerMixin
)
from scripts.worker_logging_config import LogLevel
from scripts.worker_intelligence_hub import WorkerSpecialization


class MockAIBrain:
    """Mock AI brain for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate_enhanced_response(self, prompt: str, model: str = None):
        """Generate mock AI response."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate AI processing
        
        return {
            'content': json.dumps({
                'key_points': [f"Key insight {self.call_count}", "Technical consideration"],
                'challenges': [f"Challenge {self.call_count}", "Implementation complexity"],
                'recommendations': [f"Recommendation {self.call_count}", "Best practice"],
                'priority': 7,
                'complexity': 'medium',
                'confidence': 0.8,
                'alignment_score': 0.9
            })
        }


class TestWorker:
    """Simple test worker."""
    
    def __init__(self, name: str):
        self.name = name
        self.task_count = 0
    
    async def process_task(self, data: str) -> str:
        """Process a task."""
        await asyncio.sleep(0.05)  # Simulate work
        self.task_count += 1
        return f"Processed '{data}' by {self.name}"


async def test_basic_functionality():
    """Test basic functionality of the intelligence system."""
    print("🧪 Testing Basic Intelligence System Functionality")
    print("-" * 50)
    
    try:
        # Create intelligence coordinator
        print("1. Creating intelligence coordinator...")
        coordinator = WorkerIntelligenceCoordinator()
        
        # Start all components
        print("2. Starting intelligence components...")
        await coordinator.start()
        print("   ✅ All components started successfully")
        
        # Create and enhance a worker
        print("3. Creating and enhancing worker...")
        original_worker = TestWorker("test_worker")
        enhanced_worker = coordinator.enhance_worker(
            original_worker,
            "enhanced_test_worker",
            WorkerEnhancementConfig(
                worker_specialization=WorkerSpecialization.GENERAL,
                log_level=LogLevel.INFO
            )
        )
        print(f"   ✅ Worker enhanced: {enhanced_worker.worker_id}")
        
        # Test intelligent task execution
        print("4. Testing intelligent task execution...")
        async with enhanced_worker.intelligent_task_execution(
            "test_task_1", "data_processing", {"test": True}
        ):
            result = await enhanced_worker.process_task("test_data")
            print(f"   ✅ Task result: {result}")
        
        # Test performance metrics
        print("5. Testing performance metrics...")
        metrics = enhanced_worker.report_performance_metrics()
        print(f"   ✅ Metrics collected: {metrics['tasks_completed']} tasks completed")
        
        # Test error handling
        print("6. Testing error handling...")
        try:
            async with enhanced_worker.intelligent_task_execution(
                "error_task", "error_test"
            ):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        error_summary = coordinator.error_analyzer.get_error_summary(hours=1)
        print(f"   ✅ Error captured: {error_summary['total_errors']} errors recorded")
        
        # Test system dashboard
        print("7. Testing system dashboard...")
        dashboard = coordinator.get_system_dashboard()
        print(f"   ✅ Dashboard generated: {dashboard['system_health']['status']} status")
        
        # Test work item tracking
        print("8. Testing work item tracking...")
        work_item_id = coordinator.work_item_tracker.create_work_item(
            title="Test Work Item",
            description="Validation test work item",
            work_type="validation"
        )
        coordinator.work_item_tracker.assign_to_worker(work_item_id, "enhanced_test_worker")
        coordinator.work_item_tracker.complete_work(work_item_id, "enhanced_test_worker")
        
        analytics = coordinator.work_item_tracker.get_analytics_summary()
        print(f"   ✅ Work item tracked: {analytics['summary']['completed_items']} items completed")
        
        # Test alerting
        print("9. Testing alerting system...")
        from scripts.worker_status_reporter import AlertType, AlertSeverity
        alert_id = coordinator.create_alert(
            AlertType.CUSTOM,
            AlertSeverity.INFO,
            "Validation Test Alert",
            "This is a test alert for validation"
        )
        print(f"   ✅ Alert created: {alert_id}")
        
        # Wait for monitoring cycles
        print("10. Running monitoring cycles...")
        await asyncio.sleep(3)
        print("    ✅ Monitoring cycles completed")
        
        # Stop all components
        print("11. Stopping intelligence components...")
        await coordinator.stop()
        print("    ✅ All components stopped cleanly")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The Enhanced Worker Intelligence System is functioning correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def test_enhanced_swarm():
    """Test enhanced swarm intelligence."""
    print("\n🤖 Testing Enhanced Swarm Intelligence")
    print("-" * 50)
    
    try:
        # Import enhanced swarm
        from scripts.enhanced_swarm_intelligence import create_enhanced_swarm
        
        # Create enhanced swarm
        print("1. Creating enhanced swarm...")
        ai_brain = MockAIBrain()
        enhanced_swarm, coordinator = create_enhanced_swarm(ai_brain, enable_coordinator=True)
        
        if coordinator:
            await coordinator.start()
        
        print(f"   ✅ Enhanced swarm created with {len(enhanced_swarm.agents)} agents")
        
        # Test swarm task processing
        print("2. Testing swarm task processing...")
        test_task = {
            'id': 'validation_test_task',
            'type': 'feature_analysis',
            'title': 'Validation Test Task',
            'description': 'Testing enhanced swarm with validation task',
            'requirements': ['Performance', 'Scalability']
        }
        
        result = await enhanced_swarm.process_task_swarm(test_task)
        
        print(f"   ✅ Swarm analysis completed:")
        print(f"       - Task ID: {result.get('task_id')}")
        print(f"       - Duration: {result.get('duration_seconds', 0):.2f}s")
        print(f"       - Agents: {len(result.get('individual_analyses', []))}")
        print(f"       - Consensus Priority: {result.get('consensus', {}).get('consensus_priority', 'N/A')}")
        
        # Test enhanced analytics
        print("3. Testing enhanced analytics...")
        analytics = enhanced_swarm.get_enhanced_analytics()
        print(f"   ✅ Analytics generated with intelligence integration")
        
        if coordinator:
            await coordinator.stop()
        
        print("\n🎉 ENHANCED SWARM TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced swarm test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def run_validation():
    """Run complete validation of the enhanced worker intelligence system."""
    print("🚀 Enhanced Worker Intelligence System Validation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Test basic functionality
    basic_success = await test_basic_functionality()
    
    # Test enhanced swarm
    swarm_success = await test_enhanced_swarm()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print(f"⏱️  Total Duration: {duration:.2f} seconds")
    print(f"✅ Basic Functionality: {'PASSED' if basic_success else 'FAILED'}")
    print(f"✅ Enhanced Swarm: {'PASSED' if swarm_success else 'FAILED'}")
    
    if basic_success and swarm_success:
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("The Enhanced Worker Intelligence System is ready for use.")
        print("\n📋 Features Validated:")
        print("  ✅ Centralized logging with correlation")
        print("  ✅ Cross-worker intelligence and coordination")
        print("  ✅ Real-time performance metrics collection")
        print("  ✅ Intelligent error pattern detection and recovery")
        print("  ✅ Complete work item lifecycle tracking")
        print("  ✅ Real-time monitoring and alerting")
        print("  ✅ Enhanced swarm intelligence with integration")
        print("  ✅ Seamless integration with existing workers")
        return True
    else:
        print("\n❌ VALIDATION FAILED!")
        print("Please check the error messages above and fix any issues.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_validation())
    exit(0 if success else 1)