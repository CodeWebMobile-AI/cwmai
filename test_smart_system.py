"""
Test Suite for Smart AI Worker System
"""
import asyncio
import pytest
import sys
import os
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from ai_worker_agent import AIWorkerAgent, WorkerCapability, TaskOutcome
from ai_task_marketplace import AITaskMarketplace, TaskBid
from worker_capability_store import WorkerCapabilityStore
from smart_orchestrator import SmartOrchestrator
from smart_redis_integration import SmartRedisIntegration
from work_item_types import WorkItem, TaskPriority


class TestAIWorkerAgent:
    """Test AI Worker Agent functionality"""
    
    @pytest.mark.asyncio
    async def test_worker_creation(self):
        """Test worker creation and initialization"""
        worker = AIWorkerAgent("test_worker_1", "system_tasks")
        
        assert worker.worker_id == "test_worker_1"
        assert worker.primary_specialization == "system_tasks"
        assert worker.current_load == 0.0
        assert worker.total_tasks_completed == 0
    
    @pytest.mark.asyncio
    async def test_task_evaluation(self):
        """Test task evaluation logic"""
        worker = AIWorkerAgent("test_worker_2", "ai-creative-studio")
        
        # Test matching repository
        work_item = WorkItem(
            id="task_1",
            task_type="UPDATE_FEATURE",
            title="Update UI components",
            description="Update the UI components for better UX",
            priority=TaskPriority.HIGH,
            repository="ai-creative-studio"
        )
        
        confidence, time, reasoning = await worker.evaluate_task(work_item)
        
        assert confidence > 0.4  # Should have decent confidence for matching repo
        assert "Primary specialization match" in reasoning
    
    @pytest.mark.asyncio
    async def test_task_bidding(self):
        """Test task bidding process"""
        worker = AIWorkerAgent("test_worker_3", None)
        worker.confidence_threshold = 0.1  # Very low threshold for testing
        worker.exploration_rate = 0.9  # High exploration to ensure bidding
        
        work_item = WorkItem(
            id="task_2",
            task_type="REFACTOR_CODE",
            title="Refactor authentication module",
            description="Refactor the authentication module for better performance",
            priority=TaskPriority.MEDIUM
        )
        
        bid = await worker.submit_bid(work_item)
        
        assert bid is not None
        assert bid["worker_id"] == "test_worker_3"
        assert bid["work_item_id"] == "task_2"
        assert "confidence" in bid
        assert "estimated_time" in bid
        assert "reasoning" in bid
    
    @pytest.mark.asyncio
    async def test_capability_learning(self):
        """Test capability learning from task outcomes"""
        worker = AIWorkerAgent("test_worker_4", "system_tasks")
        
        work_item = WorkItem(
            id="task_3",
            task_type="OPTIMIZE_PERFORMANCE",
            title="Optimize database queries",
            description="Optimize database queries for faster response",
            priority=TaskPriority.HIGH
        )
        
        # Start task
        await worker.start_task(work_item)
        assert len(worker.active_tasks) == 1
        
        # Complete task successfully
        await worker.complete_task(work_item, True, 2.5, 0.9)
        
        # Check capability was updated
        cap_key = "OPTIMIZE_PERFORMANCE:none"
        assert cap_key in worker.capabilities
        assert worker.capabilities[cap_key].total_tasks == 1
        assert worker.capabilities[cap_key].success_rate > 0
        assert worker.total_tasks_completed == 1


class TestAITaskMarketplace:
    """Test AI Task Marketplace functionality"""
    
    @pytest.mark.asyncio
    async def test_marketplace_creation(self):
        """Test marketplace creation"""
        marketplace = AITaskMarketplace()
        
        assert len(marketplace.workers) == 0
        assert len(marketplace.active_auctions) == 0
    
    @pytest.mark.asyncio
    async def test_worker_registration(self):
        """Test worker registration"""
        marketplace = AITaskMarketplace()
        worker = AIWorkerAgent("test_worker_5", "system_tasks")
        
        await marketplace.register_worker(worker)
        
        assert "test_worker_5" in marketplace.workers
        assert marketplace.workers["test_worker_5"] == worker
    
    @pytest.mark.asyncio
    async def test_task_auction(self):
        """Test task auction process"""
        marketplace = AITaskMarketplace()
        
        # Register multiple workers
        worker1 = AIWorkerAgent("worker_1", "system_tasks")
        worker2 = AIWorkerAgent("worker_2", "ai-creative-studio")
        worker3 = AIWorkerAgent("worker_3", None)
        
        for worker in [worker1, worker2, worker3]:
            await marketplace.register_worker(worker)
        
        # Submit task for auction
        work_item = WorkItem(
            id="auction_task_1",
            task_type="NEW_FEATURE",
            title="Implement user dashboard",
            description="Create a new user dashboard with analytics",
            priority=TaskPriority.HIGH
        )
        
        winner_id = await marketplace.submit_task(work_item)
        
        assert winner_id is not None
        assert winner_id in ["worker_1", "worker_2", "worker_3"]
    
    @pytest.mark.asyncio
    async def test_bid_scoring(self):
        """Test bid scoring mechanism"""
        bid = TaskBid(
            worker_id="test_worker",
            work_item_id="test_task",
            confidence=0.8,
            estimated_time=2.0,
            reasoning="Test reasoning",
            current_load=0.3,
            total_experience=50,
            timestamp=datetime.now(timezone.utc)
        )
        
        weights = {
            "confidence": 0.4,
            "speed": 0.2,
            "availability": 0.2,
            "experience": 0.2
        }
        
        score = bid.score(weights)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be a decent score


class TestWorkerCapabilityStore:
    """Test Worker Capability Store functionality"""
    
    @pytest.mark.asyncio
    async def test_capability_storage(self):
        """Test capability storage and retrieval"""
        store = WorkerCapabilityStore()
        
        capabilities = {
            "overall_success_rate": 0.85,
            "total_completed": 25,
            "capabilities": {
                "REFACTOR_CODE:system": {
                    "success_rate": 0.9,
                    "avg_duration": 1.5,
                    "total_tasks": 10
                }
            }
        }
        
        await store.save_worker_capabilities("test_worker_6", capabilities)
        
        # Load capabilities
        loaded = await store.load_worker_capabilities("test_worker_6")
        
        assert loaded is not None
        assert loaded["overall_success_rate"] == 0.85
        assert loaded["total_completed"] == 25
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing(self):
        """Test knowledge sharing functionality"""
        store = WorkerCapabilityStore()
        
        knowledge_data = {
            "pattern": "Complex refactoring tasks benefit from breaking into smaller pieces",
            "confidence": 0.9,
            "source": "test_worker_7"
        }
        
        await store.share_knowledge("refactoring_tip_1", knowledge_data)
        
        # Retrieve knowledge
        retrieved = await store.get_shared_knowledge("refactoring_tip_1")
        
        assert retrieved is not None
        assert retrieved["data"]["pattern"] == knowledge_data["pattern"]
    
    @pytest.mark.asyncio
    async def test_expert_finding(self):
        """Test finding expert workers"""
        store = WorkerCapabilityStore()
        
        # Add test data
        expert_capabilities = {
            "worker_id": "expert_worker",
            "capabilities": {
                "OPTIMIZE_PERFORMANCE:database": {
                    "success_rate": 0.95,
                    "avg_duration": 1.2,
                    "total_tasks": 20
                }
            }
        }
        
        store.capability_cache["expert_worker"] = expert_capabilities
        
        experts = await store.find_experts("OPTIMIZE_PERFORMANCE", min_success_rate=0.9)
        
        assert "expert_worker" in experts


class TestSmartOrchestrator:
    """Test Smart Orchestrator functionality"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = SmartOrchestrator(num_workers=3)
        
        await orchestrator.initialize(["system_tasks", "ai-creative-studio", None])
        
        assert len(orchestrator.workers) == 3
        assert orchestrator.is_running == True
        
        # Clean up
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_work_item_processing(self):
        """Test work item processing"""
        orchestrator = SmartOrchestrator(num_workers=2)
        await orchestrator.initialize(["system_tasks", None])
        
        work_item = WorkItem(
            id="orchestrator_task_1",
            task_type="UPDATE_DEPENDENCIES",
            title="Update project dependencies",
            description="Update all npm dependencies to latest versions",
            priority=TaskPriority.MEDIUM
        )
        
        winner_id = await orchestrator.process_work_item(work_item)
        
        assert winner_id is not None
        assert winner_id in orchestrator.workers
        
        # Clean up
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance tracking"""
        orchestrator = SmartOrchestrator(num_workers=2)
        await orchestrator.initialize()
        
        work_item = WorkItem(
            id="perf_task_1",
            task_type="FIX_BUG",
            title="Fix memory leak",
            description="Fix memory leak in data processing module",
            priority=TaskPriority.HIGH
        )
        
        # Process and complete task
        winner_id = await orchestrator.process_work_item(work_item)
        if winner_id:
            await orchestrator.report_task_completion(
                work_item, winner_id, True, 1.5, 0.95
            )
        
        # Check performance window
        assert len(orchestrator.performance_window) == 1
        assert orchestrator.performance_window[0]["success"] == True
        
        # Get stats
        stats = orchestrator.get_system_stats()
        assert stats["orchestrator"]["num_workers"] == 2
        assert stats["recent_performance"]["success_rate"] == 1.0
        
        # Clean up
        await orchestrator.shutdown()


class TestSmartRedisIntegration:
    """Test Smart Redis Integration"""
    
    @pytest.mark.asyncio
    async def test_integration_creation(self):
        """Test integration creation"""
        integration = SmartRedisIntegration(None, num_workers=3)
        
        assert integration.tasks_processed == 0
        assert integration.tasks_succeeded == 0
        assert not integration.is_running
    
    @pytest.mark.asyncio
    async def test_integration_stats(self):
        """Test integration statistics"""
        integration = SmartRedisIntegration(None, num_workers=2)
        
        stats = integration.get_integration_stats()
        
        assert "integration" in stats
        assert "orchestrator" in stats
        assert stats["integration"]["tasks_processed"] == 0
        assert stats["integration"]["success_rate"] == 0


async def run_integration_test():
    """Run a full integration test"""
    print("\n=== Running Full Integration Test ===\n")
    
    # Create smart system
    integration = SmartRedisIntegration(None, num_workers=3)
    
    # Start system
    await integration.start(["system_tasks", "ai-creative-studio", None])
    
    print("✓ System started with 3 workers")
    
    # Create test work items
    test_items = [
        WorkItem(
            id=f"test_item_{i}",
            task_type="SYSTEM_IMPROVEMENT" if i % 2 == 0 else "NEW_PROJECT",
            title=f"Test task {i}",
            description=f"This is test task number {i}",
            priority=TaskPriority.HIGH if i < 3 else TaskPriority.MEDIUM,
            repository="ai-creative-studio" if i % 3 == 0 else None
        )
        for i in range(5)
    ]
    
    print(f"✓ Created {len(test_items)} test work items")
    
    # Process work items
    for item in test_items:
        await integration._process_work_item(item)
    
    # Wait for processing
    await asyncio.sleep(10)
    
    # Get final stats
    stats = integration.get_integration_stats()
    
    print("\n=== Final Statistics ===")
    print(f"Tasks processed: {stats['integration']['tasks_processed']}")
    print(f"Tasks succeeded: {stats['integration']['tasks_succeeded']}")
    print(f"Success rate: {stats['integration']['success_rate']:.2%}")
    if 'marketplace' in stats['orchestrator']:
        print(f"Active workers: {stats['orchestrator']['marketplace']['active_workers']}")
        print(f"Total auctions: {stats['orchestrator']['marketplace']['total_tasks_processed']}")
    
    # Print worker performance
    print("\n=== Worker Performance ===")
    if 'marketplace' in stats['orchestrator'] and 'worker_performance' in stats['orchestrator']['marketplace']:
        for worker_id, perf in stats['orchestrator']['marketplace']['worker_performance'].items():
            print(f"\n{worker_id}:")
            print(f"  - Specialization: {perf['specialization']}")
            print(f"  - Tasks completed: {perf['total_completed']}")
            print(f"  - Success rate: {perf['overall_success_rate']:.2%}")
            print(f"  - Current load: {perf['current_load']:.2%}")
    
    # Stop system
    await integration.stop()
    
    print("\n✓ System stopped successfully")
    
    # Verify results
    assert stats['integration']['tasks_processed'] > 0
    assert stats['integration']['success_rate'] > 0.5
    print("\n✓ All integration tests passed!")


if __name__ == "__main__":
    print("Starting Smart AI Worker System Tests...\n")
    
    # Run unit tests
    print("Running unit tests...")
    pytest.main([__file__, "-v", "-k", "not integration"])
    
    # Run integration test
    print("\nRunning integration test...")
    asyncio.run(run_integration_test())