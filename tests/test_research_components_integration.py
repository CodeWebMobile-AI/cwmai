"""
Test Research Components Integration

This test verifies that all research components integrate properly
and can work together.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime
import json

# Import all research components
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.research_need_analyzer import ResearchNeedAnalyzer
from scripts.intelligent_research_selector import IntelligentResearchSelector
from scripts.research_scheduler import ResearchScheduler, ResearchPriority
from scripts.research_query_generator import ResearchQueryGenerator
from scripts.research_action_engine import ResearchActionEngine
from scripts.research_learning_system import ResearchLearningSystem
from scripts.knowledge_graph_builder import KnowledgeGraphBuilder
from scripts.research_insight_processor import ResearchInsightProcessor
from scripts.dynamic_research_trigger import DynamicResearchTrigger
from scripts.cross_research_analyzer import CrossResearchAnalyzer


class TestResearchIntegration(unittest.TestCase):
    """Test integration of research components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_state_manager = Mock()
        self.mock_ai_brain = AsyncMock()
        self.mock_task_generator = Mock()
        self.mock_self_improver = Mock()
        self.mock_outcome_learning = Mock()
        
        # Mock state
        self.mock_state = {
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
                    {"id": "1", "status": "completed", "created_at": "2024-01-01"},
                    {"id": "2", "status": "failed", "created_at": "2024-01-02"},
                    {"id": "3", "status": "in_progress", "created_at": "2024-01-03"}
                ]
            },
            "recent_errors": [],
            "projects": {"project1": {}, "project2": {}},
            "metrics": {
                "claude_interactions": {"success_rate": 0.85}
            }
        }
        
        self.mock_state_manager.load_state.return_value = self.mock_state
        
        # Mock AI brain response
        self.mock_ai_brain.generate_enhanced_response.return_value = {
            "content": "Research insight: Implement caching to improve performance by 30%"
        }
    
    def test_knowledge_store_integration(self):
        """Test knowledge store integration."""
        print("\n=== Testing Knowledge Store Integration ===")
        
        # Create knowledge store
        store = ResearchKnowledgeStore()
        
        # Store research
        research_data = {
            "topic": "Performance optimization",
            "content": "Cache frequently accessed data",
            "timestamp": datetime.now().isoformat()
        }
        
        research_id = store.store_research("performance", research_data, quality_score=0.8)
        self.assertIsNotNone(research_id)
        print(f"✓ Research stored with ID: {research_id}")
        
        # Retrieve research
        retrieved = store.get_research_by_area("performance", limit=1)
        self.assertEqual(len(retrieved), 1)
        print("✓ Research retrieved successfully")
        
        # Get statistics
        stats = store.get_statistics()
        self.assertEqual(stats["total_research"], 1)
        self.assertEqual(stats["research_by_area"]["performance"], 1)
        print(f"✓ Statistics tracked: {stats}")
        
        print("\n✅ Knowledge store integration test passed!")
    
    def test_need_analyzer_integration(self):
        """Test need analyzer integration."""
        print("\n=== Testing Need Analyzer Integration ===")
        
        # Create need analyzer
        analyzer = ResearchNeedAnalyzer(self.mock_state_manager)
        
        # Analyze performance gaps
        gaps = analyzer.analyze_performance_gaps()
        
        self.assertIsInstance(gaps, dict)
        print("✓ Performance gaps analyzed")
        
        # Get immediate needs
        needs = analyzer.get_immediate_research_needs()
        self.assertIsInstance(needs, list)
        print(f"✓ Identified {len(needs)} immediate research needs")
        
        # Get proactive opportunities
        opportunities = analyzer.get_proactive_research_opportunities()
        self.assertIsInstance(opportunities, list)
        print(f"✓ Identified {len(opportunities)} proactive opportunities")
        
        # Test learning trigger
        events = [
            {"type": "task", "status": "failed"},
            {"type": "task", "status": "failed"},
            {"type": "error", "message": "API timeout"}
        ]
        should_trigger = analyzer.should_trigger_learning_research(events)
        self.assertIsInstance(should_trigger, bool)
        print(f"✓ Learning trigger evaluated: {should_trigger}")
        
        print("\n✅ Need analyzer integration test passed!")
    
    def test_research_selector_integration(self):
        """Test research selector integration."""
        print("\n=== Testing Research Selector Integration ===")
        
        # Create components
        store = ResearchKnowledgeStore()
        analyzer = ResearchNeedAnalyzer(self.mock_state_manager)
        selector = IntelligentResearchSelector(store, analyzer)
        
        # Select research topics
        context = {
            "system_health": "degraded",
            "metrics": {"claude_success_rate": 0.65},
            "performance_gaps": {"high": ["efficiency"]}
        }
        
        topics = selector.select_research_topics(context)
        
        self.assertIsInstance(topics, list)
        print(f"✓ Selected {len(topics)} research topics")
        
        if topics:
            print("✓ Sample topic structure:", json.dumps(topics[0], indent=2))
        
        # Update effectiveness
        selector.update_topic_effectiveness("efficiency", 0.8)
        print("✓ Topic effectiveness updated")
        
        print("\n✅ Research selector integration test passed!")
    
    def test_scheduler_integration(self):
        """Test scheduler integration."""
        print("\n=== Testing Scheduler Integration ===")
        
        # Create components
        store = ResearchKnowledgeStore()
        scheduler = ResearchScheduler(self.mock_state_manager, store)
        
        # Schedule research
        request = {
            "area": "performance",
            "priority": ResearchPriority.HIGH,
            "description": "Optimize task execution"
        }
        
        request_id = scheduler.schedule_research(request)
        self.assertIsNotNone(request_id)
        print(f"✓ Research scheduled with ID: {request_id}")
        
        # Get next research
        next_research = scheduler.get_next_research()
        self.assertIsNotNone(next_research)
        print("✓ Next research retrieved")
        
        # Get schedule status
        status = scheduler.get_schedule_status()
        self.assertEqual(status["total_scheduled"], 1)
        self.assertEqual(status["pending"], 1)
        print(f"✓ Schedule status: {status}")
        
        print("\n✅ Scheduler integration test passed!")
    
    def test_query_generator_integration(self):
        """Test query generator integration."""
        print("\n=== Testing Query Generator Integration ===")
        
        # Create query generator
        generator = ResearchQueryGenerator()
        
        # Generate queries
        topic = {
            "topic": "error_handling",
            "area": "reliability",
            "priority": "high",
            "context": {"error_types": ["timeout", "connection"]}
        }
        
        queries = generator.generate_queries(topic)
        
        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)
        print(f"✓ Generated {len(queries)} queries")
        
        if queries:
            print("✓ Sample query:", queries[0]["query"][:100] + "...")
        
        print("\n✅ Query generator integration test passed!")
    
    def test_action_engine_integration(self):
        """Test action engine integration."""
        print("\n=== Testing Action Engine Integration ===")
        
        # Create action engine
        engine = ResearchActionEngine(self.mock_task_generator, self.mock_self_improver)
        
        # Extract insights
        research = {
            "content": {
                "content": "Implement caching to reduce API calls by 40%. Use Redis for distributed cache."
            },
            "area": "performance"
        }
        
        insights = engine.extract_actionable_insights(research)
        
        self.assertIsInstance(insights, list)
        print(f"✓ Extracted {len(insights)} insights")
        
        # Generate tasks
        if insights:
            tasks = engine.generate_implementation_tasks(insights)
            self.assertIsInstance(tasks, list)
            print(f"✓ Generated {len(tasks)} implementation tasks")
        
        print("\n✅ Action engine integration test passed!")
    
    def test_learning_system_integration(self):
        """Test learning system integration."""
        print("\n=== Testing Learning System Integration ===")
        
        # Create components
        store = ResearchKnowledgeStore()
        learning = ResearchLearningSystem(store)
        
        # Record outcome
        outcome = {
            "research_quality": 0.8,
            "implementation_success": True,
            "performance_impact": 0.15,
            "value_delivered": 7.5
        }
        
        learning.record_research_outcome("research_1", outcome)
        print("✓ Research outcome recorded")
        
        # Get learning summary
        summary = learning.get_learning_summary()
        self.assertIsInstance(summary, dict)
        print(f"✓ Learning summary: {summary}")
        
        # Get insights
        insights = learning.get_research_insights("performance")
        self.assertIsInstance(insights, dict)
        print("✓ Research insights retrieved")
        
        print("\n✅ Learning system integration test passed!")
    
    def test_knowledge_graph_integration(self):
        """Test knowledge graph builder integration."""
        print("\n=== Testing Knowledge Graph Integration ===")
        
        # Create knowledge graph
        graph = KnowledgeGraphBuilder()
        
        # Process research
        research = {
            "id": "test_1",
            "area": "performance",
            "content": {"content": "Implement caching and optimize queries"},
            "metadata": {"provider": "test"}
        }
        
        results = graph.process_research(research)
        
        self.assertIsInstance(results, dict)
        print("✓ Research processed into knowledge graph")
        
        # Get insights
        insights = graph.get_insights()
        self.assertIsInstance(insights, dict)
        print(f"✓ Graph insights: {insights}")
        
        print("\n✅ Knowledge graph integration test passed!")
    
    def test_insight_processor_integration(self):
        """Test insight processor integration."""
        print("\n=== Testing Insight Processor Integration ===")
        
        # Create insight processor
        processor = ResearchInsightProcessor()
        
        # Process research batch
        research_batch = [
            {
                "id": "r1",
                "content": {"content": "Implement caching for 30% improvement"},
                "area": "performance"
            },
            {
                "id": "r2",
                "content": {"content": "Add error handling for network failures"},
                "area": "reliability"
            }
        ]
        
        results = processor.process_research_batch(research_batch)
        
        self.assertIsInstance(results, dict)
        self.assertIn("insights", results)
        print(f"✓ Processed {len(results['insights'])} insights")
        
        # Get pattern analysis
        patterns = processor.get_pattern_analysis()
        self.assertIsInstance(patterns, dict)
        print(f"✓ Pattern analysis: {patterns['total_patterns']} patterns found")
        
        print("\n✅ Insight processor integration test passed!")
    
    def test_cross_research_analyzer_integration(self):
        """Test cross-research analyzer integration."""
        print("\n=== Testing Cross-Research Analyzer Integration ===")
        
        # Create components
        store = ResearchKnowledgeStore()
        graph = KnowledgeGraphBuilder()
        analyzer = CrossResearchAnalyzer(store, graph)
        
        # Store some research data
        for i in range(3):
            store.store_research("performance", {
                "topic": f"optimization_{i}",
                "content": f"Research content {i}"
            }, quality_score=0.7 + i * 0.1)
        
        # Analyze research corpus
        analysis = analyzer.analyze_research_corpus(time_window=7)
        
        self.assertIsInstance(analysis, dict)
        print("✓ Cross-research analysis completed")
        
        if "pattern_analysis" in analysis:
            print(f"✓ Patterns found: {analysis['pattern_analysis'].get('total_patterns', 0)}")
        
        if "recommendations" in analysis:
            print(f"✓ Recommendations: {len(analysis['recommendations'])}")
        
        print("\n✅ Cross-research analyzer integration test passed!")
    
    async def test_full_research_cycle(self):
        """Test a full research cycle."""
        print("\n=== Testing Full Research Cycle ===")
        
        # Create research evolution engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain,
            task_generator=self.mock_task_generator,
            self_improver=self.mock_self_improver,
            outcome_learning=self.mock_outcome_learning
        )
        
        # Execute a research cycle
        print("✓ Starting research cycle...")
        cycle_results = await engine.execute_research_cycle()
        
        # Verify results
        self.assertIsInstance(cycle_results, dict)
        self.assertEqual(cycle_results["status"], "completed")
        print(f"✓ Research cycle completed in {cycle_results.get('duration_seconds', 0):.2f} seconds")
        
        # Check cycle components
        self.assertIn("research_conducted", cycle_results)
        print(f"✓ Research conducted: {len(cycle_results['research_conducted'])}")
        
        self.assertIn("insights_extracted", cycle_results)
        print(f"✓ Insights extracted: {len(cycle_results['insights_extracted'])}")
        
        self.assertIn("tasks_generated", cycle_results)
        print(f"✓ Tasks generated: {len(cycle_results['tasks_generated'])}")
        
        self.assertIn("implementations", cycle_results)
        print(f"✓ Implementations: {len(cycle_results['implementations'])}")
        
        # Get engine status
        status = engine.get_status()
        self.assertEqual(status["research_cycles_completed"], 1)
        print("✓ Engine status updated correctly")
        
        print("\n✅ Full research cycle test passed!")


def run_tests():
    """Run all tests."""
    # Run synchronous tests
    print("="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    # Load synchronous tests
    suite = unittest.TestSuite()
    
    # Add all test methods except the async one
    test_case = TestResearchIntegration()
    for method_name in dir(test_case):
        if method_name.startswith('test_') and method_name != 'test_full_research_cycle':
            suite.addTest(TestResearchIntegration(method_name))
    
    # Run synchronous tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run async test separately
    print("\n" + "="*60)
    print("RUNNING ASYNC INTEGRATION TEST")
    print("="*60)
    
    async def run_async_test():
        test_case = TestResearchIntegration()
        test_case.setUp()
        await test_case.test_full_research_cycle()
        return True
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_success = loop.run_until_complete(run_async_test())
    loop.close()
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Synchronous tests run: {result.testsRun}")
    print(f"Synchronous failures: {len(result.failures)}")
    print(f"Synchronous errors: {len(result.errors)}")
    print(f"Async test: {'PASSED' if async_success else 'FAILED'}")
    
    success = result.wasSuccessful() and async_success
    
    if success:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("The self-amplifying intelligence system components are working together correctly.")
    else:
        print("\n❌ Some integration tests failed. Please check the output above.")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)