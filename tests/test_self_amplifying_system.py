"""
Test Self-Amplifying Intelligence System

This test verifies that the self-amplifying intelligence system components
work correctly together without external dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio
from datetime import datetime
import json
import tempfile
import shutil

# Import the main engine
from scripts.research_evolution_engine import ResearchEvolutionEngine


class TestSelfAmplifyingSystem(unittest.TestCase):
    """Test the self-amplifying intelligence system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_state_manager = Mock()
        self.mock_ai_brain = AsyncMock()
        self.mock_task_generator = Mock()
        self.mock_self_improver = Mock()
        self.mock_outcome_learning = Mock()
        
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
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
                    {"id": "2", "status": "failed", "created_at": "2024-01-02", 
                     "failure_reason": "timeout"},
                    {"id": "3", "status": "in_progress", "created_at": "2024-01-03"}
                ]
            },
            "recent_errors": [
                {"type": "timeout", "message": "API timeout", "timestamp": "2024-01-02"}
            ],
            "projects": {"project1": {}, "project2": {}},
            "metrics": {
                "claude_interactions": {"success_rate": 0.85}
            }
        }
        
        self.mock_state_manager.load_state.return_value = self.mock_state
        
        # Mock AI brain responses
        self.mock_ai_brain.generate_enhanced_response.return_value = {
            "content": """
            Research insight: To improve system performance:
            1. Implement caching to reduce API calls by 40%
            2. Use connection pooling for better resource management
            3. Add retry logic with exponential backoff
            
            Expected improvement: 30-40% reduction in errors, 25% faster response times.
            """
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_system_initialization(self):
        """Test that the self-amplifying system initializes correctly."""
        print("\n=== Testing Self-Amplifying System Initialization ===")
        
        # Create engine with self-amplifying configuration
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain,
            task_generator=self.mock_task_generator,
            self_improver=self.mock_self_improver,
            outcome_learning=self.mock_outcome_learning
        )
        
        # Verify self-amplifying features are enabled
        self.assertTrue(engine.config["enable_dynamic_triggering"])
        print("✓ Dynamic triggering enabled")
        
        self.assertTrue(engine.config["enable_fixed_interval"])
        print("✓ Fixed interval research enabled (continuous learning)")
        
        self.assertTrue(engine.config["enable_proactive_research"])
        print("✓ Proactive research enabled")
        
        self.assertTrue(engine.config["enable_external_agent_research"])
        print("✓ External agent research enabled")
        
        # Check increased capacities
        self.assertEqual(engine.config["max_concurrent_research"], 5)
        print(f"✓ Increased concurrent research: {engine.config['max_concurrent_research']}")
        
        self.assertEqual(engine.config["max_research_per_cycle"], 8)
        print(f"✓ Increased research per cycle: {engine.config['max_research_per_cycle']}")
        
        self.assertEqual(engine.config["min_insight_confidence"], 0.5)
        print(f"✓ Lower insight threshold: {engine.config['min_insight_confidence']}")
        
        self.assertEqual(engine.config["auto_implement_threshold"], 0.75)
        print(f"✓ Higher auto-implement threshold: {engine.config['auto_implement_threshold']}")
        
        print("\n✅ Self-amplifying system initialized successfully!")
    
    async def test_research_cycle_execution(self):
        """Test that a research cycle executes with self-amplifying features."""
        print("\n=== Testing Research Cycle with Self-Amplifying Features ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain,
            task_generator=self.mock_task_generator,
            self_improver=self.mock_self_improver,
            outcome_learning=self.mock_outcome_learning
        )
        
        # Patch file operations to avoid file system dependencies
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = "{}"
                    
                    # Execute research cycle
                    results = await engine.execute_research_cycle()
        
        # Verify cycle executed
        self.assertEqual(results["status"], "completed")
        print("✓ Research cycle completed successfully")
        
        # Check that research was conducted
        self.assertIn("research_conducted", results)
        print(f"✓ Research conducted: {len(results['research_conducted'])} items")
        
        # Verify AI brain was called
        self.mock_ai_brain.generate_enhanced_response.assert_called()
        print("✓ AI brain utilized for research")
        
        # Check insights extraction
        self.assertIn("insights_extracted", results)
        print(f"✓ Insights extracted: {len(results['insights_extracted'])} items")
        
        # Verify proactive elements
        self.assertEqual(engine.research_cycles, 1)
        print(f"✓ Research cycle counter updated: {engine.research_cycles}")
        
        print("\n✅ Research cycle with self-amplifying features executed successfully!")
    
    def test_dynamic_trigger_integration(self):
        """Test that dynamic trigger integrates with the engine."""
        print("\n=== Testing Dynamic Trigger Integration ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Verify dynamic trigger is created
        self.assertIsNotNone(engine.dynamic_trigger)
        print("✓ Dynamic trigger component created")
        
        # Test event addition
        engine.dynamic_trigger.add_event("critical_error", {
            "error": "System failure",
            "severity": "critical"
        })
        
        self.assertEqual(len(engine.dynamic_trigger.event_queue), 1)
        print("✓ Events can be added to dynamic trigger")
        
        # Test metrics collection
        metrics = engine.dynamic_trigger._collect_current_metrics()
        self.assertEqual(metrics["claude_success_rate"], 85.0)
        print(f"✓ Dynamic trigger can collect metrics: {metrics['claude_success_rate']}%")
        
        print("\n✅ Dynamic trigger integration test passed!")
    
    def test_external_research_configuration(self):
        """Test external research configuration."""
        print("\n=== Testing External Research Configuration ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Check external research status
        external_status = engine.get_external_research_status()
        
        self.assertTrue(external_status["external_research_enabled"])
        print("✓ External research is enabled")
        
        self.assertEqual(external_status["external_research_frequency"], 2)
        print(f"✓ External research frequency: every {external_status['external_research_frequency']} cycles")
        
        self.assertIn("ai_papers_repositories", external_status)
        self.assertGreater(len(external_status["ai_papers_repositories"]), 0)
        print(f"✓ AI papers repositories configured: {len(external_status['ai_papers_repositories'])}")
        
        # Check external components
        components = external_status["external_components_status"]
        self.assertEqual(components["agent_discoverer"], "active")
        self.assertEqual(components["capability_extractor"], "active")
        self.assertEqual(components["capability_synthesizer"], "active")
        self.assertEqual(components["knowledge_integrator"], "active")
        print("✓ All external learning components active")
        
        print("\n✅ External research configuration test passed!")
    
    def test_performance_context_gathering(self):
        """Test that the system can gather performance context."""
        print("\n=== Testing Performance Context Gathering ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Get performance context
        context = engine._get_performance_context()
        
        self.assertEqual(context["claude_success_rate"], 85.0)
        print(f"✓ Claude success rate: {context['claude_success_rate']}%")
        
        self.assertEqual(context["task_completion_rate"], 90.0)
        print(f"✓ Task completion rate: {context['task_completion_rate']}%")
        
        self.assertEqual(context["system_health"], "good")
        print(f"✓ System health assessment: {context['system_health']}")
        
        self.assertIn("recent_errors", context)
        print(f"✓ Recent errors tracked: {len(context['recent_errors'])}")
        
        self.assertIn("projects", context)
        print(f"✓ Projects tracked: {len(context['projects'])}")
        
        print("\n✅ Performance context gathering test passed!")
    
    def test_learning_and_adaptation(self):
        """Test learning and adaptation capabilities."""
        print("\n=== Testing Learning and Adaptation ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Test need analysis
        needs = engine.need_analyzer.analyze_performance_gaps()
        self.assertIsInstance(needs, dict)
        print("✓ Performance gaps can be analyzed")
        
        # Test proactive opportunities
        opportunities = engine.need_analyzer.get_proactive_research_opportunities()
        self.assertIsInstance(opportunities, list)
        print(f"✓ Proactive opportunities identified: {len(opportunities)}")
        
        # Test learning trigger
        events = [
            {"type": "task", "status": "failed"},
            {"type": "task", "status": "failed"},
            {"type": "error", "message": "repeated error"}
        ]
        should_learn = engine.need_analyzer.should_trigger_learning_research(events)
        self.assertTrue(should_learn)
        print("✓ Learning research triggered by failure patterns")
        
        # Test effectiveness metrics
        metrics = engine._get_effectiveness_metrics()
        self.assertIn("research_to_insight_ratio", metrics)
        self.assertIn("learning_efficiency", metrics)
        print("✓ Effectiveness metrics calculated")
        
        print("\n✅ Learning and adaptation test passed!")
    
    def test_continuous_improvement_loop(self):
        """Test the continuous improvement loop configuration."""
        print("\n=== Testing Continuous Improvement Loop ===")
        
        # Create engine
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Check cycle intervals
        self.assertEqual(engine.config["cycle_interval_seconds"], 20 * 60)  # 20 minutes
        print(f"✓ Regular cycle interval: {engine.config['cycle_interval_seconds'] / 60} minutes")
        
        self.assertEqual(engine.config["emergency_cycle_interval"], 3 * 60)  # 3 minutes
        print(f"✓ Emergency cycle interval: {engine.config['emergency_cycle_interval'] / 60} minutes")
        
        # Test cycle interval calculation
        normal_interval = engine._calculate_next_cycle_interval()
        self.assertEqual(normal_interval, engine.config["cycle_interval_seconds"])
        print("✓ Normal cycle interval calculated correctly")
        
        # Simulate critical state
        engine._assess_system_health = Mock(return_value="critical")
        emergency_interval = engine._calculate_next_cycle_interval()
        self.assertEqual(emergency_interval, engine.config["emergency_cycle_interval"])
        print("✓ Emergency cycle interval triggered for critical state")
        
        print("\n✅ Continuous improvement loop test passed!")


def run_tests():
    """Run all tests."""
    # Run synchronous tests
    print("="*60)
    print("SELF-AMPLIFYING INTELLIGENCE SYSTEM TESTS")
    print("="*60)
    
    # Create test suite for sync tests
    suite = unittest.TestSuite()
    
    # Add all test methods except async ones
    test_case = TestSelfAmplifyingSystem()
    for method_name in dir(test_case):
        if method_name.startswith('test_') and not method_name.startswith('test_research_cycle'):
            suite.addTest(TestSelfAmplifyingSystem(method_name))
    
    # Run synchronous tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run async test separately
    print("\n" + "="*60)
    print("RUNNING ASYNC TESTS")
    print("="*60)
    
    async def run_async_test():
        test_case = TestSelfAmplifyingSystem()
        test_case.setUp()
        await test_case.test_research_cycle_execution()
        test_case.tearDown()
        return True
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_success = loop.run_until_complete(run_async_test())
    loop.close()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Synchronous tests run: {result.testsRun}")
    print(f"Synchronous failures: {len(result.failures)}")
    print(f"Synchronous errors: {len(result.errors)}")
    print(f"Async test: {'PASSED' if async_success else 'FAILED'}")
    
    success = result.wasSuccessful() and async_success
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe self-amplifying intelligence system is working correctly:")
        print("- Research evolution engine configured with amplified settings")
        print("- Dynamic research triggering functional")
        print("- Continuous learning enabled")
        print("- Proactive research opportunities detected")
        print("- External agent research configured")
        print("- Performance monitoring and adaptation working")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)