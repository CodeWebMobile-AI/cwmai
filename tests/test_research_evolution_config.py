"""
Test Research Evolution Engine Configuration

This test verifies that the research evolution engine can be instantiated
with the new self-amplifying configuration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock
import asyncio
from datetime import datetime

# Import the research evolution engine
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.research_scheduler import ResearchPriority


class TestResearchEvolutionConfig(unittest.TestCase):
    """Test the research evolution engine configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_state_manager = Mock()
        self.mock_ai_brain = Mock()
        self.mock_task_generator = Mock()
        self.mock_self_improver = Mock()
        self.mock_outcome_learning = Mock()
        
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
                "tasks": []
            },
            "recent_errors": []
        }
    
    def test_engine_instantiation(self):
        """Test that the engine can be instantiated with the new configuration."""
        print("\n=== Testing Research Evolution Engine Instantiation ===")
        
        try:
            engine = ResearchEvolutionEngine(
                state_manager=self.mock_state_manager,
                ai_brain=self.mock_ai_brain,
                task_generator=self.mock_task_generator,
                self_improver=self.mock_self_improver,
                outcome_learning=self.mock_outcome_learning
            )
            
            # Verify engine was created
            self.assertIsNotNone(engine)
            print("✓ Engine instantiated successfully")
            
            # Check configuration
            self.assertIsInstance(engine.config, dict)
            print(f"✓ Configuration loaded: {len(engine.config)} settings")
            
            # Verify self-amplifying settings
            self.assertTrue(engine.config.get("enable_dynamic_triggering"))
            print(f"✓ Dynamic triggering enabled: {engine.config['enable_dynamic_triggering']}")
            
            self.assertTrue(engine.config.get("enable_fixed_interval"))
            print(f"✓ Fixed interval enabled: {engine.config['enable_fixed_interval']}")
            
            self.assertTrue(engine.config.get("enable_proactive_research"))
            print(f"✓ Proactive research enabled: {engine.config['enable_proactive_research']}")
            
            # Check increased limits
            self.assertEqual(engine.config["max_concurrent_research"], 5)
            print(f"✓ Concurrent research limit: {engine.config['max_concurrent_research']}")
            
            self.assertEqual(engine.config["max_research_per_cycle"], 8)
            print(f"✓ Research per cycle: {engine.config['max_research_per_cycle']}")
            
            self.assertEqual(engine.config["min_insight_confidence"], 0.5)
            print(f"✓ Min insight confidence: {engine.config['min_insight_confidence']}")
            
            self.assertEqual(engine.config["auto_implement_threshold"], 0.75)
            print(f"✓ Auto-implement threshold: {engine.config['auto_implement_threshold']}")
            
            # External research settings
            self.assertTrue(engine.config.get("enable_external_agent_research"))
            print(f"✓ External agent research enabled: {engine.config['enable_external_agent_research']}")
            
            self.assertEqual(engine.config["external_research_frequency"], 2)
            print(f"✓ External research frequency: {engine.config['external_research_frequency']}")
            
            self.assertEqual(engine.config["max_external_capabilities_per_cycle"], 5)
            print(f"✓ Max external capabilities: {engine.config['max_external_capabilities_per_cycle']}")
            
            print("\n✅ All configuration tests passed!")
            
        except Exception as e:
            self.fail(f"Failed to instantiate engine: {e}")
    
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        print("\n=== Testing Component Initialization ===")
        
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain,
            task_generator=self.mock_task_generator,
            self_improver=self.mock_self_improver,
            outcome_learning=self.mock_outcome_learning
        )
        
        # Check research components
        self.assertIsNotNone(engine.knowledge_store)
        print("✓ Knowledge store initialized")
        
        self.assertIsNotNone(engine.need_analyzer)
        print("✓ Need analyzer initialized")
        
        self.assertIsNotNone(engine.research_selector)
        print("✓ Research selector initialized")
        
        self.assertIsNotNone(engine.scheduler)
        print("✓ Scheduler initialized")
        
        self.assertIsNotNone(engine.query_generator)
        print("✓ Query generator initialized")
        
        self.assertIsNotNone(engine.action_engine)
        print("✓ Action engine initialized")
        
        self.assertIsNotNone(engine.learning_system)
        print("✓ Learning system initialized")
        
        # Check new components
        self.assertIsNotNone(engine.knowledge_graph)
        print("✓ Knowledge graph builder initialized")
        
        self.assertIsNotNone(engine.insight_processor)
        print("✓ Insight processor initialized")
        
        self.assertIsNotNone(engine.dynamic_trigger)
        print("✓ Dynamic trigger initialized")
        
        self.assertIsNotNone(engine.cross_analyzer)
        print("✓ Cross-research analyzer initialized")
        
        # Check external learning components
        self.assertIsNotNone(engine.external_agent_discoverer)
        print("✓ External agent discoverer initialized")
        
        self.assertIsNotNone(engine.capability_extractor)
        print("✓ Capability extractor initialized")
        
        self.assertIsNotNone(engine.capability_synthesizer)
        print("✓ Capability synthesizer initialized")
        
        self.assertIsNotNone(engine.knowledge_integrator)
        print("✓ Knowledge integrator initialized")
        
        print("\n✅ All components initialized successfully!")
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        print("\n=== Testing Metrics Initialization ===")
        
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain
        )
        
        # Check metrics
        self.assertIsInstance(engine.metrics, dict)
        print(f"✓ Metrics dictionary created with {len(engine.metrics)} metrics")
        
        # Check specific metrics
        expected_metrics = [
            "research_effectiveness",
            "implementation_success_rate",
            "performance_improvement_rate",
            "learning_accuracy",
            "external_research_cycles",
            "external_repositories_analyzed",
            "capabilities_extracted",
            "capabilities_synthesized",
            "external_integrations_successful"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, engine.metrics)
            print(f"✓ Metric '{metric}' initialized: {engine.metrics[metric]}")
        
        print("\n✅ All metrics initialized correctly!")
    
    def test_async_methods_exist(self):
        """Test that all required async methods exist."""
        print("\n=== Testing Async Methods ===")
        
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Check async methods
        async_methods = [
            "start_continuous_research",
            "execute_research_cycle",
            "execute_emergency_research",
            "_analyze_system_needs",
            "_select_research_topics",
            "_execute_research",
            "_extract_insights",
            "_generate_implementation_tasks",
            "_implement_improvements",
            "_measure_performance_changes",
            "_learn_from_cycle",
            "_execute_external_agent_research",
            "_perform_cross_research_analysis",
            "trigger_manual_research",
            "execute_targeted_external_research"
        ]
        
        for method_name in async_methods:
            self.assertTrue(hasattr(engine, method_name))
            method = getattr(engine, method_name)
            self.assertTrue(asyncio.iscoroutinefunction(method))
            print(f"✓ Async method '{method_name}' exists")
        
        print("\n✅ All async methods verified!")
    
    def test_status_methods(self):
        """Test status and monitoring methods."""
        print("\n=== Testing Status Methods ===")
        
        engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager
        )
        
        # Test get_status
        status = engine.get_status()
        self.assertIsInstance(status, dict)
        print("✓ get_status() returns dictionary")
        
        expected_keys = [
            "is_running",
            "research_cycles_completed",
            "cycle_statistics",
            "current_metrics",
            "learning_summary",
            "knowledge_store_stats",
            "scheduler_status",
            "effectiveness_metrics"
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
            print(f"✓ Status contains '{key}'")
        
        # Test external research status
        external_status = engine.get_external_research_status()
        self.assertIsInstance(external_status, dict)
        print("\n✓ get_external_research_status() returns dictionary")
        
        self.assertTrue(external_status["external_research_enabled"])
        print(f"✓ External research enabled: {external_status['external_research_enabled']}")
        
        print("\n✅ All status methods working correctly!")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResearchEvolutionConfig)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! The research evolution engine is configured correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)