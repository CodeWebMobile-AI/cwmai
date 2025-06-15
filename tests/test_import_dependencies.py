"""
Test Import Dependencies

This test checks for any import errors or missing dependencies
in the self-amplifying intelligence system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import importlib
import traceback


class TestImportDependencies(unittest.TestCase):
    """Test that all modules can be imported without errors."""
    
    def test_core_research_modules(self):
        """Test importing core research modules."""
        print("\n=== Testing Core Research Module Imports ===")
        
        modules_to_test = [
            ("research_evolution_engine", "Research Evolution Engine"),
            ("research_knowledge_store", "Research Knowledge Store"),
            ("research_need_analyzer", "Research Need Analyzer"),
            ("intelligent_research_selector", "Intelligent Research Selector"),
            ("research_scheduler", "Research Scheduler"),
            ("research_query_generator", "Research Query Generator"),
            ("research_action_engine", "Research Action Engine"),
            ("research_learning_system", "Research Learning System")
        ]
        
        for module_name, display_name in modules_to_test:
            try:
                module = importlib.import_module(f"scripts.{module_name}")
                print(f"✓ {display_name} imported successfully")
                
                # Check for main classes
                if module_name == "research_evolution_engine":
                    self.assertTrue(hasattr(module, "ResearchEvolutionEngine"))
                elif module_name == "research_scheduler":
                    self.assertTrue(hasattr(module, "ResearchScheduler"))
                    self.assertTrue(hasattr(module, "ResearchPriority"))
                
            except ImportError as e:
                self.fail(f"❌ Failed to import {display_name}: {e}")
            except Exception as e:
                self.fail(f"❌ Error importing {display_name}: {e}")
    
    def test_advanced_research_modules(self):
        """Test importing advanced research modules."""
        print("\n=== Testing Advanced Research Module Imports ===")
        
        modules_to_test = [
            ("knowledge_graph_builder", "Knowledge Graph Builder"),
            ("research_insight_processor", "Research Insight Processor"),
            ("dynamic_research_trigger", "Dynamic Research Trigger"),
            ("cross_research_analyzer", "Cross-Research Analyzer")
        ]
        
        for module_name, display_name in modules_to_test:
            try:
                module = importlib.import_module(f"scripts.{module_name}")
                print(f"✓ {display_name} imported successfully")
                
                # Check for main classes
                if module_name == "knowledge_graph_builder":
                    self.assertTrue(hasattr(module, "KnowledgeGraphBuilder"))
                elif module_name == "dynamic_research_trigger":
                    self.assertTrue(hasattr(module, "DynamicResearchTrigger"))
                
            except ImportError as e:
                self.fail(f"❌ Failed to import {display_name}: {e}")
            except Exception as e:
                self.fail(f"❌ Error importing {display_name}: {e}")
    
    def test_external_learning_modules(self):
        """Test importing external learning modules."""
        print("\n=== Testing External Learning Module Imports ===")
        
        modules_to_test = [
            ("external_agent_discoverer", "External Agent Discoverer"),
            ("capability_extractor", "Capability Extractor"),
            ("capability_synthesizer", "Capability Synthesizer"),
            ("external_knowledge_integrator", "External Knowledge Integrator")
        ]
        
        for module_name, display_name in modules_to_test:
            try:
                module = importlib.import_module(f"scripts.{module_name}")
                print(f"✓ {display_name} imported successfully")
                
                # Check for main classes
                if module_name == "external_agent_discoverer":
                    self.assertTrue(hasattr(module, "ExternalAgentDiscoverer"))
                    self.assertTrue(hasattr(module, "DiscoveryConfig"))
                
            except ImportError as e:
                self.fail(f"❌ Failed to import {display_name}: {e}")
            except Exception as e:
                self.fail(f"❌ Error importing {display_name}: {e}")
    
    def test_dependency_chain(self):
        """Test the full dependency chain by importing ResearchEvolutionEngine."""
        print("\n=== Testing Full Dependency Chain ===")
        
        try:
            # This will test all transitive dependencies
            from scripts.research_evolution_engine import ResearchEvolutionEngine
            
            # Try to instantiate with None parameters (just to test construction)
            engine = ResearchEvolutionEngine()
            
            print("✓ ResearchEvolutionEngine instantiated with default parameters")
            
            # Check that all components are created
            self.assertIsNotNone(engine.knowledge_store)
            self.assertIsNotNone(engine.need_analyzer)
            self.assertIsNotNone(engine.research_selector)
            self.assertIsNotNone(engine.scheduler)
            self.assertIsNotNone(engine.query_generator)
            self.assertIsNotNone(engine.action_engine)
            self.assertIsNotNone(engine.learning_system)
            self.assertIsNotNone(engine.knowledge_graph)
            self.assertIsNotNone(engine.insight_processor)
            self.assertIsNotNone(engine.dynamic_trigger)
            self.assertIsNotNone(engine.cross_analyzer)
            self.assertIsNotNone(engine.external_agent_discoverer)
            self.assertIsNotNone(engine.capability_extractor)
            self.assertIsNotNone(engine.capability_synthesizer)
            self.assertIsNotNone(engine.knowledge_integrator)
            
            print("✓ All components initialized successfully")
            
        except Exception as e:
            print(f"\n❌ Error in dependency chain:")
            traceback.print_exc()
            self.fail(f"Dependency chain test failed: {e}")
    
    def test_required_packages(self):
        """Test that required Python packages are available."""
        print("\n=== Testing Required Python Packages ===")
        
        required_packages = [
            ("asyncio", "Asyncio (built-in)"),
            ("datetime", "Datetime (built-in)"),
            ("json", "JSON (built-in)"),
            ("logging", "Logging (built-in)"),
            ("pathlib", "Pathlib (built-in)"),
            ("collections", "Collections (built-in)"),
            ("enum", "Enum (built-in)"),
            ("typing", "Typing (built-in)")
        ]
        
        for package_name, display_name in required_packages:
            try:
                module = importlib.import_module(package_name)
                print(f"✓ {display_name} available")
            except ImportError:
                self.fail(f"❌ Required package {display_name} not available")
    
    def test_configuration_loading(self):
        """Test that configuration can be loaded properly."""
        print("\n=== Testing Configuration Loading ===")
        
        try:
            from scripts.research_evolution_engine import ResearchEvolutionEngine
            
            # Create engine
            engine = ResearchEvolutionEngine()
            
            # Check configuration
            self.assertIsInstance(engine.config, dict)
            print(f"✓ Configuration loaded: {len(engine.config)} settings")
            
            # Check specific configuration values
            required_configs = [
                "max_concurrent_research",
                "cycle_interval_seconds",
                "emergency_cycle_interval",
                "max_research_per_cycle",
                "min_insight_confidence",
                "auto_implement_threshold",
                "enable_dynamic_triggering",
                "enable_fixed_interval",
                "enable_external_agent_research",
                "external_research_frequency",
                "ai_papers_repositories",
                "max_external_capabilities_per_cycle",
                "external_synthesis_threshold",
                "enable_proactive_research"
            ]
            
            for config_name in required_configs:
                self.assertIn(config_name, engine.config)
                print(f"✓ Config '{config_name}': {engine.config[config_name]}")
            
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")


def run_tests():
    """Run all import dependency tests."""
    print("="*60)
    print("IMPORT DEPENDENCY TESTS")
    print("="*60)
    print("Testing that all modules can be imported without errors...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImportDependencies)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("IMPORT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL IMPORT TESTS PASSED!")
        print("All modules can be imported successfully.")
        print("No missing dependencies detected.")
    else:
        print("\n❌ Some import tests failed.")
        print("Please check the errors above and install missing dependencies.")
        
        # Try to provide helpful information about failures
        if result.failures or result.errors:
            print("\nCommon solutions:")
            print("1. Ensure all required files exist in the scripts/ directory")
            print("2. Check that __init__.py exists in the scripts/ directory")
            print("3. Verify that all module names are spelled correctly")
            print("4. Install any missing Python packages with pip")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)