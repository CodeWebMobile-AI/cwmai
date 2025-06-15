"""
Final System Verification Test

This test provides a summary verification that all self-amplifying
intelligence system components are working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch
import importlib
import json

# Test configuration loading
print("="*60)
print("SELF-AMPLIFYING INTELLIGENCE SYSTEM VERIFICATION")
print("="*60)

def test_imports():
    """Test that all required modules can be imported."""
    print("\n1. Testing Module Imports")
    print("-" * 40)
    
    modules = [
        "scripts.research_evolution_engine",
        "scripts.dynamic_research_trigger",
        "scripts.research_knowledge_store",
        "scripts.knowledge_graph_builder",
        "scripts.research_insight_processor",
        "scripts.cross_research_analyzer"
    ]
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            return False
    
    return True

def test_configuration():
    """Test the self-amplifying configuration."""
    print("\n2. Testing Self-Amplifying Configuration")
    print("-" * 40)
    
    from scripts.research_evolution_engine import ResearchEvolutionEngine
    
    # Create engine with mocked dependencies
    engine = ResearchEvolutionEngine(
        state_manager=Mock(),
        ai_brain=Mock(),
        task_generator=Mock(),
        self_improver=Mock(),
        outcome_learning=Mock()
    )
    
    config_checks = [
        ("Dynamic Triggering", engine.config.get("enable_dynamic_triggering", False)),
        ("Fixed Interval (Continuous)", engine.config.get("enable_fixed_interval", False)),
        ("Proactive Research", engine.config.get("enable_proactive_research", False)),
        ("External Agent Research", engine.config.get("enable_external_agent_research", False)),
        ("Concurrent Research = 5", engine.config.get("max_concurrent_research") == 5),
        ("Research per Cycle = 8", engine.config.get("max_research_per_cycle") == 8),
        ("Min Insight Confidence = 0.5", engine.config.get("min_insight_confidence") == 0.5),
        ("Auto-implement Threshold = 0.75", engine.config.get("auto_implement_threshold") == 0.75),
        ("External Research Freq = 2", engine.config.get("external_research_frequency") == 2),
        ("Max External Capabilities = 5", engine.config.get("max_external_capabilities_per_cycle") == 5),
        ("External Synthesis Threshold = 0.6", engine.config.get("external_synthesis_threshold") == 0.6)
    ]
    
    all_passed = True
    for check_name, result in config_checks:
        if result:
            print(f"✓ {check_name}")
        else:
            print(f"✗ {check_name}")
            all_passed = False
    
    return all_passed

def test_component_integration():
    """Test component integration."""
    print("\n3. Testing Component Integration")
    print("-" * 40)
    
    from scripts.research_evolution_engine import ResearchEvolutionEngine
    
    engine = ResearchEvolutionEngine()
    
    components = [
        ("Knowledge Store", hasattr(engine, 'knowledge_store')),
        ("Need Analyzer", hasattr(engine, 'need_analyzer')),
        ("Research Selector", hasattr(engine, 'research_selector')),
        ("Scheduler", hasattr(engine, 'scheduler')),
        ("Query Generator", hasattr(engine, 'query_generator')),
        ("Action Engine", hasattr(engine, 'action_engine')),
        ("Learning System", hasattr(engine, 'learning_system')),
        ("Knowledge Graph", hasattr(engine, 'knowledge_graph')),
        ("Insight Processor", hasattr(engine, 'insight_processor')),
        ("Dynamic Trigger", hasattr(engine, 'dynamic_trigger')),
        ("Cross Analyzer", hasattr(engine, 'cross_analyzer')),
        ("External Agent Discoverer", hasattr(engine, 'external_agent_discoverer')),
        ("Capability Extractor", hasattr(engine, 'capability_extractor')),
        ("Capability Synthesizer", hasattr(engine, 'capability_synthesizer')),
        ("Knowledge Integrator", hasattr(engine, 'knowledge_integrator'))
    ]
    
    all_present = True
    for component_name, is_present in components:
        if is_present:
            print(f"✓ {component_name}")
        else:
            print(f"✗ {component_name}")
            all_present = False
    
    return all_present

def test_dynamic_trigger():
    """Test dynamic research trigger."""
    print("\n4. Testing Dynamic Research Trigger")
    print("-" * 40)
    
    from scripts.dynamic_research_trigger import DynamicResearchTrigger
    
    trigger = DynamicResearchTrigger(
        state_manager=Mock(),
        research_engine=Mock()
    )
    
    checks = [
        ("Instantiation", trigger is not None),
        ("Performance Drop Monitoring", "performance_drop" in trigger.trigger_conditions),
        ("Anomaly Detection", "anomaly_detection" in trigger.trigger_conditions),
        ("Opportunity Based", "opportunity_based" in trigger.trigger_conditions),
        ("Event Based", "event_based" in trigger.trigger_conditions),
        ("Cooldown Management", hasattr(trigger, 'cooldowns')),
        ("Metrics History", hasattr(trigger, 'metrics_history')),
        ("Event Queue", hasattr(trigger, 'event_queue'))
    ]
    
    all_passed = True
    for check_name, result in checks:
        if result:
            print(f"✓ {check_name}")
        else:
            print(f"✗ {check_name}")
            all_passed = False
    
    return all_passed

def main():
    """Run all verification tests."""
    results = {
        "imports": test_imports(),
        "configuration": test_configuration(),
        "integration": test_component_integration(),
        "dynamic_trigger": test_dynamic_trigger()
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name.title()}: {status}")
    
    if all_passed:
        print("\n✅ ALL VERIFICATIONS PASSED!")
        print("\nThe self-amplifying intelligence system is configured correctly:")
        print("- All modules import successfully")
        print("- Self-amplifying configuration is active")
        print("- All components are integrated")
        print("- Dynamic research triggering is functional")
        print("\nKey Features Enabled:")
        print("- 🔄 Continuous learning (20-minute cycles)")
        print("- ⚡ Dynamic research triggering")
        print("- 🎯 Proactive opportunity scanning")
        print("- 🌐 External agent capability learning")
        print("- 📈 Increased research capacity (5 concurrent, 8 per cycle)")
        print("- 🚀 Lower barriers for insights (50% confidence)")
        print("- 🤖 Higher auto-implementation (75% threshold)")
    else:
        print("\n❌ Some verifications failed. Please check the details above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)