#!/usr/bin/env python3
"""
Test Full System Startup

This test verifies that the entire self-amplifying system can start up
without errors and all components initialize correctly.
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

def test_continuous_orchestrator():
    """Test that continuous orchestrator can be imported and initialized."""
    print("Testing Continuous Orchestrator initialization...")
    
    try:
        from continuous_orchestrator import ContinuousOrchestrator
        
        # Create instance
        orchestrator = ContinuousOrchestrator()
        
        # Check components
        assert orchestrator.state_manager is not None, "State manager not initialized"
        assert hasattr(orchestrator, 'research_evolver'), "Research evolver not found"
        
        print("✅ Continuous Orchestrator initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Continuous Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_evolution_startup():
    """Test research evolution engine startup."""
    print("\nTesting Research Evolution Engine startup...")
    
    try:
        from research_evolution_engine import ResearchEvolutionEngine
        from state_manager import StateManager
        
        # Create with state manager
        state_manager = StateManager()
        engine = ResearchEvolutionEngine(state_manager=state_manager)
        
        # Verify configuration
        config_checks = {
            "enable_fixed_interval": True,
            "enable_dynamic_triggering": True,
            "enable_proactive_research": True,
            "cycle_interval_seconds": 20 * 60,
            "max_research_per_cycle": 8
        }
        
        for key, expected in config_checks.items():
            actual = engine.config.get(key)
            assert actual == expected, f"{key}: expected {expected}, got {actual}"
        
        print("✅ Research Evolution Engine configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ Research Evolution Engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_startup():
    """Test async components startup."""
    print("\nTesting async components startup...")
    
    try:
        from research_evolution_engine import ResearchEvolutionEngine
        from state_manager import StateManager
        
        # Create engine
        state_manager = StateManager()
        engine = ResearchEvolutionEngine(state_manager=state_manager)
        
        # Test initialization of dynamic components
        await engine.dynamic_trigger.initialize()
        
        # Verify dynamic trigger is ready
        assert engine.dynamic_trigger._initialized, "Dynamic trigger not initialized"
        assert engine.dynamic_trigger.monitoring_active, "Monitoring not active"
        
        # Stop monitoring
        engine.dynamic_trigger.stop_monitoring()
        
        print("✅ Async components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Async components failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_directories():
    """Test that required directories exist."""
    print("\nTesting knowledge directories...")
    
    required_dirs = [
        "research_knowledge",
        "research_knowledge/raw_research",
        "research_knowledge/processed_insights",
        "research_knowledge/metadata",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_ai_client():
    """Test HTTP AI client initialization."""
    print("\nTesting HTTP AI Client...")
    
    try:
        from http_ai_client import HTTPAIClient
        
        client = HTTPAIClient()
        
        # Check providers
        providers = client.providers_available
        print(f"  Available providers: {sum(providers.values())} of {len(providers)}")
        
        for provider, available in providers.items():
            status = "✅" if available else "⚠️"
            print(f"    {status} {provider}")
        
        # At least one provider should be available
        assert any(providers.values()), "No AI providers available"
        
        print("✅ HTTP AI Client ready")
        return True
        
    except Exception as e:
        print(f"❌ HTTP AI Client failed: {e}")
        return False


def test_state_persistence():
    """Test state persistence."""
    print("\nTesting state persistence...")
    
    try:
        from state_manager import StateManager
        
        # Create and save test state
        sm = StateManager()
        test_data = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "self_amplifying_active": True
        }
        sm.state.update(test_data)
        sm.save_state()
        
        # Load and verify
        sm2 = StateManager()
        loaded_state = sm2.load_state()
        
        assert loaded_state.get("self_amplifying_active") == True, "State not persisted"
        
        print("✅ State persistence working")
        return True
        
    except Exception as e:
        print(f"❌ State persistence failed: {e}")
        return False


def main():
    """Run all startup tests."""
    print("=" * 60)
    print("CWMAI Self-Amplifying System Startup Test")
    print("=" * 60)
    
    tests = [
        test_continuous_orchestrator,
        test_research_evolution_startup,
        test_knowledge_directories,
        test_ai_client,
        test_state_persistence
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Run async test
    print("\nRunning async tests...")
    async_result = asyncio.run(test_async_startup())
    results.append(async_result)
    
    # Summary
    print("\n" + "=" * 60)
    print("STARTUP TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✨ ALL SYSTEMS GO! ✨")
        print("The self-amplifying intelligence system is ready to start.")
        print("\nTo begin: python start_self_amplifying_ai.py")
        return 0
    else:
        print("\n⚠️  Some components need attention")
        print("Review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)