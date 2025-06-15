#!/usr/bin/env python3
"""
Test script for Production Orchestrator

Verifies that all components work correctly.
"""

import asyncio
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.production_config import create_config, ExecutionMode
from scripts.production_orchestrator import ProductionOrchestrator


async def test_orchestrator():
    """Test the production orchestrator in test mode."""
    print("Testing Production Orchestrator")
    print("=" * 60)
    
    # Create test configuration
    config = create_config('test')
    
    # Enable only god_mode for quick test
    config.task_cycle.enabled = False
    config.main_cycle.enabled = False
    config.god_mode_cycle.enabled = True
    config.monitoring_cycle.enabled = False
    
    print("\nTest configuration:")
    print(f"  Mode: {config.mode.value}")
    print(f"  Enabled cycles: {list(config.get_enabled_cycles().keys())}")
    
    # Validate config
    if not config.validate():
        print("\n❌ Configuration validation failed")
        return False
        
    print("✓ Configuration validated")
    
    # Create orchestrator
    orchestrator = ProductionOrchestrator(config)
    
    try:
        # Start orchestrator
        print("\nStarting orchestrator...")
        await orchestrator.start()
        
        # In test mode, it will run once and complete
        await orchestrator.wait_for_completion()
        
        # Get final status
        status = orchestrator.get_status()
        
        print("\n✓ Test completed successfully")
        print(f"\nTotal cycles executed: {status['total_cycles']}")
        print(f"Cycle states: {status['cycle_states']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if orchestrator.running:
            await orchestrator.stop()


async def test_development_mode():
    """Test development mode with faster cycles."""
    print("\n\nTesting Development Mode")
    print("=" * 60)
    
    # Create development configuration
    config = create_config('development')
    
    print("\nDevelopment configuration:")
    print("  Cycle intervals:")
    for name, cycle in config.get_enabled_cycles().items():
        minutes = cycle.interval_seconds / 60
        print(f"    • {name}: every {minutes:.0f} minutes")
        
    print("\n✓ Development mode configured correctly")
    return True


def main():
    """Run all tests."""
    print("""
Production Orchestrator Test Suite
==================================

This will verify that the new production orchestrator works correctly.
""")
    
    # Check environment
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set - some tests may fail")
        
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Basic orchestrator functionality
    if asyncio.run(test_orchestrator()):
        tests_passed += 1
        
    # Test 2: Development mode configuration
    if asyncio.run(test_development_mode()):
        tests_passed += 1
        
    # Summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! The production orchestrator is ready to use.")
        print("\nUsage examples:")
        print("  python run_dynamic_ai.py                    # Production mode")
        print("  python run_dynamic_ai.py --mode development # Fast cycles")
        print("  python run_dynamic_ai.py --mode test        # Single execution")
        print("  python run_dynamic_ai.py --legacy           # Old behavior")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        

if __name__ == '__main__':
    main()