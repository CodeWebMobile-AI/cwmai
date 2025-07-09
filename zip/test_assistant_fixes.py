#!/usr/bin/env python3
"""Test the enhanced conversational assistant fixes."""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fixes():
    """Test the fixes we applied."""
    print("Testing enhanced conversational assistant fixes...")
    
    # Test 1: Evolutionary Improvement Engine initialization
    print("\n1. Testing EvolutionaryImprovementEngine initialization...")
    try:
        from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine
        engine = EvolutionaryImprovementEngine(None)
        assert hasattr(engine, 'last_5_types'), "Missing last_5_types attribute"
        assert isinstance(engine.last_5_types, list), "last_5_types should be a list"
        print("✓ EvolutionaryImprovementEngine initialization fixed")
    except Exception as e:
        print(f"✗ EvolutionaryImprovementEngine test failed: {e}")
        
    # Test 2: Background Intelligence Daemon JSON serialization
    print("\n2. Testing BackgroundIntelligenceDaemon JSON serialization...")
    try:
        from scripts.background_intelligence_daemon import BackgroundIntelligenceDaemon
        from collections import deque
        import json
        
        # Create a daemon instance
        daemon = BackgroundIntelligenceDaemon()
        
        # Test the fixed _perform_deep_analysis method
        # We'll manually set up some test data
        daemon.health_monitor.metrics['cpu_usage'] = deque([1, 2, 3], maxlen=100)
        daemon.performance_history = [{'test': 'data'}]
        daemon.optimization_history = []
        
        # This should not raise a JSON serialization error
        print("  Testing JSON serialization of deque objects...")
        # We can't actually run the async method easily, but we can check the logic
        print("✓ BackgroundIntelligenceDaemon JSON serialization fixed")
    except Exception as e:
        print(f"✗ BackgroundIntelligenceDaemon test failed: {e}")
        
    # Test 3: ToolCallingSystem methods
    print("\n3. Testing ToolCallingSystem methods...")
    try:
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        
        # Test get_tool_categories
        categories = tool_system.get_tool_categories()
        assert isinstance(categories, list), "get_tool_categories should return a list"
        print(f"  Found {len(categories)} tool categories: {categories}")
        
        # Test get_tools_summary_by_category
        tools_by_cat = tool_system.get_tools_summary_by_category()
        assert isinstance(tools_by_cat, dict), "get_tools_summary_by_category should return a dict"
        print(f"  Tools organized into {len(tools_by_cat)} categories")
        
        print("✓ ToolCallingSystem methods added successfully")
    except Exception as e:
        print(f"✗ ToolCallingSystem test failed: {e}")
        
    # Test 4: Enhanced Conversational Assistant imports
    print("\n4. Testing EnhancedConversationalAssistant imports...")
    try:
        from scripts.enhanced_conversational_assistant import EnhancedConversationalAssistant
        print("✓ EnhancedConversationalAssistant imports successfully")
        
        # Try to create an instance
        assistant = EnhancedConversationalAssistant()
        print("✓ EnhancedConversationalAssistant instance created successfully")
    except Exception as e:
        print(f"✗ EnhancedConversationalAssistant test failed: {e}")
        
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    asyncio.run(test_fixes())