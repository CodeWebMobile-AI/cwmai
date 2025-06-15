#!/usr/bin/env python3
"""Test script to verify all fixes are working correctly."""

import asyncio
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

async def test_redis_client_sismember():
    """Test that RedisClient has sismember method."""
    print("Testing Redis Client sismember fix...")
    try:
        from redis_integration.redis_client import RedisClient
        client = RedisClient()
        # Check that the method exists
        assert hasattr(client, 'sismember'), "RedisClient missing sismember method"
        assert hasattr(client, 'zrangebyscore'), "RedisClient missing zrangebyscore method"
        assert hasattr(client, 'zremrangebyrank'), "RedisClient missing zremrangebyrank method"
        print("✅ Redis Client methods fixed successfully")
        return True
    except Exception as e:
        print(f"❌ Redis Client test failed: {e}")
        return False

async def test_research_knowledge_store():
    """Test that ResearchKnowledgeStore has get_insights method."""
    print("\nTesting ResearchKnowledgeStore get_insights fix...")
    try:
        from research_knowledge_store import ResearchKnowledgeStore
        store = ResearchKnowledgeStore()
        # Check that the method exists
        assert hasattr(store, 'get_insights'), "ResearchKnowledgeStore missing get_insights method"
        # Test the method
        insights = store.get_insights('continuous_improvement')
        assert isinstance(insights, list), "get_insights should return a list"
        print("✅ ResearchKnowledgeStore get_insights fixed successfully")
        return True
    except Exception as e:
        print(f"❌ ResearchKnowledgeStore test failed: {e}")
        return False

async def test_capability_synthesizer():
    """Test that CapabilitySynthesizer handles string patterns."""
    print("\nTesting CapabilitySynthesizer string/dict fix...")
    try:
        from capability_synthesizer import CapabilitySynthesizer
        from capability_extractor import ExtractedCapability, CapabilityType, ExtractionMethod, IntegrationComplexity
        
        synthesizer = CapabilitySynthesizer()
        
        # Create a test capability with string patterns
        test_capability = ExtractedCapability(
            id="test_capability",
            name="Test Capability",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Test capability for validation",
            source_repository="test_repo",
            source_files=["test.py"],
            extraction_method=ExtractionMethod.AST_ANALYSIS,
            integration_complexity=IntegrationComplexity.SIMPLE,
            patterns=["test_pattern"],  # String patterns instead of dicts
            interfaces=["test_interface"]  # String interfaces instead of dicts
        )
        
        # This should not raise an error anymore
        compatibility = await synthesizer._analyze_cwmai_compatibility(test_capability)
        assert isinstance(compatibility, dict), "Compatibility analysis should return a dict"
        print("✅ CapabilitySynthesizer string handling fixed successfully")
        return True
    except Exception as e:
        print(f"❌ CapabilitySynthesizer test failed: {e}")
        return False

async def test_redis_imports():
    """Test that Redis component imports are fixed."""
    print("\nTesting Redis component imports...")
    try:
        # Test imports don't raise errors
        from redis_intelligence_hub import RedisIntelligenceHub
        from redis_event_analytics import RedisEventAnalytics
        from redis_distributed_workflows import RedisWorkflowEngine
        from redis_streams_integration import RedisIntelligenceIntegrator
        
        print("✅ Redis component imports fixed successfully")
        return True
    except Exception as e:
        print(f"❌ Redis imports test failed: {e}")
        return False

async def test_workflow_initialize():
    """Test that RedisWorkflowEngine.initialize() works without arguments."""
    print("\nTesting RedisWorkflowEngine.initialize fix...")
    try:
        from redis_distributed_workflows import RedisWorkflowEngine
        
        # Create engine
        engine = RedisWorkflowEngine()
        
        # Check initialize method signature
        import inspect
        sig = inspect.signature(engine.initialize)
        # Should have only 'self' parameter
        params = list(sig.parameters.keys())
        assert len(params) == 0 or (len(params) == 1 and params[0] == 'self'), \
            f"initialize() should take no arguments, but has parameters: {params}"
        
        print("✅ RedisWorkflowEngine.initialize() signature fixed successfully")
        return True
    except Exception as e:
        print(f"❌ RedisWorkflowEngine test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("=== Testing All Fixes ===\n")
    
    results = []
    
    # Run each test
    results.append(await test_redis_client_sismember())
    results.append(await test_research_knowledge_store())
    results.append(await test_capability_synthesizer())
    results.append(await test_redis_imports())
    results.append(await test_workflow_initialize())
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All fixes verified successfully!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)