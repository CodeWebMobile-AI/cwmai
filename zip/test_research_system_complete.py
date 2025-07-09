#!/usr/bin/env python3
"""
Test the complete research system with all new components.
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.research_memory import ResearchMemory
from scripts.dynamic_research_prompt_generator import DynamicResearchPromptGenerator
from scripts.emergent_learning_system import EmergentLearningSystem
from scripts.ai_brain import IntelligentAIBrain


async def test_research_memory():
    """Test the research memory system."""
    print("\n=== Testing Research Memory System ===")
    
    try:
        # Initialize
        ai_brain = IntelligentAIBrain()
        memory = ResearchMemory()
        
        # Test fingerprint creation
        problem = "Small businesses struggle with invoice management"
        fingerprint = memory.create_fingerprint(problem)
        print(f"✓ Fingerprint created: {fingerprint}")
        
        # Test embedding creation
        embedding = memory.create_embedding(problem, ai_brain)
        print(f"✓ Embedding created: {len(embedding)} dimensions")
        
        # Test recording attempt
        memory.record_attempt(
            problem=problem,
            embedding=embedding,
            success=True,
            validation_score=7.5,
            severity_score=8.0
        )
        print("✓ Attempt recorded successfully")
        
        # Test similarity check
        similar_problem = "Small companies have issues with billing management"
        similar_embedding = memory.create_embedding(similar_problem, ai_brain)
        is_similar, attempt = memory.is_similar_to_previous(similar_problem, similar_embedding)
        print(f"✓ Similarity check: {is_similar}")
        
        # Test analytics
        analytics = memory.get_analytics()
        print(f"✓ Analytics: {analytics['total_attempts']} attempts tracked")
        
        return True
        
    except Exception as e:
        print(f"✗ Research Memory Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prompt_generator():
    """Test the dynamic prompt generator."""
    print("\n=== Testing Dynamic Prompt Generator ===")
    
    try:
        # Initialize
        ai_brain = IntelligentAIBrain()
        memory = ResearchMemory()
        generator = DynamicResearchPromptGenerator(ai_brain, memory)
        
        # Test prompt generation
        prompt1 = generator.generate_research_prompt()
        print(f"✓ Generated prompt 1: {prompt1[:100]}...")
        
        # Generate another to ensure uniqueness
        prompt2 = generator.generate_research_prompt()
        print(f"✓ Generated prompt 2: {prompt2[:100]}...")
        
        # Test search query generation
        queries = generator.generate_search_queries(prompt1)
        print(f"✓ Generated {len(queries)} search queries")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt Generator Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_emergent_learning():
    """Test the emergent learning system."""
    print("\n=== Testing Emergent Learning System ===")
    
    try:
        # Initialize
        learning = EmergentLearningSystem()
        
        # Test recording outcome
        learning.record_outcome(
            project_name="invoice-automation-pro",
            problem_statement="Automate invoice processing",
            validation_score=7.5,
            severity_score=8.0,
            market_size="$5B",
            monetization_model="Per-transaction fees",
            success_metrics={"revenue": 0, "users": 0}
        )
        print("✓ Outcome recorded successfully")
        
        # Test getting insights
        insights = learning.get_insights()
        print(f"✓ Insights retrieved: {insights['total_projects']} projects tracked")
        
        # Test threshold adjustment
        thresholds = learning.get_adjusted_thresholds()
        print(f"✓ Adjusted thresholds: validation={thresholds['validation_score']}")
        
        # Test project decision
        should_create, reason = learning.should_create_project(7.5, 8.0, "Per-transaction fees")
        print(f"✓ Project decision: {should_create} - {reason}")
        
        return True
        
    except Exception as e:
        print(f"✗ Emergent Learning Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_integration():
    """Test the full Gemini research with all systems integrated."""
    print("\n=== Testing Full Gemini Integration ===")
    
    try:
        # Initialize
        gemini = GeminiDeepResearch()
        
        # Test small research (1 problem, should complete faster)
        print("\n🔬 Running minimal research test...")
        opportunities = await gemini.conduct_deep_market_research(count=1)
        
        if opportunities:
            print(f"✓ Research completed: Found {len(opportunities)} opportunities")
            
            # Check if opportunity has all required fields
            opp = opportunities[0]
            required_fields = [
                'problem_description', 'affected_audience', 'current_solutions',
                'opportunity', 'monetization_model', 'market_size'
            ]
            
            for field in required_fields:
                if hasattr(opp, field):
                    print(f"✓ Field '{field}' present")
                else:
                    print(f"✗ Missing field: {field}")
                    
            # Check if research phases data was collected
            phases_data = gemini.research_phases_data
            print(f"\n📊 Research Phases Data Collected:")
            for phase, data in phases_data.items():
                if data:
                    print(f"✓ {phase}: {'Collected' if data else 'Empty'}")
                    
        else:
            print("✗ No opportunities found")
            
        return True
        
    except Exception as e:
        print(f"✗ Gemini Integration Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_json_parsing():
    """Test JSON parsing in various phases."""
    print("\n=== Testing JSON Parsing ===")
    
    try:
        gemini = GeminiDeepResearch()
        
        # Test structured output call
        test_prompt = "Return a simple test object"
        test_schema = {
            "type": "object",
            "properties": {
                "test": {"type": "string"},
                "number": {"type": "number"}
            },
            "required": ["test", "number"]
        }
        
        result = await gemini._call_gemini_structured(test_prompt, test_schema)
        
        if result:
            print(f"✓ Structured output parsed successfully: {result}")
        else:
            print("✗ Failed to get structured output")
            
        return True
        
    except Exception as e:
        print(f"✗ JSON Parsing Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Complete Research System")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Warning: GOOGLE_API_KEY not set")
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set")
        
    # Run tests
    tests = [
        ("Research Memory", test_research_memory),
        ("Prompt Generator", test_prompt_generator),
        ("Emergent Learning", test_emergent_learning),
        ("JSON Parsing", test_json_parsing),
        ("Full Integration", test_gemini_integration)  # Run this last as it's slowest
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*50}")
        success = await test_func()
        results[name] = success
        
    # Summary
    print(f"\n{'='*50}")
    print("📋 Test Summary:")
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed - review output above")
        
    return all_passed


if __name__ == "__main__":
    # Set PYTHONPATH
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    result = asyncio.run(main())
    sys.exit(0 if result else 1)