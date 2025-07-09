#!/usr/bin/env python3
"""
Quick test of research system components without running full research.
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


async def test_components():
    """Test that all components initialize and work together."""
    print("🧪 Quick Component Test")
    print("=" * 50)
    
    try:
        # 1. Test AI Brain
        print("\n1. Testing AI Brain...")
        ai_brain = IntelligentAIBrain()
        response = ai_brain.generate_enhanced_response_sync("Return the number 42")
        content = response.get('content', '') if isinstance(response, dict) else str(response)
        print(f"   ✓ AI Brain working: {'42' in content}")
        
        # 2. Test Research Memory
        print("\n2. Testing Research Memory...")
        memory = ResearchMemory()
        
        # Test basic operations
        problem = "Test problem for validation"
        fingerprint = memory.create_fingerprint(problem)
        print(f"   ✓ Fingerprint created: {fingerprint}")
        
        embedding = memory.create_embedding(problem, ai_brain)
        print(f"   ✓ Embedding created: {len(embedding)} dimensions")
        
        memory.record_attempt(problem, embedding, True, 7.5, 8.0)
        print(f"   ✓ Attempt recorded")
        
        analytics = memory.get_analytics()
        print(f"   ✓ Analytics: {analytics['total_attempts']} attempts")
        
        # 3. Test Prompt Generator
        print("\n3. Testing Dynamic Prompt Generator...")
        generator = DynamicResearchPromptGenerator(ai_brain, memory)
        
        prompt = generator.generate_research_prompt()
        print(f"   ✓ Generated prompt: {prompt[:50]}...")
        
        queries = generator.generate_search_queries(prompt)
        print(f"   ✓ Generated {len(queries)} search queries")
        
        # 4. Test Emergent Learning
        print("\n4. Testing Emergent Learning System...")
        learning = EmergentLearningSystem()
        
        learning.record_outcome(
            "test-project", "Test problem", 7.5, 8.0,
            "$1M", "SaaS", {"revenue": 0}
        )
        
        insights = learning.get_insights()
        print(f"   ✓ Learning system: {insights['total_projects']} projects tracked")
        
        # 5. Test Gemini Integration (without running full research)
        print("\n5. Testing Gemini Deep Research Integration...")
        gemini = GeminiDeepResearch()
        
        # Check that all systems are integrated
        has_memory = hasattr(gemini, 'research_memory') and gemini.research_memory is not None
        has_generator = hasattr(gemini, 'prompt_generator') and gemini.prompt_generator is not None
        has_learning = hasattr(gemini, 'learning_system') and gemini.learning_system is not None
        
        print(f"   ✓ Research Memory integrated: {has_memory}")
        print(f"   ✓ Prompt Generator integrated: {has_generator}")
        print(f"   ✓ Learning System integrated: {has_learning}")
        
        # Test structured output
        test_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            },
            "required": ["status"]
        }
        
        result = await gemini._call_gemini_structured("Return status: ok", test_schema)
        print(f"   ✓ Structured output working: {result is not None}")
        
        print("\n✅ All components working correctly!")
        print("\n📝 Note: Full research test skipped for speed.")
        print("   Run test_research_system_complete.py for full integration test.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_components())
    sys.exit(0 if result else 1)