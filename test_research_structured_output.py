#!/usr/bin/env python3
"""
Test script to verify research engine structured output implementation.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager


async def test_research_structured_output():
    """Test the research engine with structured output."""
    print("=" * 80)
    print("Testing Research Engine Structured Output")
    print("=" * 80)
    
    # Initialize components
    state_manager = StateManager()
    system_state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(system_state, {})
    
    # Create a minimal research engine instance
    research_engine = ResearchEvolutionEngine(
        state_manager=state_manager,
        ai_brain=ai_brain
    )
    
    print("\n1. Testing single research execution with structured output...")
    
    # Create a test query
    test_query = {
        "query": "What are the best practices for implementing structured output in AI systems?",
        "provider": "gemini",
        "confidence": 0.8,
        "source": "test",
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "area": "ai_improvement",
            "topic": "structured output implementation"
        }
    }
    
    test_topic = {
        "topic": "Structured Output Implementation",
        "area": "ai_improvement",
        "priority": "high",
        "context": {
            "purpose": "Improve AI response parsing reliability",
            "current_issue": "Complex regex parsing failing on various response formats"
        }
    }
    
    # Execute single research
    try:
        result = await research_engine._execute_single_research(test_query, test_topic)
        
        if result:
            print("\n✅ Research execution successful!")
            print(f"\nResearch ID: {result['id']}")
            print(f"Quality Score: {result['quality_score']}")
            print(f"Structured Output: {result['metadata'].get('structured_output', False)}")
            
            # Check the content structure
            content = result.get('content', {})
            print("\n📊 Content Structure Analysis:")
            
            if isinstance(content, dict):
                print(f"  - Summary: {'✓' if content.get('summary') else '✗'}")
                print(f"  - Key Insights: {len(content.get('key_insights', []))} found")
                print(f"  - Recommendations: {len(content.get('recommendations', []))} found")
                print(f"  - Implementation Details: {len(content.get('implementation_details', []))} found")
                print(f"  - Success Factors: {len(content.get('success_factors', []))} found")
                print(f"  - Potential Challenges: {len(content.get('potential_challenges', []))} found")
                print(f"  - Metrics: {'✓' if content.get('metrics') else '✗'}")
                
                # Display sample content
                print("\n📝 Sample Content:")
                if content.get('summary'):
                    print(f"\nSummary: {content['summary'][:200]}...")
                
                if content.get('key_insights'):
                    print("\nKey Insights:")
                    for i, insight in enumerate(content['key_insights'][:3], 1):
                        print(f"  {i}. {insight.get('insight', 'N/A')} (Impact: {insight.get('impact', 'N/A')})")
                
                if content.get('recommendations'):
                    print("\nRecommendations:")
                    for i, rec in enumerate(content['recommendations'][:3], 1):
                        print(f"  {i}. {rec.get('recommendation', 'N/A')} (Priority: {rec.get('priority', 'N/A')})")
            else:
                print("  ⚠️ Content is not structured (fallback mode)")
                print(f"  Raw content type: {type(content)}")
        else:
            print("\n❌ Research execution failed - no result returned")
            
    except Exception as e:
        print(f"\n❌ Error during research execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing research quality scoring...")
    
    # Test quality scoring with different content types
    test_contents = [
        {
            "name": "Well-structured content",
            "content": {
                "summary": "This is a comprehensive summary of the research findings that provides clear insights.",
                "key_insights": [
                    {"insight": "Structured output improves reliability", "impact": "high", "category": "reliability"},
                    {"insight": "JSON schemas ensure consistency", "impact": "high", "category": "maintainability"},
                    {"insight": "Provider-specific handling needed", "impact": "medium", "category": "integration"}
                ],
                "recommendations": [
                    {"recommendation": "Implement JSON schema validation", "priority": "critical", "effort": "moderate"},
                    {"recommendation": "Add fallback parsing", "priority": "high", "effort": "minimal"}
                ],
                "implementation_details": [
                    {"task": "Create schema definitions", "approach": "Define TypeScript-like schemas"},
                    {"task": "Update AI client", "approach": "Add response_schema parameter"}
                ],
                "success_factors": ["Clear schema definitions", "Provider documentation"],
                "metrics": {
                    "current_state": "60% parsing failures",
                    "target_state": "< 5% parsing failures",
                    "measurement_approach": "Track parse errors per provider"
                }
            }
        },
        {
            "name": "Minimal structured content",
            "content": {
                "summary": "Brief summary",
                "key_insights": [],
                "recommendations": []
            }
        },
        {
            "name": "Non-structured content (fallback)",
            "content": "This is raw text content that would trigger fallback parsing mode."
        }
    ]
    
    for test_case in test_contents:
        score = research_engine._score_research_quality(test_case["content"])
        print(f"\n{test_case['name']}: Score = {score:.2f}")
    
    print("\n3. Checking research storage format...")
    
    # Check if research would be stored correctly
    if result:
        # Simulate storage (without actually storing)
        storage_preview = {
            "id": result["id"],
            "type": result["area"],
            "content": result["content"],  # This should now be structured
            "timestamp": result["timestamp"],
            "quality_score": result["quality_score"]
        }
        
        print("\n📦 Storage Preview:")
        print(json.dumps(storage_preview, indent=2, default=str)[:500] + "...")
        
        # Verify content is structured
        if isinstance(storage_preview["content"], dict) and "summary" in storage_preview["content"]:
            print("\n✅ Content will be stored as structured data (not raw text)")
        else:
            print("\n⚠️ Content appears to be in fallback mode")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Set up environment variables if needed
    if os.path.exists('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    
    # Run the test
    asyncio.run(test_research_structured_output())