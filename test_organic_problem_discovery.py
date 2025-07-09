#!/usr/bin/env python3
"""
Test script to verify the updated organic problem discovery in gemini_deep_research.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain


async def test_organic_problem_discovery():
    """Test the new organic problem discovery approach."""
    print("=" * 80)
    print("Testing Organic Problem Discovery in Gemini Deep Research")
    print("=" * 80)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    researcher = GeminiDeepResearch(ai_brain)
    
    print("\n1. Testing problem discovery...")
    
    # Test the _discover_real_world_problems method
    try:
        problems = await researcher._discover_real_world_problems()
        
        if problems:
            print(f"\n✅ Successfully discovered {len(problems)} real-world problems!")
            print("\nSample problems discovered:")
            
            for i, problem in enumerate(problems[:3], 1):
                print(f"\n{i}. Problem: {problem.get('problem_statement', 'N/A')}")
                print(f"   Affected: {problem.get('affected_audience', 'N/A')}")
                print(f"   Pain: {problem.get('current_pain', 'N/A')}")
                print(f"   Severity: {problem.get('severity_score', 0)}/10")
                print(f"   Web Solvable: {problem.get('web_solvable', False)}")
                
        else:
            print("\n❌ No problems discovered")
            
    except Exception as e:
        print(f"\n❌ Error during problem discovery: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not problems:
        print("\nNo problems to validate, exiting test")
        return
        
    print("\n2. Testing problem validation...")
    
    # Test the _validate_problems_with_search method
    try:
        validated = await researcher._validate_problems_with_search(problems)
        
        if validated:
            print(f"\n✅ Successfully validated {len(validated)} problems!")
            print("\nSample validated problems:")
            
            for i, problem in enumerate(validated[:2], 1):
                print(f"\n{i}. Problem: {problem.get('problem_statement', 'N/A')}")
                print(f"   Validation Score: {problem.get('validation_score', 0)}/10")
                print(f"   Existing Solutions: {len(problem.get('existing_solutions', []))}")
                print(f"   Why They Fail: {problem.get('why_they_fail', 'N/A')[:100]}...")
                print(f"   Unique Angle: {problem.get('unique_angle', 'N/A')[:100]}...")
                
        else:
            print("\n❌ No problems validated")
            
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3. Testing full research flow...")
    
    # Test the complete flow
    try:
        opportunities = await researcher.conduct_deep_market_research()
        
        if opportunities:
            print(f"\n✅ Successfully generated {len(opportunities)} market opportunities!")
            
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{i}. {opp.problem_description}")
                print(f"   🎯 Target: {opp.affected_audience}")
                print(f"   💰 Revenue: {opp.potential_revenue}")
                print(f"   🏆 Competition: {opp.competition_level}")
                print(f"   📊 Market Size: {opp.market_size}")
                
        else:
            print("\n❌ No opportunities generated")
            
    except Exception as e:
        print(f"\n❌ Error during full research: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Set up environment variables if needed
    if os.path.exists('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    
    # Run the test
    asyncio.run(test_organic_problem_discovery())