#!/usr/bin/env python3
"""
Test script to verify the schema fix handles large data properly.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain

async def test_with_large_data():
    """Test the schema with large amounts of data to ensure no errors."""
    
    print("Testing Schema Fix with Large Data")
    print("=" * 60)
    
    # Initialize components
    ai_brain = IntelligentAIBrain() 
    research = GeminiDeepResearch(ai_brain)
    
    # Create many opportunities with lots of data
    large_opportunities = []
    for i in range(20):  # Create 20 opportunities
        opp = {
            'solution_name': f'TestSolution{i}',
            'problem_description': 'A' * 500,  # Long description
            'affected_audience': 'B' * 200,
            'current_solutions': 'C' * 300,
            'opportunity': 'D' * 400,
            'monetization_model': 'E' * 200,
            'market_size': '$1B',
            'competition_level': 'medium',
            'validation_sources': [f'source{j}' for j in range(10)],
            'key_insights': [f'insight{j}' for j in range(10)],
            'implementation_difficulty': 'medium',
            'potential_revenue': '$10K/month',
            'unique_advantages': [f'advantage{j}' for j in range(10)],
            'search_evidence': {'summary': 'Evidence summary'},
            # Add all the phase data
            'user_research': {'data': 'X' * 1000},
            'revenue_validation': {'data': 'Y' * 1000},
            'technical_validation': {'data': 'Z' * 1000},
            'market_score': 8,
            'technical_score': 8,
            'economics_score': 8,
            'mvp_score': 8,
            'passive_income_score': 8,
            'automation_level': 'high'
        }
        large_opportunities.append(opp)
    
    print(f"Testing with {len(large_opportunities)} opportunities")
    print(f"Total data size: ~{sum(len(str(opp)) for opp in large_opportunities):,} chars")
    
    try:
        # Test the final ranking with large data
        result = await research._filter_and_rank(
            opportunities=large_opportunities,
            count=1,
            existing_portfolio={}
        )
        
        print(f"\n✅ SUCCESS! Schema handled large data without errors")
        print(f"Returned {len(result)} opportunities")
        
        if result:
            print(f"\nFirst opportunity: {result[0].problem_description[:50]}...")
        
    except Exception as e:
        if "too many states" in str(e) or "INVALID_ARGUMENT" in str(e):
            print(f"\n❌ SCHEMA ERROR STILL OCCURS: {e}")
            print("\nThe fix needs more work!")
        else:
            print(f"\n❌ Other error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_with_large_data())