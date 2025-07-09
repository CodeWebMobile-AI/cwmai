#!/usr/bin/env python3
"""
Quick test to verify the final ranking schema works without running the full pipeline.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearch

async def test_final_ranking_schema():
    """Test just the final ranking schema with mock data."""
    
    print("Testing Final Ranking Schema Only")
    print("=" * 60)
    
    # Initialize the research module
    research = GeminiDeepResearch()
    
    # Create mock validated opportunities
    mock_opportunities = [{
        'opportunity_score': 85,
        'problem_description': 'Healthcare workers struggle with shift scheduling',
        'affected_audience': 'Healthcare professionals',
        'solution_name': 'ShiftSync AI',
        'key_features': ['AI scheduling', 'Mobile app', 'Team coordination'],
        'revenue_potential': '$5000-10000/month',
        'market_size': '$2B',
        'competition_level': 'medium',
        'technical_score': 8,
        'market_score': 8,
        'economics_score': 8,
        'mvp_score': 8
    }]
    
    # Mock portfolio
    mock_portfolio = []
    
    print("Testing _filter_and_rank with simplified schema...")
    
    try:
        # Call the method directly
        result = await research._filter_and_rank(
            opportunities=mock_opportunities,
            count=1,
            existing_portfolio=mock_portfolio
        )
        
        print(f"\n✅ Schema test passed! Returned {len(result)} opportunities")
        
        if result:
            print("\nFirst opportunity structure:")
            opp = result[0]
            print(f"- Type: {type(opp)}")
            print(f"- Has problem_description: {'problem_description' in opp.__dict__ if hasattr(opp, '__dict__') else 'N/A'}")
            print(f"- Has key_features: {hasattr(opp, 'key_features')}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        if "too many states" in str(e) or "INVALID_ARGUMENT" in str(e):
            print("\n🔴 Schema is still too complex!")
            print("Error details:", str(e))

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat ../.env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_final_ranking_schema())