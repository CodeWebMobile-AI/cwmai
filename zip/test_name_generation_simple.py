#!/usr/bin/env python3
"""
Simple test for creative name generation in final selection
"""

import asyncio
import logging
from scripts.gemini_deep_research import GeminiDeepResearchOptimized

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_name_generation():
    """Test creative name generation in final selection phase."""
    research = GeminiDeepResearchOptimized()
    
    # Create test opportunities
    test_opportunities = [
        {
            'problem_description': 'Small businesses struggle to track inventory across multiple locations',
            'affected_audience': 'Small Business Owners',
            'market_size': '$5B',
            'revenue_model': 'SaaS subscription',
            'validation_score': 8.5,
            'competition_score': 7.0,
            'final_score': 8.0
        },
        {
            'problem_description': 'Freelancers have difficulty managing invoices and payments',
            'affected_audience': 'Freelancers',
            'market_size': '$2B',
            'revenue_model': 'Transaction fees',
            'validation_score': 8.0,
            'competition_score': 6.5,
            'final_score': 7.5
        }
    ]
    
    # Test the final selection with name generation
    print("Testing creative name generation...")
    opportunities = await research._phase_5_final_selection_optimized(
        test_opportunities,
        existing_portfolio={},
        target_count=2
    )
    
    print(f"\nGenerated {len(opportunities)} opportunities:")
    for i, opp in enumerate(opportunities):
        print(f"\n{i+1}. Problem: {opp.problem_description[:60]}...")
        print(f"   Project Name: '{opp.project_name}'")
        print(f"   Has Creative Name: {'✅' if opp.project_name and len(opp.project_name) > 0 else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_name_generation())