#!/usr/bin/env python3
"""
Test the duplicate filtering in market research engine
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_research_engine import MarketResearchEngine, ProjectOpportunity
from scripts.ai_brain import IntelligentAIBrain

async def test_duplicate_filtering():
    """Test the _filter_duplicate_opportunities method."""
    
    print("🧪 Testing Market Research Duplicate Filtering")
    print("=" * 60)
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    research_engine = MarketResearchEngine(ai_brain)
    
    # Create mock existing portfolio
    existing_portfolio = {
        'ai-text-summarizer': {
            'description': 'AI-powered tool to summarize text, audio, and video',
            'tech_stack': ['Laravel', 'React'],
            'target_market': 'students and professionals'
        },
        'subtrack-ai': {
            'description': 'AI-powered subscription manager',
            'tech_stack': ['Laravel', 'React'],  
            'target_market': 'consumers with subscriptions'
        }
    }
    
    # Create test opportunities (some duplicates, some unique)
    test_opportunities = [
        ProjectOpportunity(
            title="ContentDigest Pro",
            description="AI summarization tool for busy professionals",
            problem_statement="Information overload - people need quick summaries",
            solution_approach="Use AI to summarize content",
            tech_stack=['React', 'Node.js'],
            target_market="Busy professionals",
            unique_value_proposition="Quick AI summaries",
            estimated_complexity="medium",
            market_demand=0.8,
            innovation_score=0.3,
            monetization_model="Subscription",
            revenue_potential="$10K MRR"
        ),
        ProjectOpportunity(
            title="FarmConnect Local", 
            description="Connect local farmers with consumers",
            problem_statement="Hard to find local produce",
            solution_approach="Marketplace for local farms",
            tech_stack=['Laravel', 'React'],
            target_market="Health-conscious consumers",
            unique_value_proposition="Fresh local produce",
            estimated_complexity="high",
            market_demand=0.7,
            innovation_score=0.8,
            monetization_model="Commission",
            revenue_potential="$20K MRR"
        ),
        ProjectOpportunity(
            title="BillTracker AI",
            description="Track and optimize subscriptions with AI",
            problem_statement="People lose money on forgotten subscriptions", 
            solution_approach="AI-powered subscription tracking",
            tech_stack=['Python', 'React'],
            target_market="Individual consumers",
            unique_value_proposition="Save money on subscriptions",
            estimated_complexity="medium",
            market_demand=0.9,
            innovation_score=0.4,
            monetization_model="Freemium",
            revenue_potential="$15K MRR"
        )
    ]
    
    print(f"\n📊 Testing with:")
    print(f"- Existing portfolio: {len(existing_portfolio)} projects")
    print(f"- New opportunities: {len(test_opportunities)} opportunities")
    
    print("\n📋 Test Opportunities:")
    for i, opp in enumerate(test_opportunities):
        print(f"\n{i+1}. {opp.title}")
        print(f"   Description: {opp.description}")
        print(f"   Target: {opp.target_market}")
    
    print("\n\n🔍 Running duplicate filter...")
    
    try:
        # Test the filter
        filtered = await research_engine._filter_duplicate_opportunities(
            test_opportunities,
            existing_portfolio
        )
        
        print(f"\n✅ Filter Results:")
        print(f"- Original opportunities: {len(test_opportunities)}")
        print(f"- Filtered opportunities: {len(filtered)}")
        print(f"- Removed as duplicates: {len(test_opportunities) - len(filtered)}")
        
        if filtered:
            print("\n📋 Kept Opportunities:")
            for opp in filtered:
                print(f"- {opp.title}: {opp.description}")
        
        # Show which ones were filtered out
        filtered_titles = {opp.title for opp in filtered}
        removed = [opp for opp in test_opportunities if opp.title not in filtered_titles]
        
        if removed:
            print("\n❌ Removed as Duplicates:")
            for opp in removed:
                print(f"- {opp.title}: {opp.description}")
                
    except Exception as e:
        print(f"\n❌ Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        
    # Test the condition check
    print("\n\n🔍 Testing the condition check...")
    print(f"hasattr(research_engine, 'duplicate_detector'): {hasattr(research_engine, 'duplicate_detector')}")
    print(f"existing_portfolio: {bool(existing_portfolio)}")
    print(f"Condition result: {hasattr(research_engine, 'duplicate_detector') or existing_portfolio}")
    
    # Test with empty portfolio
    print("\n\n🔍 Testing with empty portfolio...")
    empty_result = await research_engine._filter_duplicate_opportunities(
        test_opportunities[:1],
        {}
    )
    print(f"With empty portfolio, kept {len(empty_result)} of 1 opportunity")


def main():
    """Run the test."""
    try:
        asyncio.run(test_duplicate_filtering())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()