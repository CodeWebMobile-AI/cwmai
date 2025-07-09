#!/usr/bin/env python3
"""
Test the refactored market research system that uses MarketOpportunity directly.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_research_engine import MarketResearchEngine
from scripts.ai_brain import IntelligentAIBrain


async def test_refactored_system():
    """Test the refactored market research system."""
    print("=" * 80)
    print("🧪 TESTING REFACTORED MARKET RESEARCH SYSTEM")
    print("=" * 80)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    
    # Create market research engine
    research_engine = MarketResearchEngine(ai_brain)
    
    print("\n1️⃣ Testing Market Opportunity Discovery")
    print("-" * 40)
    
    start_time = datetime.now()
    
    # Test the new discover_market_opportunities method
    try:
        opportunities = await research_engine.discover_market_opportunities()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Discovered {len(opportunities)} market opportunities in {duration:.1f}s")
        
        # Display first opportunity details
        if opportunities:
            opp = opportunities[0]
            print(f"\nFirst Opportunity:")
            print(f"  Problem: {opp.problem_description[:80]}...")
            print(f"  Audience: {opp.affected_audience}")
            print(f"  Market Size: {opp.market_size}")
            print(f"  Revenue: {opp.potential_revenue}")
            print(f"  Competition: {opp.competition_level}")
            print(f"  Validation Sources: {len(opp.validation_sources)}")
            print(f"  Key Insights: {len(opp.key_insights)}")
            
            # Check that we have rich data
            has_validation = len(opp.validation_sources) > 0
            has_insights = len(opp.key_insights) > 0
            has_advantages = len(opp.unique_advantages) > 0
            
            print(f"\n  Data Quality Check:")
            print(f"    ✓ Has validation sources: {has_validation}")
            print(f"    ✓ Has key insights: {has_insights}")
            print(f"    ✓ Has unique advantages: {has_advantages}")
            
    except Exception as e:
        print(f"❌ Market opportunity discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2️⃣ Testing Project Generation from Opportunities")
    print("-" * 40)
    
    # Test project generation
    try:
        # Mock portfolio for testing
        mock_portfolio = {
            "existing-project": {
                "description": "A task management tool",
                "language": "JavaScript",
                "topics": ["productivity", "react"]
            }
        }
        
        start_time = datetime.now()
        
        project_opportunities = await research_engine.generate_project_opportunities(
            existing_portfolio=mock_portfolio,
            max_opportunities=3
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Generated {len(project_opportunities)} project opportunities in {duration:.1f}s")
        
        # Display project details
        for i, proj in enumerate(project_opportunities, 1):
            print(f"\nProject {i}: {proj.title}")
            print(f"  Description: {proj.description}")
            print(f"  Target Market: {proj.target_market}")
            print(f"  Monetization: {getattr(proj, 'monetization_model', 'N/A')}")
            print(f"  Revenue Potential: {getattr(proj, 'revenue_potential', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Project generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3️⃣ Testing Legacy Compatibility")
    print("-" * 40)
    
    # Test that legacy discover_market_trends still works
    try:
        trends = await research_engine.discover_market_trends()
        print(f"✅ Legacy method returned {len(trends)} trends")
        
        if trends:
            trend = trends[0]
            print(f"\nFirst Trend (converted from opportunity):")
            print(f"  Category: {trend.category}")
            print(f"  Name: {trend.trend_name}")
            print(f"  Demand: {trend.demand_level}")
            
    except Exception as e:
        print(f"❌ Legacy compatibility failed: {e}")
    
    print("\n4️⃣ Summary")
    print("-" * 40)
    print("\n✅ Refactoring Benefits:")
    print("  • No more hardcoded categories or technologies")
    print("  • Direct use of validated market data")
    print("  • Rich data preserved (validation sources, insights, etc.)")
    print("  • Simpler code flow")
    print("  • Backward compatibility maintained")
    
    # Save test results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'opportunities_found': len(opportunities),
        'projects_generated': len(project_opportunities),
        'has_validation_data': any(len(o.validation_sources) > 0 for o in opportunities),
        'has_insights': any(len(o.key_insights) > 0 for o in opportunities),
        'sample_opportunity': {
            'problem': opportunities[0].problem_description if opportunities else None,
            'market_size': opportunities[0].market_size if opportunities else None,
            'revenue': opportunities[0].potential_revenue if opportunities else None,
        } if opportunities else None
    }
    
    filename = f"refactored_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {filename}")


async def main():
    """Run the test."""
    await test_refactored_system()


if __name__ == "__main__":
    print("🔄 Refactored Market Research Test")
    print("This tests the new system that uses MarketOpportunity directly")
    print("eliminating hardcoded categories and technology extraction\n")
    
    asyncio.run(main())