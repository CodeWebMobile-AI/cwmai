"""
Test just the market research component.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from scripts.ai_brain import IntelligentAIBrain
from scripts.market_research_engine import MarketResearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_market_research():
    """Test the market research engine."""
    print("\n🔬 Testing Market Research Engine...")
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create market research engine
    market_research = MarketResearchEngine(ai_brain)
    
    # Discover market trends
    print("\n📊 Discovering market trends...")
    trends = await market_research.discover_market_trends()
    
    print(f"\nDiscovered {len(trends)} market trends:")
    for trend in trends[:3]:  # Show first 3
        print(f"\n- {trend.trend_name}")
        print(f"  Category: {trend.category}")
        print(f"  Demand: {trend.demand_level}")
        print(f"  Opportunity Score: {trend.opportunity_score:.2f}")
        print(f"  Technologies: {', '.join(trend.technologies[:3])}")
        print(f"  Problem Space: {trend.problem_space[:100]}...")
    
    # Test project opportunity generation
    print("\n\n💡 Generating project opportunities...")
    
    # Mock portfolio for testing
    mock_portfolio = {
        "test-cms": {
            "name": "test-cms",
            "description": "A content management system",
            "language": "JavaScript",
            "topics": ["cms", "nodejs", "react"],
            "recent_activity": {"recent_commits": 10}
        },
        "test-api": {
            "name": "test-api", 
            "description": "RESTful API service",
            "language": "Python",
            "topics": ["api", "fastapi", "postgresql"],
            "recent_activity": {"recent_commits": 5}
        }
    }
    
    opportunities = await market_research.generate_project_opportunities(
        mock_portfolio,
        max_opportunities=3
    )
    
    print(f"\nGenerated {len(opportunities)} project opportunities:")
    for opp in opportunities:
        print(f"\n📦 {opp.title}")
        print(f"   Problem: {opp.problem_statement[:100]}...")
        print(f"   Solution: {opp.solution_approach[:100]}...")
        tech_stack_preview = opp.tech_stack[:3] if isinstance(opp.tech_stack, list) else [str(opp.tech_stack)]
        print(f"   Tech Stack: {', '.join(tech_stack_preview)}")
        print(f"   Market Demand: {opp.market_demand:.2f}")
        print(f"   Innovation Score: {opp.innovation_score:.2f}")
        print(f"   Unique Value: {opp.unique_value_proposition[:100]}...")
        if hasattr(opp, 'monetization_model') and opp.monetization_model:
            print(f"   💰 Monetization: {opp.monetization_model}")
        if hasattr(opp, 'revenue_potential') and opp.revenue_potential:
            print(f"   💵 Revenue Potential: {opp.revenue_potential}")
    
    # Get research stats
    stats = market_research.get_research_stats()
    print(f"\n📈 Research Stats: {json.dumps(stats, indent=2)}")


async def main():
    """Run the test."""
    print("🚀 Testing Market Research Engine with Real AI")
    print("=" * 60)
    
    try:
        await test_market_research()
        print("\n\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())