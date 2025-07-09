#!/usr/bin/env python3
"""
Test script to verify Gemini Deep Research focuses on real-world problems
and uses Google Search grounding effectively.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.gemini_deep_research import GeminiDeepResearch, MarketOpportunity


async def test_gemini_deep_research():
    """Test Gemini Deep Research to see what types of opportunities it discovers."""
    print("🧪 Testing Gemini Deep Research Focus")
    print("=" * 60)
    
    # Initialize Gemini Deep Research
    researcher = GeminiDeepResearch()
    
    # Show the research areas
    print(f"\n📋 Research Areas ({len(researcher.DEFAULT_FOCUS_AREAS)} sectors):")
    print("-" * 60)
    for i, area in enumerate(researcher.DEFAULT_FOCUS_AREAS[:10], 1):
        print(f"  {i:2d}. {area}")
    print(f"  ... and {len(researcher.DEFAULT_FOCUS_AREAS) - 10} more sectors")
    
    # Mock portfolio to test gap analysis
    mock_portfolio = {
        "task-manager-pro": {
            "description": "A task management application",
            "language": "JavaScript",
            "topics": ["productivity", "tasks"]
        },
        "recipe-share": {
            "description": "Recipe sharing social platform",
            "language": "Python", 
            "topics": ["food", "social"]
        }
    }
    
    print("\n\n🔬 Starting Deep Market Research with Google Search Grounding...")
    print("-" * 60)
    print("This will take a few minutes as it performs comprehensive research...\n")
    
    try:
        # Conduct deep research
        opportunities = await researcher.conduct_deep_market_research(
            existing_portfolio=mock_portfolio
        )
        
        print(f"\n✅ Discovered {len(opportunities)} validated opportunities:\n")
        
        # Display each opportunity
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{'='*60}")
            print(f"Opportunity {i}")
            print(f"{'='*60}")
            print(f"  📝 Problem: {opp.problem_description[:150]}...")
            print(f"  🎯 Audience: {opp.affected_audience}")
            print(f"  💡 Solution: {opp.opportunity[:150]}...")
            print(f"  💰 Revenue: {opp.potential_revenue}")
            print(f"  💼 Model: {opp.monetization_model}")
            print(f"  📊 Market Size: {opp.market_size}")
            print(f"  🏆 Competition: {opp.competition_level}")
            print(f"  🔧 Difficulty: {opp.implementation_difficulty}")
            print(f"  🔗 Validation Sources: {len(opp.validation_sources)} sources")
            print(f"  ✨ Key Insights: {len(opp.key_insights)} insights")
            print(f"  🚀 Advantages: {', '.join(opp.unique_advantages[:2])}...")
        
        # Analyze the types of opportunities
        print(f"\n\n📈 Analysis of Discovered Opportunities:")
        print("-" * 60)
        
        # Check for developer-focused opportunities
        dev_focused_count = 0
        real_world_count = 0
        
        dev_keywords = ['developer', 'coding', 'programming', 'api', 'devops', 
                       'github', 'code completion', 'ide', 'debug', 'compile',
                       'software engineer', 'programmer']
        
        real_world_keywords = ['business', 'health', 'education', 'family', 
                              'financial', 'community', 'environment', 'wellness',
                              'elderly', 'children', 'social', 'agriculture',
                              'retail', 'hospitality', 'manufacturing']
        
        opportunity_analysis = []
        
        for opp in opportunities:
            full_text = f"{opp.problem_description} {opp.affected_audience} {opp.opportunity}".lower()
            
            is_dev_focused = any(keyword in full_text for keyword in dev_keywords)
            is_real_world = any(keyword in full_text for keyword in real_world_keywords)
            
            if is_dev_focused:
                dev_focused_count += 1
                focus_type = "Developer-focused"
                symbol = "⚠️"
            elif is_real_world:
                real_world_count += 1
                focus_type = "Real-world focus"
                symbol = "✅"
            else:
                focus_type = "Unclear focus"
                symbol = "❓"
            
            print(f"  {symbol} Opportunity {opportunities.index(opp) + 1}: {focus_type}")
            print(f"     Problem area: {opp.problem_description[:60]}...")
            
            opportunity_analysis.append({
                "problem": opp.problem_description,
                "audience": opp.affected_audience,
                "focus_type": focus_type,
                "revenue": opp.potential_revenue,
                "competition": opp.competition_level
            })
        
        print(f"\n📊 Summary:")
        print(f"  - Developer-focused opportunities: {dev_focused_count}")
        print(f"  - Real-world focused opportunities: {real_world_count}")
        print(f"  - Other/Unclear: {len(opportunities) - dev_focused_count - real_world_count}")
        
        # Check validation quality
        print(f"\n🔍 Validation Quality:")
        total_sources = sum(len(opp.validation_sources) for opp in opportunities)
        avg_sources = total_sources / len(opportunities) if opportunities else 0
        print(f"  - Total validation sources: {total_sources}")
        print(f"  - Average sources per opportunity: {avg_sources:.1f}")
        print(f"  - Opportunities with search evidence: {sum(1 for opp in opportunities if opp.search_evidence)}")
        
        # Save results for review
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "developer_focused": dev_focused_count,
            "real_world_focused": real_world_count,
            "research_areas_used": len(researcher.DEFAULT_FOCUS_AREAS),
            "validation_stats": {
                "total_sources": total_sources,
                "avg_sources_per_opportunity": avg_sources
            },
            "opportunities": [
                {
                    "problem": opp.problem_description,
                    "audience": opp.affected_audience,
                    "solution": opp.opportunity,
                    "revenue": opp.potential_revenue,
                    "market_size": opp.market_size,
                    "competition": opp.competition_level,
                    "monetization": opp.monetization_model,
                    "validation_sources_count": len(opp.validation_sources),
                    "key_insights": opp.key_insights[:3] if opp.key_insights else []
                }
                for opp in opportunities
            ],
            "analysis": opportunity_analysis
        }
        
        with open("gemini_deep_research_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to gemini_deep_research_test_results.json")
        
        # Show one detailed opportunity
        if opportunities:
            print(f"\n\n📋 Detailed View - Top Opportunity:")
            print("=" * 60)
            top = opportunities[0]
            print(f"Problem: {top.problem_description}")
            print(f"\nAffected Audience: {top.affected_audience}")
            print(f"\nCurrent Solutions: {top.current_solutions}")
            print(f"\nOpportunity: {top.opportunity}")
            print(f"\nMonetization Model: {top.monetization_model}")
            print(f"\nKey Insights:")
            for insight in top.key_insights[:3]:
                print(f"  • {insight}")
            print(f"\nUnique Advantages:")
            for advantage in top.unique_advantages[:3]:
                print(f"  • {advantage}")
            if top.validation_sources:
                print(f"\nValidation Sources ({len(top.validation_sources)} total):")
                for source in top.validation_sources[:3]:
                    print(f"  • {source}")
        
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Gemini Deep Research focus test...")
    print("This test will use Google Search grounding to find real market opportunities.\n")
    
    # Run the test
    asyncio.run(test_gemini_deep_research())