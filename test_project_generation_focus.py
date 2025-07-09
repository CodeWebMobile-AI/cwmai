#!/usr/bin/env python3
"""
Test script to verify project generation focuses on real-world problems
instead of developer tools.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.ai_brain import IntelligentAIBrain
from scripts.http_ai_client import HTTPAIClient
from scripts.ai_response_enhancer import AIResponseEnhancer
from scripts.market_research_engine import MarketResearchEngine
from scripts.project_creator import ProjectCreator


async def test_project_generation():
    """Test the project generation to see what types of projects are created."""
    print("🧪 Testing Project Generation Focus")
    print("=" * 60)
    
    # Initialize AI components
    ai_brain = IntelligentAIBrain()
    
    # Initialize market research engine
    market_research = MarketResearchEngine(ai_brain)
    
    # Mock portfolio to test gap analysis
    mock_portfolio = {
        "existing-project-1": {
            "description": "A task management tool",
            "language": "JavaScript",
            "topics": ["productivity", "task-management"]
        },
        "existing-project-2": {
            "description": "A recipe sharing platform",
            "language": "Python",
            "topics": ["food", "social"]
        }
    }
    
    print("\n📊 Discovering Market Opportunities...")
    print("-" * 60)
    
    try:
        # Generate project opportunities
        opportunities = await market_research.generate_project_opportunities(
            existing_portfolio=mock_portfolio,
            max_opportunities=5
        )
        
        print(f"\n✅ Generated {len(opportunities)} project opportunities:\n")
        
        # Display each opportunity
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{'='*60}")
            print(f"Project {i}: {opp.title}")
            print(f"{'='*60}")
            print(f"  📝 Description: {opp.description}")
            print(f"  🎯 Target Market: {opp.target_market}")
            print(f"  💡 Problem: {opp.problem_statement}")
            print(f"  🔧 Solution: {opp.solution_approach}")
            print(f"  💰 Monetization: {opp.monetization_model}")
            print(f"  💵 Revenue Potential: {opp.revenue_potential}")
            print(f"  🚀 Unique Value: {opp.unique_value_proposition}")
            print(f"  📊 Market Demand: {opp.market_demand}")
            print(f"  🔨 Complexity: {opp.estimated_complexity}")
            print(f"  💻 Tech Stack: Laravel, React, MySQL (standard stack)")
        
        # Analyze the types of projects generated
        print(f"\n\n📈 Analysis of Generated Projects:")
        print("-" * 60)
        
        # Check for developer-focused projects
        dev_focused_count = 0
        real_world_count = 0
        
        dev_keywords = ['developer', 'coding', 'programming', 'api', 'devops', 
                       'github', 'code completion', 'ide', 'debug', 'compile']
        
        real_world_keywords = ['business', 'health', 'education', 'family', 
                              'financial', 'community', 'environment', 'wellness',
                              'productivity', 'management', 'social']
        
        for opp in opportunities:
            full_text = f"{opp.title} {opp.description} {opp.target_market}".lower()
            
            if any(keyword in full_text for keyword in dev_keywords):
                dev_focused_count += 1
                print(f"  ⚠️  {opp.title} - Appears developer-focused")
            elif any(keyword in full_text for keyword in real_world_keywords):
                real_world_count += 1
                print(f"  ✅ {opp.title} - Real-world problem focus")
            else:
                print(f"  ❓ {opp.title} - Unclear focus")
        
        print(f"\n📊 Summary:")
        print(f"  - Developer-focused projects: {dev_focused_count}")
        print(f"  - Real-world focused projects: {real_world_count}")
        print(f"  - Other/Unclear: {len(opportunities) - dev_focused_count - real_world_count}")
        
        # Save results for review
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "developer_focused": dev_focused_count,
            "real_world_focused": real_world_count,
            "opportunities": [
                {
                    "title": opp.title,
                    "description": opp.description,
                    "target_market": opp.target_market,
                    "problem_statement": opp.problem_statement,
                    "monetization_model": opp.monetization_model,
                    "revenue_potential": opp.revenue_potential
                }
                for opp in opportunities
            ]
        }
        
        with open("project_generation_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to project_generation_test_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting project generation focus test...")
    print("This will use the updated market research engine.\n")
    
    # Run the test
    asyncio.run(test_project_generation())