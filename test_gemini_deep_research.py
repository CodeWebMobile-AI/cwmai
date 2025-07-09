#!/usr/bin/env python3
"""
Test script for Gemini Deep Research integration.

This demonstrates how the system now uses Gemini with Google Search grounding
to conduct comprehensive market research across 25 sectors.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain
from scripts.http_ai_client import HttpAIClient
from scripts.ai_response_enhancer import AIResponseEnhancer


async def test_deep_research():
    """Test the Gemini deep research functionality."""
    print("=" * 80)
    print("🚀 Testing Gemini Deep Research with Google Search Grounding")
    print("=" * 80)
    
    # Initialize AI components
    ai_client = HttpAIClient()
    response_enhancer = AIResponseEnhancer(ai_client)
    ai_brain = IntelligentAIBrain(ai_client, response_enhancer)
    
    # Initialize deep research
    researcher = GeminiDeepResearch(ai_brain)
    
    print(f"\n📊 Analyzing {len(researcher.DEFAULT_FOCUS_AREAS)} market sectors:")
    for i, area in enumerate(researcher.DEFAULT_FOCUS_AREAS[:10], 1):
        print(f"  {i}. {area}")
    print(f"  ... and {len(researcher.DEFAULT_FOCUS_AREAS) - 10} more sectors")
    
    print("\n🔬 Starting comprehensive market research...")
    print("This will use Gemini's Google Search to find real market data...\n")
    
    start_time = datetime.now()
    
    # Conduct research
    opportunities = await researcher.conduct_deep_market_research()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Research completed in {duration:.1f} seconds")
    print(f"📋 Found {len(opportunities)} validated opportunities\n")
    
    # Display results
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp.problem_description[:80]}...")
        print(f"   🎯 Target: {opp.affected_audience}")
        print(f"   💰 Revenue Potential: {opp.potential_revenue}")
        print(f"   🏆 Competition: {opp.competition_level}")
        print(f"   📊 Market Size: {opp.market_size}")
        print(f"   🔗 Validated by {len(opp.validation_sources)} sources")
        print()
    
    # Show detailed view of top opportunity
    if opportunities:
        print("\n" + "=" * 80)
        print("📋 DETAILED VIEW - TOP OPPORTUNITY")
        print("=" * 80)
        
        top = opportunities[0]
        print(f"\n🎯 Problem: {top.problem_description}")
        print(f"\n👥 Affected Audience: {top.affected_audience}")
        print(f"\n💡 Opportunity: {top.opportunity}")
        print(f"\n💰 Monetization Model: {top.monetization_model}")
        print(f"\n📊 Market Size: {top.market_size}")
        
        print(f"\n🔑 Key Insights:")
        for insight in top.key_insights:
            print(f"  • {insight}")
        
        print(f"\n✨ Unique Advantages:")
        for advantage in top.unique_advantages:
            print(f"  • {advantage}")
        
        print(f"\n🔗 Validation Sources:")
        for i, source in enumerate(top.validation_sources[:5], 1):
            print(f"  {i}. {source}")
        
        if len(top.validation_sources) > 5:
            print(f"  ... and {len(top.validation_sources) - 5} more sources")
    
    # Save results
    output_file = f"gemini_research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration,
        'opportunities_found': len(opportunities),
        'sectors_analyzed': len(researcher.DEFAULT_FOCUS_AREAS),
        'opportunities': [
            {
                'problem_description': opp.problem_description,
                'affected_audience': opp.affected_audience,
                'potential_revenue': opp.potential_revenue,
                'market_size': opp.market_size,
                'competition_level': opp.competition_level,
                'validation_sources': opp.validation_sources,
                'key_insights': opp.key_insights,
                'unique_advantages': opp.unique_advantages
            }
            for opp in opportunities
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


async def test_project_creation_with_research():
    """Test project creation using deep research."""
    print("\n" + "=" * 80)
    print("🏗️  Testing Project Creation with Deep Research")
    print("=" * 80)
    
    from scripts.project_creator import ProjectCreator
    from scripts.ai_brain import IntelligentAIBrain
    from scripts.http_ai_client import HttpAIClient
    from scripts.ai_response_enhancer import AIResponseEnhancer
    
    # Initialize components
    ai_client = HttpAIClient()
    response_enhancer = AIResponseEnhancer(ai_client)
    ai_brain = IntelligentAIBrain(ai_client, response_enhancer)
    
    # Create project creator in local mode for testing
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode='local')
    
    # Test task
    test_task = {
        'title': 'Create innovative project from deep research',
        'type': 'NEW_PROJECT',
        'created_at': datetime.now().isoformat()
    }
    
    print("\n🔬 Project creator will now use Gemini Deep Research...")
    
    # Generate project details (this will trigger deep research)
    project_details = await creator._generate_project_details(test_task)
    
    print("\n📋 Generated Project Details:")
    print(f"Name: {project_details.get('name', 'N/A')}")
    print(f"Description: {project_details.get('description', 'N/A')}")
    print(f"Problem: {project_details.get('problem_statement', 'N/A')}")
    print(f"Target Audience: {project_details.get('target_audience', 'N/A')}")
    print(f"Monetization: {project_details.get('monetization_strategy', 'N/A')}")
    
    if 'initial_features' in project_details:
        print("\nInitial Features:")
        for i, feature in enumerate(project_details['initial_features'][:5], 1):
            print(f"  {i}. {feature}")


async def main():
    """Run all tests."""
    try:
        # Test 1: Deep Research
        await test_deep_research()
        
        # Test 2: Project Creation with Research
        print("\n\nWould you like to test project creation with deep research? (y/n): ", end='')
        if input().lower() == 'y':
            await test_project_creation_with_research()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Gemini Deep Research Test Suite")
    print("This will demonstrate market research across 25 sectors using Google Search")
    print("\nNOTE: This requires proper API keys in .env.local")
    print("Press Enter to continue...")
    input()
    
    asyncio.run(main())