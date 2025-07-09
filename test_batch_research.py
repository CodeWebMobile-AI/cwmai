#!/usr/bin/env python3
"""
Test the batch research approach with dynamic opportunity count.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_research_engine import MarketResearchEngine
from scripts.ai_brain import IntelligentAIBrain
from scripts.gemini_deep_research import GeminiDeepResearch


async def test_dynamic_research():
    """Test dynamic research based on worker count."""
    print("🧪 Testing Dynamic Batch Research")
    print("=" * 60)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    research_engine = MarketResearchEngine(ai_brain)
    
    # Test different scenarios
    scenarios = [
        {"workers": 1, "existing_projects": 0, "description": "Single worker, no projects"},
        {"workers": 6, "existing_projects": 4, "description": "6 workers, 4 existing projects"},
        {"workers": 10, "existing_projects": 2, "description": "10 workers, few projects"},
        {"workers": 8, "existing_projects": 15, "description": "8 workers, established portfolio"}
    ]
    
    for scenario in scenarios:
        print(f"\n\n📊 Scenario: {scenario['description']}")
        print(f"   Workers: {scenario['workers']}")
        print(f"   Existing Projects: {scenario['existing_projects']}")
        
        # Calculate opportunities needed
        if scenario['existing_projects'] < 5:
            # Early stage: aggressive
            base_opportunities = min(scenario['workers'], 10)
        else:
            # Established: moderate
            base_opportunities = min(scenario['workers'] // 2, 5)
            
        # Add 20% buffer
        opportunities_with_buffer = int(base_opportunities * 1.2)
        final_count = max(opportunities_with_buffer, 3)
        
        print(f"   Calculated Opportunities: {final_count}")
        print(f"   (Base: {base_opportunities}, With Buffer: {opportunities_with_buffer})")
        
        # Simulate portfolio
        mock_portfolio = {
            f"project-{i}": {
                "description": f"Existing project {i}",
                "language": "PHP"
            }
            for i in range(scenario['existing_projects'])
        }
        
        try:
            # Request opportunities
            print(f"\n   🔍 Researching {final_count} opportunities...")
            opportunities = await research_engine.generate_project_opportunities(
                mock_portfolio,
                max_opportunities=final_count
            )
            
            print(f"   ✅ Generated {len(opportunities)} project opportunities")
            
            # Show brief summary
            for i, opp in enumerate(opportunities[:3], 1):  # Show first 3
                print(f"\n   {i}. {opp.title}")
                print(f"      Problem: {opp.problem_statement[:60]}...")
                print(f"      Market Demand: {opp.market_demand:.1f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def test_direct_gemini_research():
    """Test Gemini research with different counts."""
    print("\n\n🧪 Testing Direct Gemini Research")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    test_counts = [3, 5, 8]
    
    for count in test_counts:
        print(f"\n📊 Testing with count={count}")
        print(f"   Requesting {count} opportunities from Gemini...")
        
        try:
            opportunities = await researcher.conduct_deep_market_research(count=count)
            
            print(f"   ✅ Received {len(opportunities)} opportunities")
            
            # Show problems discovered
            for i, opp in enumerate(opportunities, 1):
                print(f"\n   {i}. {opp.problem_description[:80]}...")
                print(f"      Audience: {opp.affected_audience}")
                print(f"      Competition: {opp.competition_level}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if "overloaded" in str(e).lower():
                print("   (Gemini is overloaded, this is expected during high usage)")


async def demonstrate_worker_distribution():
    """Demonstrate how opportunities are distributed to workers."""
    print("\n\n🧪 Demonstrating Worker Distribution")
    print("=" * 60)
    
    # Simulate 6 workers and 8 opportunities
    workers = [f"worker-{i}" for i in range(6)]
    opportunities = [f"opportunity-{i}" for i in range(8)]
    
    print(f"\nWorkers: {len(workers)}")
    print(f"Opportunities: {len(opportunities)} (includes 2 buffer for quality filtering)")
    
    # Distribute opportunities
    print("\n📋 Distribution:")
    for i, worker in enumerate(workers):
        if i < len(opportunities):
            print(f"   {worker} → {opportunities[i]}")
        else:
            print(f"   {worker} → No assignment (works on existing projects)")
            
    # Show buffer usage
    if len(opportunities) > len(workers):
        print(f"\n🔄 Buffer opportunities ({len(opportunities) - len(workers)}):")
        for i in range(len(workers), len(opportunities)):
            print(f"   {opportunities[i]} (backup if others fail validation)")


async def main():
    """Run all tests."""
    print("🚀 Batch Research System Test")
    print("=" * 60)
    
    # Test dynamic research calculation
    await test_dynamic_research()
    
    # Test direct Gemini research
    # await test_direct_gemini_research()  # Commented out to avoid API calls
    
    # Demonstrate distribution
    await demonstrate_worker_distribution()
    
    print("\n\n✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())