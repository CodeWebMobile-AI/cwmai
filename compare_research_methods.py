#!/usr/bin/env python3
"""
Compare the old basic research method with the new Gemini deep research approach.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import IntelligentAIBrain
from scripts.http_ai_client import HTTPAIClient
from scripts.ai_response_enhancer import AIResponseEnhancer


async def test_basic_research():
    """Test the old basic research approach."""
    print("=" * 80)
    print("📖 TESTING BASIC RESEARCH (OLD METHOD)")
    print("=" * 80)
    
    # Initialize AI components
    ai_client = HTTPAIClient()
    response_enhancer = AIResponseEnhancer()
    ai_brain = IntelligentAIBrain(ai_client, response_enhancer)
    
    # Create project creator
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode='local')
    
    start_time = datetime.now()
    
    # Call the basic research method directly
    try:
        results = await creator._basic_market_research()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Basic research completed in {duration:.1f} seconds")
        print(f"📋 Found {len(results.get('opportunities', []))} opportunities")
        
        # Display results
        for i, opp in enumerate(results.get('opportunities', [])[:5], 1):
            print(f"\n{i}. {opp.get('problem_description', 'N/A')}")
            print(f"   Target: {opp.get('affected_audience', 'N/A')}")
            print(f"   Market Size: {opp.get('market_size', 'N/A')}")
            print(f"   Competition: {opp.get('competition_level', 'N/A')}")
        
        return results, duration
        
    except Exception as e:
        print(f"❌ Basic research failed: {e}")
        return None, 0


async def test_deep_research():
    """Test the new Gemini deep research approach."""
    print("\n" + "=" * 80)
    print("🔬 TESTING GEMINI DEEP RESEARCH (NEW METHOD)")
    print("=" * 80)
    
    # Initialize AI components
    ai_client = HTTPAIClient()
    response_enhancer = AIResponseEnhancer()
    ai_brain = IntelligentAIBrain(ai_client, response_enhancer)
    
    # Create project creator
    creator = ProjectCreator(ai_brain=ai_brain, storage_mode='local')
    
    start_time = datetime.now()
    
    # Call the new research method
    try:
        results = await creator._research_market_opportunities()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Deep research completed in {duration:.1f} seconds")
        print(f"📋 Found {len(results.get('opportunities', []))} opportunities")
        print(f"🌐 Analyzed {results.get('total_sectors_analyzed', 0)} sectors")
        print(f"🔍 Using: {results.get('source', 'unknown')}")
        
        # Display results
        for i, opp in enumerate(results.get('opportunities', [])[:5], 1):
            print(f"\n{i}. {opp.get('problem_description', 'N/A')}")
            print(f"   Target: {opp.get('affected_audience', 'N/A')}")
            print(f"   Market Size: {opp.get('market_size', 'N/A')}")
            print(f"   Competition: {opp.get('competition_level', 'N/A')}")
            print(f"   Revenue Potential: {opp.get('potential_revenue', 'N/A')}")
            print(f"   Validation Sources: {len(opp.get('validation_sources', []))}")
            
            # Show key insights if available
            insights = opp.get('key_insights', [])
            if insights:
                print(f"   Key Insights: {insights[0] if insights else 'N/A'}")
        
        return results, duration
        
    except Exception as e:
        print(f"❌ Deep research failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


async def compare_results(basic_results, deep_results, basic_time, deep_time):
    """Compare the results from both methods."""
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    
    # Time comparison
    print(f"\n⏱️  Time Comparison:")
    print(f"   Basic Research: {basic_time:.1f} seconds")
    print(f"   Deep Research: {deep_time:.1f} seconds")
    print(f"   Difference: {deep_time - basic_time:.1f} seconds slower")
    
    # Quantity comparison
    basic_count = len(basic_results.get('opportunities', [])) if basic_results else 0
    deep_count = len(deep_results.get('opportunities', [])) if deep_results else 0
    
    print(f"\n📈 Quantity Comparison:")
    print(f"   Basic Research: {basic_count} opportunities")
    print(f"   Deep Research: {deep_count} opportunities")
    
    # Quality comparison
    print(f"\n🎯 Quality Comparison:")
    
    if basic_results:
        print(f"\n   Basic Research Features:")
        print(f"   - Simple problem/solution pairs")
        print(f"   - Basic market size estimates")
        print(f"   - General competition assessment")
        print(f"   - No validation sources")
        print(f"   - Limited to predefined areas")
    
    if deep_results:
        print(f"\n   Deep Research Features:")
        print(f"   - Validated with Google Search data")
        print(f"   - {deep_results.get('total_sectors_analyzed', 25)} sectors analyzed")
        print(f"   - Revenue potential estimates")
        print(f"   - Key insights provided")
        print(f"   - Unique advantages identified")
        print(f"   - Validation sources included")
        print(f"   - Dynamic market exploration")
    
    # Data richness comparison
    if basic_results and deep_results:
        basic_opp = basic_results.get('opportunities', [{}])[0]
        deep_opp = deep_results.get('opportunities', [{}])[0]
        
        print(f"\n📊 Data Richness (First Opportunity):")
        print(f"   Basic fields: {len(basic_opp.keys())}")
        print(f"   Deep fields: {len(deep_opp.keys())}")
        
        # Show unique fields in deep research
        deep_only_fields = set(deep_opp.keys()) - set(basic_opp.keys())
        if deep_only_fields:
            print(f"\n   Additional fields in Deep Research:")
            for field in sorted(deep_only_fields):
                print(f"   - {field}")


async def main():
    """Run comparison tests."""
    print("🔄 Market Research Method Comparison")
    print("This will compare the old basic research with new Gemini deep research")
    print("\nNOTE: Requires proper API keys in .env.local\n")
    
    # Run basic research
    basic_results, basic_time = await test_basic_research()
    
    print("\n" + "-" * 40)
    
    # Run deep research
    deep_results, deep_time = await test_deep_research()
    
    # Compare results
    await compare_results(basic_results, deep_results, basic_time, deep_time)
    
    # Save comparison results
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'basic_research': {
            'duration_seconds': basic_time,
            'opportunities_count': len(basic_results.get('opportunities', [])) if basic_results else 0,
            'source': 'basic_gemini_research',
            'sample': basic_results.get('opportunities', [])[0] if basic_results and basic_results.get('opportunities') else None
        },
        'deep_research': {
            'duration_seconds': deep_time,
            'opportunities_count': len(deep_results.get('opportunities', [])) if deep_results else 0,
            'source': deep_results.get('source', 'unknown') if deep_results else 'failed',
            'sectors_analyzed': deep_results.get('total_sectors_analyzed', 0) if deep_results else 0,
            'sample': deep_results.get('opportunities', [])[0] if deep_results and deep_results.get('opportunities') else None
        }
    }
    
    filename = f"research_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())