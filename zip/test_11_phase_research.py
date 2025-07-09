#!/usr/bin/env python3
"""
Test the complete 11-phase research system with long timeouts.
This will run through all phases and show detailed progress.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_full_research():
    """Test the complete 11-phase research system."""
    print("\n" + "="*80)
    print("🚀 TESTING 11-PHASE EXHAUSTIVE RESEARCH SYSTEM")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print("This will take several minutes to complete all phases...")
    print("="*80 + "\n")
    
    try:
        # Initialize research system
        print("📋 Initializing research system...")
        ai_brain = IntelligentAIBrain()
        research = GeminiDeepResearch(ai_brain)
        
        # Show configuration
        print("\n📊 System Configuration:")
        print(f"  - Total phases: {len(research.research_phases_data)}")
        print(f"  - Phase limits: {json.dumps(research.PHASE_LIMITS, indent=2)}")
        print(f"  - Min scores: {json.dumps(research.MIN_SCORES, indent=2)}")
        print(f"  - API key available: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")
        
        if not os.getenv('GOOGLE_API_KEY'):
            print("\n❌ ERROR: GOOGLE_API_KEY not found in environment!")
            print("Please run: export $(cat .env.local | grep -v '^#' | xargs)")
            return
        
        # Run research with extended timeout
        print("\n🔬 Starting research for 1 high-quality opportunity...")
        print("Each phase will be logged as it executes...\n")
        
        start_time = datetime.now()
        
        # Use asyncio.wait_for with a very long timeout (30 minutes)
        opportunities = await asyncio.wait_for(
            research.research_market_opportunities(
                count=1,
                existing_portfolio={}
            ),
            timeout=1800  # 30 minutes
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("📈 RESEARCH RESULTS")
        print("="*80)
        
        if opportunities:
            print(f"\n✅ Found {len(opportunities)} opportunity!")
            
            for i, opp in enumerate(opportunities):
                print(f"\n{'='*60}")
                print(f"OPPORTUNITY {i+1}")
                print(f"{'='*60}")
                print(f"🏷️  Project Name: {opp.project_name or 'Not Generated'}")
                print(f"❓  Problem: {opp.problem_description[:200]}...")
                print(f"👥  Audience: {opp.affected_audience}")
                print(f"💡  Solution: {opp.opportunity[:200]}...")
                print(f"💰  Revenue Model: {opp.monetization_model}")
                print(f"📊  Market Size: {opp.market_size}")
                print(f"🎯  Success Probability: {opp.search_evidence.get('success_probability', 0):.1f}%")
                print(f"⏱️  MVP Timeline: {opp.implementation_difficulty}")
                print(f"💵  Revenue Potential: {opp.potential_revenue}")
                
                print(f"\n🔑 Key Features ({len(opp.key_features)}):")
                for j, feature in enumerate(opp.key_features[:5]):
                    print(f"   {j+1}. {feature}")
                
                print(f"\n📈 Key Insights:")
                for insight in opp.key_insights[:3]:
                    print(f"   • {insight}")
                
                print(f"\n✅ Validation Sources:")
                for source in opp.validation_sources[:3]:
                    if source:
                        print(f"   • {source[:100]}...")
        else:
            print("\n❌ No opportunities found that passed all 11 phases!")
            print("This is expected with the strict quality thresholds.")
        
        # Show phase progression data
        print("\n" + "="*80)
        print("📊 PHASE PROGRESSION ANALYSIS")
        print("="*80)
        
        phase_data = research.research_phases_data
        
        # Phase 0 - Market Timing
        if phase_data.get('phase_0_market_timing'):
            timing = phase_data['phase_0_market_timing']
            print(f"\n📅 Phase 0 - Market Timing:")
            print(f"   - Timing Score: {timing.get('timing_score', 0)}/10")
            print(f"   - Trend Score: {timing.get('trend_score', 0)}/10")
            print(f"   - Recommendation: {timing.get('recommendation', 'Unknown')}")
            if timing.get('emerging_trends'):
                print(f"   - Emerging Trends: {len(timing['emerging_trends'])}")
        
        # Phase 1 - Problems Discovered
        phase1 = phase_data.get('phase_1_problems_discovered', [])
        print(f"\n🔍 Phase 1 - Problem Discovery:")
        print(f"   - Problems discovered: {len(phase1)}")
        if phase1:
            print(f"   - Features generated per problem: {len(phase1[0].get('key_features', []))}")
        
        # Phase 2+ - Show filtering
        phases_with_data = [
            ('phase_2_problems_validated', 'Phase 2 - Market Validation'),
            ('phase_2_5_user_research', 'Phase 2.5 - User Research'),
            ('phase_2_75_revenue_validation', 'Phase 2.75 - Revenue Validation'),
            ('phase_3_competition_analysis', 'Phase 3 - Competition Analysis'),
            ('phase_3_5_technical_validation', 'Phase 3.5 - Technical Validation'),
            ('phase_4_market_sizing', 'Phase 4 - Market Sizing'),
            ('phase_4_5_unit_economics', 'Phase 4.5 - Unit Economics'),
            ('phase_4_75_mvp_simulation', 'Phase 4.75 - MVP Simulation'),
            ('phase_5_final_selection', 'Phase 5 - Final Selection')
        ]
        
        for phase_key, phase_name in phases_with_data:
            phase_items = phase_data.get(phase_key, [])
            if isinstance(phase_items, list):
                high = len([x for x in phase_items if x.get('status') == 'high_potential'])
                medium = len([x for x in phase_items if x.get('status') == 'medium_potential'])
                rejected = len([x for x in phase_items if x.get('status') == 'rejected'])
                
                print(f"\n📊 {phase_name}:")
                print(f"   - High potential: {high}")
                print(f"   - Medium potential: {medium}")
                print(f"   - Rejected: {rejected}")
                print(f"   - Total processed: {len(phase_items)}")
        
        # Phase 6 - Pre-Build Validation
        if phase_data.get('phase_6_pre_build_validation'):
            validation = phase_data['phase_6_pre_build_validation']
            print(f"\n✅ Phase 6 - Pre-Build Validation:")
            print(f"   - Overall: {validation.get('overall_recommendation', 'Unknown')}")
            validated = validation.get('opportunities_validated', [])
            if validated:
                for val in validated:
                    print(f"   - {val.get('project_name', 'Unknown')}: {val.get('validation_status', 'Unknown')}")
        
        # Show optimization stats
        print("\n" + "="*80)
        print("⚡ PERFORMANCE METRICS")
        print("="*80)
        print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"API calls made: {research.api_calls}")
        print(f"API calls/minute: {research.api_calls / (duration/60):.1f}")
        
        cache_stats = research.search_cache.get_stats()
        print(f"\nCache Performance:")
        print(f"  - Hit rate: {cache_stats['hit_rate']}")
        print(f"  - Total hits: {cache_stats['hits']}")
        print(f"  - Total misses: {cache_stats['misses']}")
        
        # Show kill reasons
        print("\n" + "="*80)
        print("🚫 OPPORTUNITIES FILTERED OUT")
        print("="*80)
        print("Common kill reasons during progressive filtering:")
        print("  - Insufficient market validation (< 6.0)")
        print("  - User pain not severe enough (< 7.0)")
        print("  - No proven revenue potential (< 7.0)")
        print("  - Market too competitive (< 6.0)")
        print("  - Technical complexity too high (< 7.0)")
        print("  - Market too small (< 7.0)")
        print("  - Poor unit economics (< 7.5)")
        
        print("\n" + "="*80)
        print(f"✅ Test completed at: {datetime.now()}")
        print("="*80)
        
    except asyncio.TimeoutError:
        print("\n❌ ERROR: Research timed out after 30 minutes!")
        print("This might indicate an issue with the API or network.")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check environment
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY not found!")
        print("Please run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_full_research())