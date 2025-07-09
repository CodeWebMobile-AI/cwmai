#!/usr/bin/env python3
"""Test the refactored Gemini Deep Research methods."""

import asyncio
import json
import sys
from pathlib import Path
import logging

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_refactored_methods():
    """Test the refactored _phase_4_5_unit_economics and _phase_4_75_mvp_simulation methods."""
    print("Testing refactored Gemini Deep Research methods...")
    
    try:
        from scripts.gemini_deep_research import GeminiDeepResearch
        from scripts.ai_brain import IntelligentAIBrain
        
        # Initialize AI brain with minimal setup
        ai_brain = IntelligentAIBrain()
        
        # Create mock research instance
        research = GeminiDeepResearch(ai_brain, max_opportunities=3)
        
        # Test data - simplified opportunities
        test_opportunities = [
            {
                'id': 'opp1',
                'problem_description': 'Small businesses struggle with inventory management',
                'opportunity': 'AI-powered inventory prediction system',
                'affected_audience': 'Small retail businesses',
                'market_size': '$5B',
                'annual_revenue_potential': '$10M',
                'revenue_model': 'SaaS subscription',
                'key_features': ['Inventory tracking', 'AI predictions', 'Supplier integration'],
                'tech_stack': ['Python', 'React', 'PostgreSQL'],
                'critical_features': ['Real-time tracking', 'Predictive analytics']
            },
            {
                'id': 'opp2',
                'problem_description': 'Remote teams lack effective collaboration tools',
                'opportunity': 'Async video collaboration platform',
                'affected_audience': 'Remote tech teams',
                'market_size': '$3B',
                'annual_revenue_potential': '$8M',
                'revenue_model': 'Per-seat pricing',
                'key_features': ['Video messages', 'Thread discussions', 'Integration APIs'],
                'tech_stack': ['Node.js', 'Vue.js', 'MongoDB'],
                'critical_features': ['Async video', 'Threading', 'Search']
            }
        ]
        
        print("\n1. Testing _phase_4_5_unit_economics (split into 3 calls)...")
        try:
            # Test unit economics
            result = await research._phase_4_5_unit_economics(test_opportunities, batch_size=2)
            
            # Verify results
            for opp in result:
                print(f"\n   Opportunity: {opp['problem_description'][:50]}...")
                print(f"   - Has CAC data: {'estimated_cac' in opp}")
                print(f"   - Has LTV data: {'average_ltv' in opp}")
                print(f"   - Has growth data: {'monthly_growth_rate' in opp}")
                
            print("   ✅ Unit economics method completed successfully")
            
        except Exception as e:
            print(f"   ❌ Unit economics test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n2. Testing _phase_4_75_mvp_simulation (split into 3 calls)...")
        try:
            # Test MVP simulation
            result = await research._phase_4_75_mvp_simulation(test_opportunities, batch_size=2)
            
            # Verify results
            for opp in result:
                print(f"\n   Opportunity: {opp['problem_description'][:50]}...")
                print(f"   - Has MVP score: {'mvp_score' in opp}")
                print(f"   - Has core features: {'core_features' in opp}")
                print(f"   - Has revenue projections: {'revenue_projection_weekly' in opp}")
                print(f"   - Has traction indicators: {'early_traction_indicators' in opp}")
                
            print("   ✅ MVP simulation method completed successfully")
            
        except Exception as e:
            print(f"   ❌ MVP simulation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n3. Checking helper methods exist...")
        helper_methods = [
            '_phase_4_5_cac_analysis',
            '_phase_4_5_ltv_margins',
            '_phase_4_5_growth_metrics',
            '_phase_4_75_mvp_features',
            '_phase_4_75_revenue_projections',
            '_phase_4_75_traction_indicators'
        ]
        
        missing_methods = []
        for method_name in helper_methods:
            if hasattr(research, method_name):
                print(f"   ✅ {method_name} exists")
            else:
                print(f"   ❌ {method_name} missing")
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"\n   ⚠️  Missing helper methods: {missing_methods}")
            return False
        
        print("\n✅ All refactored methods are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_refactored_methods())
    sys.exit(0 if success else 1)