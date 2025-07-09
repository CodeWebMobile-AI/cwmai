#!/usr/bin/env python3
"""Simple test of the project quality improvements."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_quality_improvements():
    """Test the implemented quality improvements."""
    
    logger.info("=== Testing Project Quality Improvements ===")
    
    # Initialize components directly
    ai_brain = IntelligentAIBrain()
    research_engine = GeminiDeepResearch(ai_brain)
    
    # Run a small research test
    logger.info("\n1. Running research with quality improvements...")
    opportunities = await research_engine.research_market_opportunities(
        count=1,  # Just find 1 project to test
        existing_portfolio={}
    )
    
    if opportunities:
        logger.info(f"\n✅ Found {len(opportunities)} opportunity")
        opp = opportunities[0]
        
        # Pretty print the opportunity
        logger.info("\n" + "="*60)
        logger.info(f"PROJECT: {opp.project_name}")
        logger.info("="*60)
        
        logger.info(f"\n📌 PROBLEM:")
        logger.info(f"   {opp.problem_description}")
        
        logger.info(f"\n👥 AUDIENCE:")
        logger.info(f"   {opp.affected_audience}")
        
        logger.info(f"\n💡 SOLUTION:")
        logger.info(f"   {opp.opportunity}")
        
        logger.info(f"\n🚀 KEY FEATURES ({len(opp.key_features)}):")
        for i, feature in enumerate(opp.key_features, 1):
            logger.info(f"   {i}. {feature}")
        
        logger.info(f"\n💰 REVENUE:")
        logger.info(f"   Model: {opp.monetization_model}")
        logger.info(f"   Potential: {opp.potential_revenue}")
        logger.info(f"   Market Size: {opp.market_size}")
        
        logger.info(f"\n📊 VALIDATION:")
        for source in opp.validation_sources:
            if source and source.strip():
                logger.info(f"   • {source[:100]}...")
        
        logger.info(f"\n🎯 SUCCESS METRICS:")
        for insight in opp.key_insights:
            if any(keyword in insight for keyword in ["Success", "ARPU", "MVP"]):
                logger.info(f"   • {insight}")
        
        # Quality checks
        logger.info("\n" + "="*60)
        logger.info("QUALITY CHECKS:")
        logger.info("="*60)
        
        checks = {
            "✅ Concrete features (5+)": len(opp.key_features) >= 5,
            "✅ Non-generic features": not any('Campaign Management' in f for f in opp.key_features),
            "✅ Has validation sources": any(v and v.strip() for v in opp.validation_sources),
            "✅ Has success probability": any("Success Probability" in i for i in opp.key_insights),
            "✅ Has creative name": bool(opp.project_name) and len(opp.project_name) <= 15,
            "✅ Has specific market size": "$" in opp.market_size and opp.market_size != "$1B+",
            "✅ Problem-solution alignment": opp.affected_audience.lower() in opp.opportunity.lower()
        }
        
        passed = 0
        for check, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"{status} {check}")
            if result:
                passed += 1
        
        logger.info(f"\nQuality Score: {passed}/{len(checks)} ({passed/len(checks)*100:.0f}%)")
        
        # Save test results
        output_dir = Path("test_quality_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"quality_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        test_data = {
                'timestamp': datetime.now().isoformat(),
                'quality_score': f"{passed}/{len(checks)}",
                'opportunity': {
                    'project_name': opp.project_name,
                    'problem': opp.problem_description,
                    'audience': opp.affected_audience,
                    'solution': opp.opportunity,
                    'features': opp.key_features,
                    'market_size': opp.market_size,
                    'validation_sources': opp.validation_sources,
                    'insights': opp.key_insights,
                    'revenue_potential': opp.potential_revenue,
                    'search_evidence': opp.search_evidence
                },
                'quality_checks': checks
            }
        
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"\n📁 Test results saved to: {output_file}")
        
        return passed >= 5  # Pass if at least 5/7 checks pass
    else:
        logger.error("❌ No opportunities found - test failed")
        return False

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        logger.error("Please set GOOGLE_API_KEY environment variable")
        logger.info("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    try:
        success = asyncio.run(test_quality_improvements())
        logger.info(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)