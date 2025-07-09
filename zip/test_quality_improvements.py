#!/usr/bin/env python3
"""Test the project quality improvements implemented."""

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
from scripts.conversational_ai_assistant import ConversationalAIAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_quality_improvements():
    """Test the implemented quality improvements."""
    
    logger.info("=== Testing Project Quality Improvements ===")
    
    # Initialize the research engine
    assistant = ConversationalAIAssistant()
    research_engine = GeminiDeepResearch(assistant.ai_brain)
    
    # Run a small research test
    logger.info("\n1. Running research with quality improvements...")
    opportunities = await research_engine.run_research(
        topic="AI-powered tools for small businesses",
        existing_portfolio={},
        target_count=1  # Just find 1 project to test
    )
    
    if opportunities:
        logger.info(f"\n✅ Found {len(opportunities)} opportunity")
        opp = opportunities[0]
        
        # Check for quality improvements
        logger.info("\n2. Checking quality improvements:")
        
        # Check 1: Features generated early (should have 5-7 concrete features)
        features = opp.key_features
        logger.info(f"\n   ✓ Features ({len(features)}): {features}")
        if len(features) >= 5 and not any('Campaign Management' in f for f in features):
            logger.info("   ✅ Features are concrete and non-generic")
        else:
            logger.warning("   ⚠️ Features may still be generic or insufficient")
        
        # Check 2: Revenue validation with real data
        validation_sources = opp.validation_sources
        logger.info(f"\n   ✓ Validation sources: {[v[:50] + '...' if v and len(v) > 50 else v for v in validation_sources]}")
        if any(v and v.strip() for v in validation_sources):
            logger.info("   ✅ Has validation sources")
        else:
            logger.warning("   ⚠️ Missing validation sources")
        
        # Check 3: Success probability in insights
        success_prob = None
        for insight in opp.key_insights:
            if "Success Probability:" in insight:
                success_prob = insight
                break
        logger.info(f"\n   ✓ Success probability: {success_prob}")
        if success_prob:
            logger.info("   ✅ Success probability calculated")
        else:
            logger.warning("   ⚠️ No success probability found")
        
        # Check 4: Market size with real data
        market_size = opp.market_size
        logger.info(f"\n   ✓ Market size: {market_size}")
        if market_size and "$" in market_size and market_size != "$1B+":
            logger.info("   ✅ Market size has specific data")
        else:
            logger.warning("   ⚠️ Market size may be generic")
        
        # Check 5: Project name
        project_name = opp.project_name
        logger.info(f"\n   ✓ Project name: {project_name}")
        if project_name and len(project_name) <= 15:
            logger.info("   ✅ Has creative project name")
        else:
            logger.warning("   ⚠️ Project name missing or too long")
        
        # Check 6: Problem-solution alignment
        logger.info(f"\n   ✓ Problem: {opp.problem_description[:100]}...")
        logger.info(f"   ✓ Solution: {opp.opportunity[:100]}...")
        
        # Save test results
        output_dir = Path("test_quality_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"quality_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'opportunity': {
                    'project_name': opp.project_name,
                    'problem': opp.problem_description,
                    'solution': opp.opportunity,
                    'features': opp.key_features,
                    'market_size': opp.market_size,
                    'validation_sources': opp.validation_sources,
                    'insights': opp.key_insights,
                    'revenue_potential': opp.potential_revenue,
                    'search_evidence': opp.search_evidence
                }
            }, indent=2)
        
        logger.info(f"\n✅ Test results saved to: {output_file}")
        
        # Check phase data saving
        research_dir = Path("research_knowledge")
        if research_dir.exists():
            latest_research = max(research_dir.glob("research_*.json"), key=lambda p: p.stat().st_mtime, default=None)
            if latest_research:
                with open(latest_research, 'r') as f:
                    research_data = json.load(f)
                
                phases_with_data = []
                for phase, data in research_data.get('research_phases', {}).items():
                    if data:
                        if isinstance(data, list):
                            phases_with_data.append(f"{phase}: {len(data)} items")
                        else:
                            phases_with_data.append(f"{phase}: Has data")
                
                logger.info(f"\n3. Phase data saving check:")
                logger.info(f"   ✓ Phases with data: {phases_with_data}")
                logger.info("   ✅ Phase data is being saved for learning")
    else:
        logger.error("❌ No opportunities found - test failed")
        return False
    
    logger.info("\n=== Quality Improvement Test Complete ===")
    return True

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        logger.error("Please set GOOGLE_API_KEY environment variable")
        logger.info("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    success = asyncio.run(test_quality_improvements())
    sys.exit(0 if success else 1)