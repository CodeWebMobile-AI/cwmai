#!/usr/bin/env python3
"""
Simple test of research-based project flow
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.ai_brain import IntelligentAIBrain
from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.unified_project_discovery import UnifiedProjectDiscovery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_research():
    """Test deep research modules."""
    
    logger.info("🔬 Testing Deep Research Flow")
    logger.info("=" * 60)
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain()
    
    # Initialize research modules
    logger.info("Initializing research modules...")
    deep_research = GeminiDeepResearch(ai_brain)
    project_discovery = UnifiedProjectDiscovery(ai_brain)
    
    # Step 1: Find real problems
    logger.info("\n📡 Step 1: Searching for real problems...")
    try:
        market_opportunities = await deep_research.conduct_deep_market_research(
            count=3,
            existing_portfolio={}
        )
        
        if market_opportunities:
            logger.info(f"✅ Found {len(market_opportunities)} market opportunities!")
            for i, opp in enumerate(market_opportunities[:3], 1):
                logger.info(f"\nOpportunity {i}:")
                logger.info(f"  Problem: {opp.problem_description[:100]}...")
                logger.info(f"  Audience: {opp.affected_audience}")
                logger.info(f"  Opportunity: {opp.opportunity[:100]}...")
                logger.info(f"  Market Size: {opp.market_size}")
                logger.info(f"  Competition: {opp.competition_level}")
        else:
            logger.warning("No opportunities found")
            
        # Step 2: Convert to validated project opportunities
        logger.info("\n🎯 Step 2: Converting to validated project opportunities...")
        
        # Convert market opportunities to the format expected by project discovery
        market_research = {
            'opportunities': [
                {
                    'problem_description': opp.problem_description,
                    'affected_audience': opp.affected_audience,
                    'opportunity': opp.opportunity,
                    'market_size': opp.market_size,
                    'monetization_model': opp.monetization_model,
                    'competition_level': opp.competition_level
                }
                for opp in market_opportunities
            ]
        }
        
        opportunities = await project_discovery.discover_and_create_projects(
            existing_portfolio={},
            max_projects=3
        )
        
        if opportunities:
            logger.info(f"✅ Generated {len(opportunities)} project opportunities!")
            
            best = opportunities[0]
            logger.info(f"\nBest Opportunity:")
            logger.info(f"  Project: {best.project_name}")
            logger.info(f"  Tagline: {best.tagline}")
            logger.info(f"  Problem: {best.problem_statement}")
            logger.info(f"  Solution: {best.solution_approach}")
            logger.info(f"  Score: {best.opportunity_score}/10")
            logger.info(f"  Revenue: {best.revenue_projection}")
            
            # Show this is what would be used for project creation
            logger.info("\n📦 This validated opportunity would be used to:")
            logger.info("  1. Fork Laravel React starter kit")
            logger.info("  2. Create project solving the validated problem")
            logger.info("  3. Generate documentation from research data")
            logger.info("  4. No more generic 'task management apps'!")
            
        else:
            logger.warning("No opportunities generated")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_research())