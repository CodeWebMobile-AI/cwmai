#!/usr/bin/env python3
"""
Test Research-Based Project Creation Flow

This script tests that the system now uses deep market research
to create projects based on real problems.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.ai_brain import IntelligentAIBrain
from scripts.charter_system import CharterSystem
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.project_creator import ProjectCreator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_research_flow():
    """Test the complete research-based project creation flow."""
    
    logger.info("🚀 Starting Research-Based Project Creation Test")
    logger.info("=" * 60)
    
    # Initialize AI brain
    logger.info("Initializing AI brain...")
    ai_brain = IntelligentAIBrain()
    
    # Initialize charter system
    logger.info("Initializing charter system...")
    charter_system = CharterSystem()
    
    # Initialize task generator with research modules
    logger.info("Initializing intelligent task generator...")
    task_generator = IntelligentTaskGenerator(
        ai_brain=ai_brain,
        charter_system=charter_system
    )
    
    # Check if research modules are initialized
    if task_generator.deep_research:
        logger.info("✅ Deep research module initialized")
    else:
        logger.error("❌ Deep research module NOT initialized")
        
    if task_generator.project_discovery:
        logger.info("✅ Project discovery module initialized")
    else:
        logger.error("❌ Project discovery module NOT initialized")
    
    # Generate a NEW_PROJECT task
    logger.info("\n📋 Generating NEW_PROJECT task...")
    context = {
        'active_projects': {},
        'request_new_project': True
    }
    
    try:
        task = await task_generator.generate_intelligent_task(context)
        
        if task and task.get('type') == 'NEW_PROJECT':
            logger.info(f"✅ Generated task: {task.get('title')}")
            
            # Check if it's research-based
            metadata = task.get('metadata', {})
            if metadata.get('research_based'):
                logger.info("✅ Task is based on market research!")
                
                # Show market research data
                if metadata.get('market_research'):
                    logger.info("\n📊 Market Research Data:")
                    market = metadata['market_research']
                    logger.info(f"  - Market size: {market.get('market_size', 'Unknown')}")
                    logger.info(f"  - Competition: {market.get('competition_level', 'Unknown')}")
                    
                # Show selected project
                if metadata.get('selected_project'):
                    project = metadata['selected_project']
                    logger.info(f"\n🎯 Selected Project:")
                    logger.info(f"  - Name: {project.get('project_name')}")
                    logger.info(f"  - Problem: {project.get('problem_solved')}")
                    logger.info(f"  - Target: {project.get('target_audience')}")
                    logger.info(f"  - Revenue: {project.get('monetization_strategy')}")
                    
                # Show validation metrics
                if metadata.get('validation_metrics'):
                    metrics = metadata['validation_metrics']
                    logger.info(f"\n📈 Validation Metrics:")
                    logger.info(f"  - Score: {metrics.get('validation_score', 0)}/10")
                    logger.info(f"  - Demand: {metrics.get('market_demand')}")
                    logger.info(f"  - Revenue Potential: {metrics.get('revenue_potential')}")
            else:
                logger.warning("⚠️ Task is NOT research-based (using AI fallback)")
                
            # Test project creation with the task
            logger.info("\n🏗️ Testing project creation...")
            project_creator = ProjectCreator(
                github_token=os.getenv('GITHUB_TOKEN'),
                ai_brain=ai_brain
            )
            
            # Check if metadata has required fields
            has_architecture = bool(metadata.get('architecture'))
            has_selected_project = bool(metadata.get('selected_project'))
            
            logger.info(f"  - Has architecture: {has_architecture}")
            logger.info(f"  - Has selected_project: {has_selected_project}")
            
            if has_architecture and has_selected_project:
                logger.info("✅ Task has all required metadata for project creation!")
            else:
                logger.warning("⚠️ Task missing required metadata")
                
        else:
            logger.error("❌ Failed to generate NEW_PROJECT task")
            
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_research_flow())