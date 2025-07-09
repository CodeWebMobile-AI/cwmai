#!/usr/bin/env python3
"""
Test creation of all 3 documentation files (README.md, ARCHITECTURE.md, MONETIZATION_STRATEGY.md)
with structured output using parallel chunked generation
"""

import asyncio
import logging
import json
from pathlib import Path

# Local imports
from scripts.ai_brain import IntelligentAIBrain
from scripts.project_creator import ProjectCreator
from scripts.local_github_simulator import LocalGitHubSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_all_docs_generation():
    """Test generating all 3 documentation files with research data."""
    
    logger.info("📝 Testing All 3 Documentation Files Generation")
    logger.info("=" * 60)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    project_creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Create a test repository
    local_sim = LocalGitHubSimulator()
    test_repo_name = "test-all-docs-generation"
    test_repo = local_sim.create_repository(test_repo_name, "Test repository for all 3 docs generation")
    
    # Mock project details with research data
    project_details = {
        'name': test_repo_name,
        'description': 'AI-powered inventory management for small businesses',
        'problem_statement': 'Small businesses struggle with multi-channel inventory management',
        'target_audience': 'Small business owners, particularly restaurants and retailers',
        'monetization_strategy': 'Subscription-based (tiered pricing based on usage)',
        'initial_features': [
            'Real-time inventory tracking across multiple channels',
            'AI-powered demand forecasting',
            'Automated reorder suggestions',
            'Integration with popular POS systems',
            'Mobile app for on-the-go management'
        ],
        'core_entities': ['Product', 'Inventory', 'Order', 'Supplier', 'Channel'],
        'architecture': {
            'overview': 'Microservices architecture with Laravel backend and React frontend',
            'tech_stack': {
                'backend': 'Laravel 11+',
                'frontend': 'React 19 with TypeScript',
                'database': 'MySQL',
                'cache': 'Redis',
                'realtime': 'Laravel Echo'
            }
        },
        'metadata': {
            'market_research': {
                'market_size': 'Large and growing (e-commerce and delivery are booming)',
                'competition_level': 'High (Zoho, Odoo, Square, Lightspeed, etc.)',
                'current_solutions': 'Zoho Inventory, Odoo Inventory, Square for Retail, Lightspeed Retail',
                'problem_description': 'Lack of affordable inventory management that integrates seamlessly',
                'affected_audience': 'Small business owners, particularly restaurants and retailers',
                'validation_sources': ['Reddit (r/smallbusiness)', 'fitsmallbusiness.com', 'Google Search Trends']
            },
            'validation_metrics': {
                'validation_score': 9.0,
                'market_demand': 'High',
                'revenue_potential': '$50,000 ARR in year one, scalable with user base growth',
                'competition_level': 'High'
            }
        }
    }
    
    # Test the complete customization process
    logger.info("\n🔨 Running _customize_project to generate all documentation...")
    try:
        results = await project_creator._customize_project(test_repo, project_details)
        
        logger.info(f"\n✅ Customization completed!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        # Check all files were created
        repo_path = Path(test_repo.path)
        
        # Check README.md
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            logger.info(f"\n✅ README.md created ({readme_path.stat().st_size} bytes)")
            with open(readme_path, 'r') as f:
                lines = f.readlines()[:10]
                logger.info("Preview:")
                for line in lines:
                    logger.info(f"  {line.rstrip()}")
        else:
            logger.error("❌ README.md not found!")
            
        # Check ARCHITECTURE.md
        arch_path = repo_path / "ARCHITECTURE.md"
        if arch_path.exists():
            logger.info(f"\n✅ ARCHITECTURE.md created ({arch_path.stat().st_size} bytes)")
            with open(arch_path, 'r') as f:
                lines = f.readlines()[:10]
                logger.info("Preview:")
                for line in lines:
                    logger.info(f"  {line.rstrip()}")
        else:
            logger.error("❌ ARCHITECTURE.md not found!")
            
        # Check MONETIZATION_STRATEGY.md
        monetization_path = repo_path / "MONETIZATION_STRATEGY.md"
        if monetization_path.exists():
            logger.info(f"\n✅ MONETIZATION_STRATEGY.md created ({monetization_path.stat().st_size} bytes)")
            with open(monetization_path, 'r') as f:
                lines = f.readlines()[:10]
                logger.info("Preview:")
                for line in lines:
                    logger.info(f"  {line.rstrip()}")
        else:
            logger.error("❌ MONETIZATION_STRATEGY.md not found!")
            
        # Check metadata.json
        metadata_path = repo_path / "metadata.json"
        if metadata_path.exists():
            logger.info(f"\n✅ metadata.json created ({metadata_path.stat().st_size} bytes)")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Structured docs keys: {list(metadata.get('structured_docs', {}).keys())}")
        else:
            logger.error("❌ metadata.json not found!")
            
        # Summary
        logger.info("\n📊 Summary:")
        logger.info(f"  - Documentation created: {results.get('documentation_created', {})}")
        logger.info(f"  - README updated: {results.get('readme_updated', False)}")
        logger.info(f"  - Architecture saved: {results.get('architecture_saved', False)}")
        logger.info(f"  - Monetization saved: {results.get('monetization_saved', False)}")
        logger.info(f"  - Metadata saved: {results.get('metadata_saved', False)}")
            
    except Exception as e:
        logger.error(f"❌ Error during documentation generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_docs_generation())