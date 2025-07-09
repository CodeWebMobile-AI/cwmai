#!/usr/bin/env python3
"""
Test just the README generation with research data
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

async def test_readme_generation():
    """Test generating a comprehensive README with research data."""
    
    logger.info("📝 Testing Comprehensive README Generation")
    logger.info("=" * 60)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    project_creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Create a test repository
    local_sim = LocalGitHubSimulator()
    test_repo_name = "test-readme-generation"
    test_repo = local_sim.create_repository(test_repo_name, "Test repository for README generation")
    
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
                'current_solutions': 'Zoho Inventory, Odoo Inventory, Square for Retail',
                'problem_description': 'Lack of affordable inventory management that integrates seamlessly',
                'affected_audience': 'Small business owners',
                'validation_sources': ['Reddit (r/smallbusiness)', 'fitsmallbusiness.com', 'Google Search Trends']
            },
            'validation_metrics': {
                'validation_score': 9.0,
                'market_demand': 'High',
                'revenue_potential': '$50,000 ARR in year one'
            }
        }
    }
    
    # Test the README generation
    logger.info("\n🔨 Generating comprehensive README...")
    try:
        success = await project_creator._generate_comprehensive_readme(test_repo, project_details)
        
        if success:
            logger.info("✅ README generation succeeded!")
            
            # Check the generated README
            readme_path = Path(test_repo.path) / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    content = f.read()
                    
                logger.info(f"\n📄 README.md created ({len(content)} bytes)")
                logger.info("\n📖 README Preview:")
                logger.info("-" * 60)
                
                # Show first 50 lines
                lines = content.split('\n')[:50]
                for line in lines:
                    logger.info(line)
                
                logger.info(f"\n... (Total {len(content.split(chr(10)))} lines)")
                
                # Check for key sections
                logger.info("\n✅ Key sections found:")
                sections = [
                    "Problem & Solution",
                    "Target Audience", 
                    "Key Features",
                    "Tech Stack",
                    "Getting Started",
                    "Architecture Overview",
                    "Monetization Strategy",
                    "Development Roadmap"
                ]
                
                for section in sections:
                    if section.lower() in content.lower():
                        logger.info(f"  ✓ {section}")
                    else:
                        logger.info(f"  ✗ {section} (missing)")
            else:
                logger.error("❌ README.md file not found!")
        else:
            logger.error("❌ README generation failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during README generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_readme_generation())