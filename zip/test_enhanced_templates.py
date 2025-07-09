#!/usr/bin/env python3
"""
Test the enhanced documentation template integration in project creator.
"""

import asyncio
import json
import logging
from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enhanced_documentation():
    """Test the enhanced documentation generation."""
    
    # Create AI brain for testing
    ai_brain = AIBrainFactory.create_for_testing()
    
    # Create project creator
    project_creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Test project details
    test_details = {
        'name': 'review-insight-ai',
        'description': 'AI-powered review management platform that helps businesses monitor, analyze, and respond to customer feedback across multiple platforms',
        'problem_statement': 'Small businesses struggle to manage reviews across 50+ platforms, resulting in poor response rates and damaged reputation',
        'target_audience': 'Small to medium businesses in hospitality, retail, and service industries',
        'initial_features': [
            'Multi-platform review aggregation',
            'AI-powered response generation',
            'Sentiment analysis dashboard',
            'Reputation tracking',
            'Automated alerts'
        ],
        'metadata': {
            'market_research': {
                'problem_description': 'Businesses receive reviews on Google, Yelp, TripAdvisor, Facebook, and dozens of other platforms but lack time and expertise to manage them effectively',
                'affected_audience': 'Small business owners who spend 10+ hours per week on review management',
                'market_size': '$2.3 billion review management software market',
                'competition_level': 'Moderate - existing solutions are expensive or lack AI capabilities',
                'current_solutions': 'Manual monitoring, expensive agencies, or basic aggregation tools',
                'validation_sources': ['Industry reports', 'Customer interviews', 'Competitor analysis']
            },
            'validation_metrics': {
                'validation_score': 9.2,
                'revenue_potential': '$75,000-$150,000 ARR in year one'
            }
        }
    }
    
    logger.info("🚀 Testing enhanced documentation generation...")
    
    try:
        # Generate comprehensive documentation
        # Note: The method expects a repo object, so we'll create a mock one
        class MockRepo:
            def __init__(self):
                self.full_name = "CodeWebMobile-AI/review-insight-ai"
                self.name = "review-insight-ai"
                self.path = "/tmp/test-repo"
        
        mock_repo = MockRepo()
        structured_docs = await project_creator._generate_all_structured_documents(mock_repo, test_details)
        
        # Log results
        logger.info("\n✅ Documentation generated successfully!")
        
        # Check README
        if structured_docs.get('readme'):
            logger.info("\n📄 README Structure:")
            readme = structured_docs['readme']
            logger.info(f"  - Title: {readme.get('title')}")
            logger.info(f"  - Tagline: {readme.get('tagline')}")
            logger.info(f"  - Features: {len(readme.get('key_features', []))} features")
            logger.info(f"  - Has Quick Start: {'quick_start' in readme}")
            logger.info(f"  - Has Project Structure: {'project_structure' in readme}")
        
        # Check Architecture
        if structured_docs.get('architecture'):
            logger.info("\n🏗️ Architecture Structure:")
            arch = structured_docs['architecture']
            logger.info(f"  - Executive Summary: {'executive_summary' in arch}")
            logger.info(f"  - System Context: {'system_context' in arch}")
            logger.info(f"  - Building Blocks: {'building_blocks' in arch}")
            logger.info(f"  - Starter Kit Foundation: {'starter_kit_foundation' in str(arch)}")
            logger.info(f"  - Implementation Roadmap: {'implementation_roadmap' in arch}")
            logger.info(f"  - AI Integration: {'ai_integration' in arch}")
            
        # Check Monetization
        if structured_docs.get('monetization'):
            logger.info("\n💰 Monetization Structure:")
            mon = structured_docs['monetization']
            logger.info(f"  - Executive Summary: {'executive_summary' in mon}")
            logger.info(f"  - Pricing Tiers: {len(mon.get('pricing_tiers', []))} tiers")
            logger.info(f"  - Implementation Code: {'implementation_code' in mon}")
            logger.info(f"  - Metrics Queries: {'metrics_queries' in mon}")
            logger.info(f"  - Growth Projections: {'growth_projections' in mon}")
        
        # Validate documentation
        issues = project_creator._validate_generated_documentation(structured_docs)
        if issues:
            logger.warning(f"\n⚠️ Validation issues found: {issues}")
        else:
            logger.info("\n✅ All validation checks passed!")
            
        # Save sample output
        output_file = 'test_enhanced_documentation_output.json'
        with open(output_file, 'w') as f:
            json.dump(structured_docs, f, indent=2)
        logger.info(f"\n💾 Full output saved to {output_file}")
        
        # Test markdown transformation
        logger.info("\n📝 Testing markdown transformation...")
        
        if structured_docs.get('architecture'):
            arch_md = project_creator._transform_structured_architecture_to_markdown(structured_docs['architecture'])
            logger.info(f"  - Architecture markdown: {len(arch_md)} characters")
            
            # Save sample markdown
            with open('test_architecture_output.md', 'w') as f:
                f.write(arch_md)
            logger.info("  - Sample saved to test_architecture_output.md")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_enhanced_documentation())