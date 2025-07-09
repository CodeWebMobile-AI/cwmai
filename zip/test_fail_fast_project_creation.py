#!/usr/bin/env python3
"""
Test script to verify fail-fast behavior in project creation.

This script will attempt to create a project and verify that:
1. All steps must succeed for project creation to complete
2. Any failure causes the entire project creation to fail
3. Detailed error messages are provided
"""

import asyncio
import logging
import json
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_fail_fast_behavior():
    """Test that project creation fails fast on any error."""
    
    logger.info("🚀 Testing fail-fast behavior in project creation...")
    
    try:
        # Import required modules
        from scripts.project_creator import ProjectCreator
        from scripts.ai_brain_factory import AIBrainFactory
        
        # Initialize AI brain
        logger.info("Initializing AI brain...")
        ai_brain = AIBrainFactory.create_ai_brain()
        
        # Initialize project creator in local mode
        logger.info("Initializing project creator in LOCAL mode...")
        project_creator = ProjectCreator(
            github_token=None,  # Local mode
            ai_brain=ai_brain
        )
        
        # Create a test task with incomplete data to trigger failure
        test_task = {
            'title': 'Test Fail-Fast Project',
            'metadata': {
                'selected_project': {
                    'name': 'test-fail-fast-project',
                    'description': 'Testing fail-fast behavior',
                    # Intentionally missing required fields to test validation
                    # Missing: problem_statement, initial_features
                },
                'market_research': {
                    'problem_description': 'Test problem',
                    'affected_audience': 'Test users',
                    'market_size': 'Test market',
                    'competition_level': 'Low',
                    'current_solutions': 'None',
                    'validation_sources': ['Test source']
                },
                'validation_metrics': {
                    'validation_score': '10.0',
                    'revenue_potential': '$100K',
                    'market_demand': 'High'
                }
            }
        }
        
        logger.info("Attempting to create project with incomplete data...")
        
        # This should fail due to missing required fields
        result = await project_creator.create_project(test_task)
        
        # If we get here, the test failed
        logger.error("❌ TEST FAILED: Project creation should have failed but succeeded!")
        return False
        
    except ValueError as e:
        error_msg = str(e)
        logger.info(f"✅ Project creation failed as expected with error: {error_msg}")
        
        # Verify we got a meaningful error message
        if "missing required fields" in error_msg or "CRITICAL" in error_msg:
            logger.info("✅ Got expected detailed error message")
            
            # Now test with complete data to ensure normal flow works
            logger.info("\n🔄 Now testing with complete data...")
            
            complete_task = {
                'title': 'Test Complete Project',
                'metadata': {
                    'selected_project': {
                        'name': 'test-complete-project',
                        'description': 'A complete test project with all required fields',
                        'problem_statement': 'Testing the complete project creation flow',
                        'initial_features': [
                            'User authentication and registration',
                            'Dashboard with analytics',
                            'API for third-party integrations',
                            'Real-time notifications',
                            'Admin panel'
                        ],
                        'target_audience': 'Developers and businesses',
                        'monetization_strategy': 'SaaS subscription model'
                    },
                    'market_research': {
                        'problem_description': 'Need for better project management',
                        'affected_audience': 'Development teams',
                        'market_size': '$10B market',
                        'competition_level': 'Moderate',
                        'current_solutions': 'Existing tools lack integration',
                        'validation_sources': ['Reddit discussions', 'Forum posts', 'Surveys']
                    },
                    'validation_metrics': {
                        'validation_score': '9.5',
                        'revenue_potential': '$500K ARR',
                        'market_demand': 'Very High'
                    },
                    'architecture': {
                        'design_system': {
                            'suggested_font': {
                                'font_name': 'Inter',
                                'google_font_link': 'https://fonts.google.com/specimen/Inter',
                                'font_stack': "'Inter', sans-serif",
                                'rationale': 'Modern and highly readable'
                            },
                            'color_palette': {
                                'primary': {'name': 'Blue', 'hex': '#3B82F6', 'usage': 'Primary actions'},
                                'secondary': {'name': 'Purple', 'hex': '#8B5CF6', 'usage': 'Secondary elements'},
                                'success': {'name': 'Green', 'hex': '#10B981', 'usage': 'Success states'},
                                'danger': {'name': 'Red', 'hex': '#EF4444', 'usage': 'Error states'}
                            }
                        },
                        'foundational_architecture': {
                            'core_components': {'content': 'Laravel backend, React frontend'},
                            'database_schema': {'content': 'PostgreSQL with proper indexing'},
                            'api_design': {'content': 'RESTful API with versioning'}
                        },
                        'feature_implementation_roadmap': []
                    }
                }
            }
            
            try:
                result = await project_creator.create_project(complete_task)
                
                # Verify all components were created
                if result.get('success'):
                    logger.info("✅ Project created successfully with all components!")
                    
                    # Check that all documentation was created
                    customizations = result.get('customizations', {})
                    docs_created = customizations.get('documentation_created', {})
                    
                    if all([
                        docs_created.get('readme'),
                        docs_created.get('architecture'),
                        docs_created.get('monetization'),
                        docs_created.get('metadata')
                    ]):
                        logger.info("✅ All documentation files were created")
                        
                        # Verify files exist in local mode
                        repo_name = result.get('repo_name')
                        if repo_name:
                            repo_path = Path(f"scripts/local_repos/CodeWebMobile-AI/{repo_name}")
                            files_to_check = [
                                'README.md',
                                'ARCHITECTURE.md', 
                                'MONETIZATION_STRATEGY.md',
                                'metadata.json',
                                'package.json'
                            ]
                            
                            missing_files = []
                            for file_name in files_to_check:
                                file_path = repo_path / file_name
                                if file_path.exists():
                                    size = file_path.stat().st_size
                                    logger.info(f"✅ {file_name} exists ({size} bytes)")
                                else:
                                    missing_files.append(file_name)
                                    logger.error(f"❌ {file_name} is missing!")
                            
                            if missing_files:
                                logger.error(f"❌ Missing files: {', '.join(missing_files)}")
                                return False
                            else:
                                logger.info("✅ All required files were created successfully")
                                return True
                    else:
                        logger.error("❌ Not all documentation was created")
                        return False
                else:
                    logger.error("❌ Project creation reported failure")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to create complete project: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        else:
            logger.error(f"❌ Got unexpected error message: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run the fail-fast test."""
    success = await test_fail_fast_behavior()
    
    if success:
        logger.info("\n✅ FAIL-FAST TEST PASSED!")
        logger.info("The system correctly:")
        logger.info("1. Failed when required fields were missing")
        logger.info("2. Provided detailed error messages") 
        logger.info("3. Created all components when given complete data")
    else:
        logger.error("\n❌ FAIL-FAST TEST FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())