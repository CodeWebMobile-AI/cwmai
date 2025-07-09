#!/usr/bin/env python3
"""
Test the zip folder project creation with deep research and comprehensive README
This test runs entirely within the zip folder environment
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Ensure we're using zip folder modules
sys.path.insert(0, str(Path(__file__).parent))

from scripts.ai_brain import IntelligentAIBrain
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.project_creator import ProjectCreator
from scripts.storage_mode import StorageMode

# Mock charter system since it's not needed for this test
class DynamicCharterSystem:
    def __init__(self):
        pass
    
    async def get_current_charter(self):
        return {
            'focus_areas': ['innovation', 'real-world problems'],
            'goals': ['create valuable projects', 'solve real problems'],
            'constraints': ['use Laravel React starter kit']
        }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_project_creation():
    """Test creating a project with deep research and comprehensive README."""
    
    logger.info("🚀 Testing Zip Folder Project Creation with Deep Research")
    logger.info("=" * 60)
    
    # Initialize components
    ai_brain = IntelligentAIBrain()
    charter_system = DynamicCharterSystem()
    
    # Initialize task generator with research capability
    task_generator = IntelligentTaskGenerator(
        ai_brain=ai_brain,
        charter_system=charter_system
    )
    
    # Initialize project creator
    project_creator = ProjectCreator(
        github_token=None,  # Local mode
        ai_brain=ai_brain
    )
    
    # Generate a new project task with deep research
    logger.info("\n📡 Step 1: Generating project task with deep research...")
    context = {
        'existing_projects': [],
        'market_trends': [],
        'user_feedback': []
    }
    
    task = await task_generator._generate_new_project_task(context)
    
    if task:
        logger.info(f"✅ Generated task: {task.get('title')}")
        logger.info(f"Type: {task.get('type')}")
        logger.info(f"Has metadata: {bool(task.get('metadata'))}")
        logger.info(f"Has market research: {bool(task.get('metadata', {}).get('market_research'))}")
        logger.info(f"Has architecture: {bool(task.get('metadata', {}).get('architecture'))}")
        
        # Create the project
        logger.info("\n🔨 Step 2: Creating project with comprehensive README...")
        try:
            result = await project_creator.create_project(task)
            
            logger.info(f"\n✅ Project created successfully!")
            logger.info(f"Repository: {result.get('repo_name')}")
            logger.info(f"Documentation created: {result.get('customizations', {}).get('documentation_created')}")
            
            # Check if README was created
            repo_path = Path(f"local_repos/repositories/CodeWebMobile-AI/{result.get('repo_name')}")
            readme_path = repo_path / "README.md"
            
            if readme_path.exists():
                logger.info(f"\n📄 README.md exists at: {readme_path}")
                logger.info(f"README size: {readme_path.stat().st_size} bytes")
                
                # Show first few lines
                with open(readme_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')[:30]
                    logger.info("\n📖 README Preview:")
                    logger.info("-" * 60)
                    for line in lines:
                        logger.info(line)
                    logger.info(f"\n... (Total {len(content.split(chr(10)))} lines)")
            else:
                logger.error(f"❌ README.md not found at: {readme_path}")
                
            # Check what files were created
            if repo_path.exists():
                files = list(repo_path.glob("*.md"))
                logger.info(f"\n📁 Markdown files in project:")
                for file in files:
                    logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Project creation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.error("❌ Failed to generate project task")
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_project_creation())