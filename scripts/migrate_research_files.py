"""
Migrate existing research files to their proper subdirectories.

This script will move research files from the root raw_research directory
to their appropriate subdirectories based on their type.
"""

import json
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_research_files():
    """Migrate research files to proper subdirectories."""
    research_path = Path("research_knowledge/raw_research")
    
    if not research_path.exists():
        logger.error(f"Research path {research_path} does not exist")
        return
    
    # Type mapping (same as in ResearchKnowledgeStore)
    type_mapping = {
        "innovation": "multi_agent_coordination",
        "efficiency": "task_performance",
        "growth": "portfolio_management",
        "strategic_planning": "portfolio_management",
        "continuous_improvement": "outcome_patterns",
        "adaptive_learning": "outcome_patterns",
        "pattern_learning": "outcome_patterns",
        "critical_performance": "task_performance",
        "general": "claude_interactions"
    }
    
    # Get all JSON files in root directory
    root_files = list(research_path.glob("*.json"))
    logger.info(f"Found {len(root_files)} files to migrate")
    
    migrated = 0
    errors = 0
    
    for file_path in root_files:
        try:
            # Extract research type from filename
            filename = file_path.stem
            
            # Try to determine type from filename
            research_type = None
            for rtype in type_mapping.keys():
                if rtype in filename:
                    research_type = rtype
                    break
            
            # If not found in filename, try to read from file content
            if not research_type:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        research_type = data.get("type", "general")
                except:
                    research_type = "general"
            
            # Get target directory
            target_dir_name = type_mapping.get(research_type, "claude_interactions")
            target_dir = research_path / target_dir_name
            
            # Ensure target directory exists
            target_dir.mkdir(exist_ok=True)
            
            # Move file
            target_path = target_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            
            logger.info(f"Migrated {file_path.name} to {target_dir_name}/")
            migrated += 1
            
        except Exception as e:
            logger.error(f"Error migrating {file_path.name}: {e}")
            errors += 1
    
    logger.info(f"Migration complete: {migrated} files migrated, {errors} errors")
    
    # Print directory statistics
    logger.info("\nDirectory statistics after migration:")
    for subdir in ["task_performance", "claude_interactions", 
                   "multi_agent_coordination", "outcome_patterns", 
                   "portfolio_management"]:
        subdir_path = research_path / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.glob("*.json")))
            logger.info(f"  {subdir}: {file_count} files")


if __name__ == "__main__":
    migrate_research_files()