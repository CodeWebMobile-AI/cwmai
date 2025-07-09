#!/usr/bin/env python3
"""
Update existing projects' ARCHITECTURE.md files to include detailed feature descriptions.
This script retrofits the architecture fix to already-created projects.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArchitectureUpdater:
    """Updates existing architecture files with detailed feature descriptions."""
    
    def __init__(self, ai_brain):
        """Initialize with AI brain for generating detailed descriptions."""
        self.ai_brain = ai_brain
        self.project_creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
        self.local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/repositories/CodeWebMobile-AI")
        
    async def update_all_projects(self):
        """Find and update all projects with simple feature lists."""
        logger.info("🔍 Scanning for projects to update...")
        
        projects_updated = 0
        projects_skipped = 0
        
        # Find all project directories
        for project_dir in self.local_repos_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                architecture_file = project_dir / "ARCHITECTURE.md"
                metadata_file = project_dir / "metadata.json"
                
                if architecture_file.exists() and metadata_file.exists():
                    logger.info(f"\n📂 Processing project: {project_dir.name}")
                    
                    # Check if update is needed
                    if await self._needs_update(architecture_file):
                        logger.info(f"  ✅ Update needed for {project_dir.name}")
                        success = await self._update_project(project_dir, architecture_file, metadata_file)
                        if success:
                            projects_updated += 1
                        else:
                            logger.error(f"  ❌ Failed to update {project_dir.name}")
                    else:
                        logger.info(f"  ⏭️  Skipping {project_dir.name} - already has detailed features")
                        projects_skipped += 1
                        
        logger.info(f"\n📊 Summary: Updated {projects_updated} projects, skipped {projects_skipped} projects")
        
    async def _needs_update(self, architecture_file: Path) -> bool:
        """Check if architecture file needs updating based on feature format."""
        try:
            content = architecture_file.read_text()
            
            # Look for signs of simple feature list vs detailed descriptions
            # Simple format has numbered list items like "1. **Feature Name**"
            # Detailed format has subsections with technical descriptions
            
            if "## Key Features" not in content:
                return False
                
            # Extract the Key Features section
            features_start = content.find("## Key Features")
            if features_start == -1:
                return False
                
            features_section = content[features_start:content.find("## ", features_start + 1)]
            
            # Check if features have technical descriptions
            has_technical_desc = any(phrase in features_section for phrase in [
                "Technical Description:",
                "Architecture Impact:",
                "Implementation Complexity:",
                "Key Components:",
                "Data Flow:",
                "Integration Points:"
            ])
            
            # If no technical descriptions found, it needs update
            return not has_technical_desc
            
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return False
            
    async def _update_project(self, project_dir: Path, architecture_file: Path, metadata_file: Path) -> bool:
        """Update a single project's architecture with detailed features."""
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Extract project details and features
            project_name = project_dir.name
            project_metadata = metadata.get('project', {})
            documentation = metadata.get('documentation', {})
            readme_data = documentation.get('readme', {})
            architecture_data = documentation.get('architecture', {})
            
            # Get features from README (which has descriptions) or architecture
            features_with_desc = readme_data.get('key_features', [])
            features_simple = architecture_data.get('key_features', [])
            
            if not features_simple and not features_with_desc:
                logger.warning(f"  ⚠️  No features found for {project_name}")
                return False
                
            # Prepare project details for AI generation
            project_details = {
                'name': project_name,
                'description': project_metadata.get('description', ''),
                'problem_statement': readme_data.get('overview', project_metadata.get('description', '')),
                'target_audience': 'Users who need ' + project_metadata.get('description', ''),
                'initial_features': features_simple if features_simple else [f['name'] for f in features_with_desc if 'name' in f],
                'metadata': metadata.get('market_research', {})
            }
            
            logger.info(f"  🤖 Generating detailed architecture for {len(project_details['initial_features'])} features...")
            
            # Generate new architecture with detailed features
            new_architecture = await self.project_creator._generate_structured_architecture(project_details)
            
            # Convert to markdown
            architecture_markdown = self.project_creator._transform_structured_architecture_to_markdown(new_architecture)
            
            # Backup original file
            backup_path = architecture_file.with_suffix('.md.backup')
            architecture_file.rename(backup_path)
            logger.info(f"  💾 Original backed up to {backup_path.name}")
            
            # Write new architecture
            architecture_file.write_text(architecture_markdown)
            logger.info(f"  ✅ Updated {architecture_file.name}")
            
            # Update metadata with new architecture structure
            metadata['documentation']['architecture'] = new_architecture
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"  ✅ Updated metadata.json")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error updating project: {e}")
            import traceback
            traceback.print_exc()
            
            # Restore backup if it exists
            backup_path = architecture_file.with_suffix('.md.backup')
            if backup_path.exists() and not architecture_file.exists():
                backup_path.rename(architecture_file)
                logger.info(f"  🔄 Restored original file from backup")
                
            return False
            
    async def update_specific_project(self, project_name: str):
        """Update a specific project by name."""
        project_dir = self.local_repos_path / project_name
        
        if not project_dir.exists():
            logger.error(f"Project '{project_name}' not found")
            return False
            
        architecture_file = project_dir / "ARCHITECTURE.md"
        metadata_file = project_dir / "metadata.json"
        
        if not architecture_file.exists():
            logger.error(f"ARCHITECTURE.md not found for {project_name}")
            return False
            
        if not metadata_file.exists():
            logger.error(f"metadata.json not found for {project_name}")
            return False
            
        logger.info(f"📂 Updating project: {project_name}")
        
        if await self._needs_update(architecture_file):
            return await self._update_project(project_dir, architecture_file, metadata_file)
        else:
            logger.info(f"Project {project_name} already has detailed features")
            return True


async def main():
    """Main function to update architecture files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update existing project architecture files with detailed features')
    parser.add_argument('--project', type=str, help='Update specific project by name')
    parser.add_argument('--dry-run', action='store_true', help='Check which projects need updates without modifying')
    args = parser.parse_args()
    
    # Initialize AI brain
    logger.info("🧠 Initializing AI brain...")
    ai_brain = AIBrainFactory.create_for_development()
    
    # Create updater
    updater = ArchitectureUpdater(ai_brain)
    
    if args.dry_run:
        logger.info("🔍 Running in dry-run mode - no files will be modified")
        # TODO: Implement dry-run mode
        logger.warning("Dry-run mode not yet implemented")
        return
        
    if args.project:
        # Update specific project
        success = await updater.update_specific_project(args.project)
        if success:
            logger.info(f"✅ Successfully updated {args.project}")
        else:
            logger.error(f"❌ Failed to update {args.project}")
    else:
        # Update all projects
        await updater.update_all_projects()
        

if __name__ == "__main__":
    asyncio.run(main())