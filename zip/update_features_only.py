#!/usr/bin/env python3
"""
Update only the Key Features section in existing ARCHITECTURE.md files.
This is a focused script that just updates the features to include detailed descriptions.
"""

import os
import json
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from scripts.ai_brain_factory import AIBrainFactory
from scripts.http_ai_client import HTTPAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureUpdater:
    """Updates only the features section in architecture files."""
    
    def __init__(self, ai_brain):
        """Initialize with AI brain for generating detailed descriptions."""
        self.ai_brain = ai_brain
        self.local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/repositories/CodeWebMobile-AI")
        
    async def update_all_projects(self):
        """Find and update features in all projects."""
        logger.info("🔍 Scanning for projects to update...")
        
        projects_updated = 0
        projects_skipped = 0
        
        # Find all project directories
        for project_dir in self.local_repos_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                result = await self.update_project_features(project_dir.name)
                if result == "updated":
                    projects_updated += 1
                elif result == "skipped":
                    projects_skipped += 1
                    
        logger.info(f"\n📊 Summary: Updated {projects_updated} projects, skipped {projects_skipped} projects")
        
    async def update_project_features(self, project_name: str) -> str:
        """Update features for a specific project."""
        project_dir = self.local_repos_path / project_name
        architecture_file = project_dir / "ARCHITECTURE.md"
        metadata_file = project_dir / "metadata.json"
        
        if not architecture_file.exists() or not metadata_file.exists():
            logger.warning(f"  ⚠️  Missing files for {project_name}")
            return "error"
            
        logger.info(f"\n📂 Processing: {project_name}")
        
        # Read current architecture
        content = architecture_file.read_text()
        
        # Check if already has detailed features
        if "Technical Description:" in content or "Architecture Impact:" in content:
            logger.info(f"  ⏭️  Already has detailed features")
            return "skipped"
            
        # Extract simple features list
        features = self._extract_simple_features(content)
        if not features:
            logger.warning(f"  ⚠️  No features found")
            return "error"
            
        logger.info(f"  📋 Found {len(features)} features to update")
        
        # Load project metadata for context
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Generate detailed features
        detailed_features = await self._generate_detailed_features(
            project_name, 
            features,
            metadata
        )
        
        if not detailed_features:
            logger.error(f"  ❌ Failed to generate detailed features")
            return "error"
            
        # Update the architecture file
        new_content = self._replace_features_section(content, detailed_features)
        
        # Backup and save
        backup_path = architecture_file.with_suffix('.md.backup')
        architecture_file.rename(backup_path)
        architecture_file.write_text(new_content)
        
        logger.info(f"  ✅ Updated with detailed features")
        logger.info(f"  💾 Backup saved as {backup_path.name}")
        
        return "updated"
        
    def _extract_simple_features(self, content: str) -> List[str]:
        """Extract feature names from simple numbered list."""
        features = []
        
        # Find the features section
        match = re.search(r'### AI-Generated Features from Market Research.*?\n\n((?:\d+\.\s+\*\*[^*]+\*\*\n?)+)', content, re.DOTALL)
        if match:
            features_text = match.group(1)
            # Extract feature names from numbered list
            for line in features_text.strip().split('\n'):
                feature_match = re.match(r'\d+\.\s+\*\*([^*]+)\*\*', line)
                if feature_match:
                    features.append(feature_match.group(1).strip())
                    
        return features
        
    async def _generate_detailed_features(self, project_name: str, features: List[str], metadata: Dict) -> str:
        """Generate detailed feature descriptions using AI."""
        
        # Get project context
        project_desc = metadata.get('project', {}).get('description', '')
        market_research = metadata.get('market_research', {})
        
        prompt = f"""
        Generate detailed technical descriptions for these features in a {project_name} project.
        
        Project: {project_name}
        Description: {project_desc}
        Market Problem: {market_research.get('problem_description', project_desc)}
        
        Features to detail:
        {json.dumps(features, indent=2)}
        
        For EACH feature, provide:
        1. Feature Name (use exact name from list)
        2. Technical Description (2-3 sentences on HOW it works technically)
        3. Architecture Impact (what services/components are needed)
        4. Implementation Complexity (Simple/Medium/Complex)
        5. Key Components (list 3-5 technical components)
        6. Data Flow (brief description of data movement)
        7. Integration Points (how it connects with other features)
        
        Format as markdown with each feature as a subsection.
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            return response.get('content', '')
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return ""
            
    def _replace_features_section(self, content: str, detailed_features: str) -> str:
        """Replace simple features list with detailed descriptions."""
        
        # Find the features section
        pattern = r'(### AI-Generated Features from Market Research.*?\n\n)(?:\d+\.\s+\*\*[^*]+\*\*\n?)+'
        
        # Create replacement text
        replacement = r'\1' + detailed_features.strip() + '\n'
        
        # Replace the section
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        return new_content


async def main():
    """Main function to update features."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update architecture features with detailed descriptions')
    parser.add_argument('--project', type=str, help='Update specific project by name')
    parser.add_argument('--list', action='store_true', help='List all projects')
    args = parser.parse_args()
    
    # Initialize AI brain
    logger.info("🧠 Initializing AI brain...")
    ai_brain = AIBrainFactory.create_for_development()
    
    # Create updater
    updater = FeatureUpdater(ai_brain)
    
    if args.list:
        # List all projects
        logger.info("\n📁 Available projects:")
        for project_dir in updater.local_repos_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                logger.info(f"  - {project_dir.name}")
        return
        
    if args.project:
        # Update specific project
        result = await updater.update_project_features(args.project)
        if result == "updated":
            logger.info(f"\n✅ Successfully updated {args.project}")
        elif result == "skipped":
            logger.info(f"\n⏭️  {args.project} already has detailed features")
        else:
            logger.error(f"\n❌ Failed to update {args.project}")
    else:
        # Update all projects
        await updater.update_all_projects()
        

if __name__ == "__main__":
    asyncio.run(main())