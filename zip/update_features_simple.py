#!/usr/bin/env python3
"""
Simple script to update architecture features with detailed descriptions.
"""

import os
import json
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from scripts.http_ai_client import HTTPAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleFeatureUpdater:
    """Updates features section in architecture files."""
    
    def __init__(self):
        """Initialize with AI client."""
        self.ai_client = HTTPAIClient()
        self.local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/repositories/CodeWebMobile-AI")
        
    async def list_projects(self):
        """List all available projects."""
        logger.info("\n📁 Available projects:")
        projects = []
        for project_dir in self.local_repos_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                arch_file = project_dir / "ARCHITECTURE.md"
                if arch_file.exists():
                    # Check if needs update
                    content = arch_file.read_text()
                    needs_update = "technical description:" not in content.lower()
                    status = "❌ needs update" if needs_update else "✅ already detailed"
                    projects.append((project_dir.name, status))
                    
        for name, status in sorted(projects):
            logger.info(f"  - {name:<20} {status}")
            
    async def update_project(self, project_name: str):
        """Update a specific project's features."""
        project_dir = self.local_repos_path / project_name
        architecture_file = project_dir / "ARCHITECTURE.md"
        metadata_file = project_dir / "metadata.json"
        
        if not architecture_file.exists():
            logger.error(f"❌ ARCHITECTURE.md not found for {project_name}")
            return False
            
        logger.info(f"\n📂 Processing: {project_name}")
        
        # Read current architecture
        content = architecture_file.read_text()
        
        # Check if already has detailed features
        if "technical description:" in content.lower():
            logger.info(f"  ✅ Already has detailed features")
            return True
            
        # Extract simple features list
        features = self._extract_simple_features(content)
        if not features:
            logger.warning(f"  ⚠️  No features found to update")
            return False
            
        logger.info(f"  📋 Found {len(features)} features: {', '.join(features)}")
        
        # Load metadata for context
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
        # Generate detailed features
        logger.info(f"  🤖 Generating detailed descriptions...")
        detailed_section = await self._generate_detailed_features(
            project_name,
            features,
            metadata
        )
        
        if not detailed_section:
            logger.error(f"  ❌ Failed to generate detailed features")
            return False
            
        # Replace the features section
        new_content = self._replace_features_section(content, features, detailed_section)
        
        # Backup and save
        backup_path = architecture_file.with_suffix('.md.backup')
        architecture_file.rename(backup_path)
        architecture_file.write_text(new_content)
        
        logger.info(f"  ✅ Updated successfully!")
        logger.info(f"  💾 Backup saved as {backup_path.name}")
        
        return True
        
    def _extract_simple_features(self, content: str) -> List[str]:
        """Extract feature names from simple numbered list."""
        features = []
        
        # Find lines like "1. **Feature Name**"
        lines = content.split('\n')
        in_features = False
        
        for line in lines:
            if "AI-Generated Features from Market Research" in line:
                in_features = True
                continue
                
            if in_features:
                # Stop at next section
                if line.startswith('##'):
                    break
                    
                # Extract feature from numbered list
                match = re.match(r'\d+\.\s+\*\*([^*]+)\*\*', line)
                if match:
                    features.append(match.group(1).strip())
                    
        return features
        
    async def _generate_detailed_features(self, project_name: str, features: List[str], metadata: Dict) -> str:
        """Generate detailed feature descriptions."""
        
        # Get project context
        project_info = metadata.get('project', {})
        project_desc = project_info.get('description', f'{project_name} project')
        
        # Build prompt
        prompt = f"""
You are updating the architecture documentation for {project_name}.

Project Description: {project_desc}

Update these features with detailed technical descriptions:
{chr(10).join(f'{i+1}. {feature}' for i, feature in enumerate(features))}

For EACH feature above, create a subsection with this EXACT format:

### [Feature Name Exactly As Listed]

**Technical Description**: [2-3 sentences explaining HOW this feature works technically, what algorithms or processes it uses]

**Architecture Impact**: [What new components, services, or infrastructure this feature requires]

**Implementation Complexity**: [Choose one: Simple / Medium / Complex]

**Key Components**:
- [Component 1 - be specific, e.g., "Laravel Job Queue for async processing"]
- [Component 2 - e.g., "Redis cache for temporary data storage"]
- [Component 3 - e.g., "React component for UI rendering"]
- [Add 2-3 more relevant components]

**Data Flow**: [1-2 sentences describing how data moves through the system for this feature]

**Integration Points**: [How this feature connects with other parts of the system or other features]

---

Create all {len(features)} feature sections following this exact format.
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            # The response is a dict with 'content' key directly
            if response and 'content' in response:
                return response['content']
            else:
                logger.error(f"AI response error: {response}")
                return ""
                
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return ""
            
    def _replace_features_section(self, content: str, old_features: List[str], new_section: str) -> str:
        """Replace the simple features list with detailed descriptions."""
        
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Start of features section
            if "AI-Generated Features from Market Research" in line:
                new_lines.append(line)
                # Skip the description line
                if i + 1 < len(lines):
                    new_lines.append(lines[i + 1])
                    i += 1
                    
                # Skip empty line
                if i + 1 < len(lines) and lines[i + 1].strip() == "":
                    new_lines.append("")
                    i += 1
                    
                # Now skip the old numbered list
                skip_mode = True
                i += 1
                continue
                
            # In skip mode, skip numbered features
            if skip_mode:
                if re.match(r'\d+\.\s+\*\*[^*]+\*\*', line):
                    i += 1
                    continue
                else:
                    # End of numbered list, insert new content
                    skip_mode = False
                    new_lines.append(new_section)
                    new_lines.append("")  # Extra line for spacing
                    
            # Normal line processing
            new_lines.append(line)
            i += 1
            
        return '\n'.join(new_lines)


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update architecture features')
    parser.add_argument('--project', type=str, help='Update specific project')
    parser.add_argument('--list', action='store_true', help='List all projects')
    parser.add_argument('--all', action='store_true', help='Update all projects')
    args = parser.parse_args()
    
    updater = SimpleFeatureUpdater()
    
    if args.list:
        await updater.list_projects()
    elif args.project:
        success = await updater.update_project(args.project)
        if success:
            logger.info(f"\n✅ Successfully updated {args.project}")
        else:
            logger.error(f"\n❌ Failed to update {args.project}")
    elif args.all:
        # Update all projects that need it
        for project_dir in updater.local_repos_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                arch_file = project_dir / "ARCHITECTURE.md"
                if arch_file.exists():
                    content = arch_file.read_text()
                    if "Technical Description:" not in content:
                        await updater.update_project(project_dir.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())