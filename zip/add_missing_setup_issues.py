#!/usr/bin/env python3
"""
Add missing initial setup issues to existing local repositories.

This script retroactively creates the initial setup issues for repositories
that were created before the issue creation bug was fixed.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.project_creator import ProjectCreator
from scripts.local_github_simulator import LocalRepository, LocalGitHubSimulator
from scripts.task_manager import TaskManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def add_setup_issues_to_repository(repo_name: str, org: str = "CodeWebMobile-AI"):
    """Add initial setup issues to a single repository.
    
    Args:
        repo_name: Name of the repository
        org: Organization name (default: CodeWebMobile-AI)
    """
    logger.info(f"Processing repository: {org}/{repo_name}")
    
    # Path to repository - must be absolute for LocalGitHubSimulator
    repo_path = Path(f"local_repos/repositories/{org}/{repo_name}").resolve()
    if not repo_path.exists():
        logger.error(f"Repository not found: {repo_path}")
        return False
    
    # Check if issues already exist
    issues_dir = repo_path / "issues"
    if issues_dir.exists() and any(issues_dir.glob("issue-*.json")):
        existing_issues = list(issues_dir.glob("issue-*.json"))
        logger.info(f"Repository already has {len(existing_issues)} issues, skipping")
        return False
    
    # Load repository metadata
    metadata_path = repo_path / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract project details from metadata
    project_data = metadata.get('project_data', {})
    
    # Build details dictionary matching what _create_initial_issues expects
    details = {
        'name': repo_name,
        'description': metadata.get('description', ''),
        'problem_statement': project_data.get('problem_solved', ''),
        'target_audience': project_data.get('target_audience', ''),
        'initial_features': project_data.get('key_features', []),
        'architecture': metadata.get('architecture', {}),
        'metadata': metadata
    }
    
    # Create LocalRepository object
    repo = LocalRepository(
        name=repo_name,
        full_name=f"{org}/{repo_name}",
        description=metadata.get('description', ''),
        path=str(repo_path)
    )
    
    # Initialize ProjectCreator with task manager
    logger.info("Initializing ProjectCreator...")
    project_creator = ProjectCreator(github_token=None, ai_brain=None)
    
    try:
        # Call _create_initial_issues
        logger.info(f"Creating initial setup issues for {repo_name}...")
        issue_numbers = await project_creator._create_initial_issues(repo, details)
        
        if issue_numbers:
            logger.info(f"✅ Successfully created {len(issue_numbers)} issues: {issue_numbers}")
            
            # Update metadata statistics if needed
            if 'statistics' not in metadata:
                metadata['statistics'] = {}
            if 'issues' not in metadata['statistics']:
                metadata['statistics']['issues'] = {
                    'total': 0,
                    'open': 0,
                    'closed': 0
                }
            
            metadata['statistics']['issues']['total'] = len(issue_numbers)
            metadata['statistics']['issues']['open'] = len(issue_numbers)
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        else:
            logger.warning("No issues were created")
            return False
            
    except Exception as e:
        logger.error(f"Error creating issues: {e}")
        import traceback
        traceback.print_exc()
        return False


async def process_all_repositories(org: str = "CodeWebMobile-AI"):
    """Process all repositories in the organization."""
    org_path = Path(f"local_repos/repositories/{org}")
    if not org_path.exists():
        logger.error(f"Organization path not found: {org_path}")
        return
    
    # Get all repositories
    repos = [d.name for d in org_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(repos)} repositories")
    
    success_count = 0
    for repo_name in repos:
        success = await add_setup_issues_to_repository(repo_name, org)
        if success:
            success_count += 1
    
    logger.info(f"Processed {success_count}/{len(repos)} repositories successfully")


async def main():
    """Main function."""
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Process all repositories
            await process_all_repositories()
        else:
            # Process specific repository
            repo_name = sys.argv[1]
            await add_setup_issues_to_repository(repo_name)
    else:
        # Default: test with verisafe
        logger.info("Testing with verisafe repository...")
        await add_setup_issues_to_repository("verisafe")


if __name__ == "__main__":
    # Set environment for local mode
    os.environ['STORAGE_MODE'] = 'LOCAL'
    
    # Run the async main function
    asyncio.run(main())