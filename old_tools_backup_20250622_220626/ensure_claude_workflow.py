"""
Ensure Claude Workflow Tool
===========================

Checks if the Claude Code GitHub Actions workflow exists in repositories
and adds it if missing to enable @claude mentions in issues and PRs.
"""

import os
import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback

from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.state_manager import StateManager
from scripts.logger import logger

__description__ = "Ensure Claude Code GitHub Actions workflow exists in all managed repositories to enable @claude mentions"
__parameters__ = {
    'dry_run': {'type': 'boolean', 'required': False, 'description': 'If true, only check without making changes'},
    'repo_name': {'type': 'string', 'required': False, 'description': 'Specific repository to check (if not provided, checks all)'}
}
__examples__ = [
    {'dry_run': True},
    {'repo_name': 'reputation-ai'},
    {}
]

# Claude workflow content
CLAUDE_WORKFLOW_CONTENT = """name: Claude PR Action 

permissions:
  contents: write
  pull-requests: write
  issues: write
  id-token: write 

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]

jobs:
  claude-pr:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
    runs-on: ubuntu-latest
    env:
      AWS_REGION: us-west-2
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Generate GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v2
        with:
          app-id: ${{ secrets.APP_ID }}
          private-key: ${{ secrets.APP_PRIVATE_KEY }}

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: us-west-2

      - uses: ./.github/actions/claude-pr-action
        with:
          trigger_phrase: "@claude"
          timeout_minutes: "60"
          github_token: ${{ steps.app-token.outputs.token }}
          use_bedrock: "true"
          model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
"""

async def ensure_claude_workflow(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure Claude Code GitHub Actions workflow exists in repositories.
    
    This enables @claude mentions in issues and pull requests.
    """
    try:
        dry_run = kwargs.get('dry_run', False)
        specific_repo = kwargs.get('repo_name')
        
        # Initialize components
        mcp_creator = MCPGitHubIssueCreator()
        state_manager = StateManager()
        state_manager.load_state()
        state = state_manager.get_state()
        
        # Get repositories to check
        projects = state.get('projects', {})
        if specific_repo:
            # Check only specific repository
            if specific_repo in projects:
                repos_to_check = {specific_repo: projects[specific_repo]}
            else:
                # Try with organization prefix
                for key, project in projects.items():
                    if project.get('name') == specific_repo:
                        repos_to_check = {key: project}
                        break
                else:
                    return {
                        "success": False,
                        "error": f"Repository '{specific_repo}' not found in managed projects"
                    }
        else:
            # Check all repositories
            repos_to_check = projects
        
        results = {
            "checked": 0,
            "already_exists": 0,
            "added": 0,
            "failed": 0,
            "repositories": {}
        }
        
        for repo_key, project in repos_to_check.items():
            repo_name = project.get('full_name')
            if not repo_name:
                logger.warning(f"Skipping {repo_key}: No full_name found")
                continue
                
            results["checked"] += 1
            
            try:
                # Check if workflow file exists
                logger.info(f"Checking {repo_name} for Claude workflow...")
                
                # Use gh CLI to check if file exists
                import subprocess
                check_cmd = f"gh api repos/{repo_name}/contents/.github/workflows/claude.yml --silent"
                check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                
                if check_result.returncode == 0:
                    # File exists
                    results["already_exists"] += 1
                    results["repositories"][repo_name] = {
                        "status": "already_exists",
                        "message": "Claude workflow already present"
                    }
                    logger.info(f"✓ {repo_name}: Claude workflow already exists")
                else:
                    # File doesn't exist, create it
                    if dry_run:
                        results["repositories"][repo_name] = {
                            "status": "would_add",
                            "message": "Would add Claude workflow (dry run)"
                        }
                        logger.info(f"→ {repo_name}: Would add Claude workflow (dry run)")
                    else:
                        # Create directories first
                        default_branch = project.get('default_branch', 'main')
                        
                        # Create the workflow file using gh CLI
                        import base64
                        import json
                        
                        # Encode content to base64
                        content_b64 = base64.b64encode(CLAUDE_WORKFLOW_CONTENT.encode()).decode()
                        
                        # Create the file
                        create_data = {
                            "message": "Add Claude Code GitHub Actions workflow\n\nThis enables @claude mentions in issues and pull requests",
                            "content": content_b64,
                            "branch": default_branch
                        }
                        
                        create_cmd = f"gh api repos/{repo_name}/contents/.github/workflows/claude.yml -X PUT --input -"
                        create_result = subprocess.run(
                            create_cmd, 
                            shell=True, 
                            input=json.dumps(create_data),
                            capture_output=True, 
                            text=True
                        )
                        
                        if create_result.returncode == 0:
                            results["added"] += 1
                            results["repositories"][repo_name] = {
                                "status": "added",
                                "message": "Claude workflow added successfully"
                            }
                            logger.info(f"✓ {repo_name}: Claude workflow added successfully")
                        else:
                            results["failed"] += 1
                            error_msg = create_result.stderr or create_result.stdout or 'Failed to create file'
                            results["repositories"][repo_name] = {
                                "status": "failed",
                                "error": error_msg
                            }
                            logger.error(f"✗ {repo_name}: Failed to add workflow - {error_msg}")
                            
            except Exception as e:
                results["failed"] += 1
                results["repositories"][repo_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {repo_name}: Error - {e}")
        
        # Summary
        summary = f"""
Claude Workflow Check Complete:
- Repositories checked: {results['checked']}
- Already have workflow: {results['already_exists']}
- Workflow added: {results['added']}
- Failed: {results['failed']}
"""
        
        if dry_run:
            summary += "\n(DRY RUN - No changes were made)"
            
        logger.info(summary)
        
        return {
            "success": True,
            "summary": summary.strip(),
            "dry_run": dry_run,
            **results
        }
        
    except Exception as e:
        error_msg = f"Error ensuring Claude workflow: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

# For testing
if __name__ == "__main__":
    import sys
    
    async def test():
        # Test dry run
        print("Testing dry run...")
        result = await ensure_claude_workflow(dry_run=True)
        print(f"Result: {result}")
        
        # Test specific repo if provided
        if len(sys.argv) > 1:
            repo = sys.argv[1]
            print(f"\nTesting specific repo: {repo}")
            result = await ensure_claude_workflow(repo_name=repo, dry_run=True)
            print(f"Result: {result}")
    
    asyncio.run(test())

# Export for tool system
ensure_claude_workflow_tool = ensure_claude_workflow