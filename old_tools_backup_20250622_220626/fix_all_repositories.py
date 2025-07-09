"""
Fix All Repositories Tool
=========================

Comprehensive tool that runs all repository fixes: Laravel setup, missing configs, and architecture updates.
This tool orchestrates the execution of three other tools to ensure all managed repositories are properly configured.
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, List
from pathlib import Path
import traceback

from scripts.state_manager import StateManager
from scripts.logger import logger
from scripts.github_issue_creator import GitHubIssueCreator

# Import the three tools we need to run
from scripts.custom_tools.setup_laravel_react_starter import setup_laravel_react_starter_tool
from scripts.custom_tools.add_missing_configs import add_missing_configs_tool
from scripts.custom_tools.update_architecture_docs import update_architecture_docs_tool
from scripts.custom_tools.check_laravel_react_completeness import check_laravel_react_completeness_tool

__description__ = "Comprehensively fix all managed repositories: setup Laravel starter kits, add missing configs, and update architecture docs"
__parameters__ = {
    'dry_run': {'type': 'boolean', 'required': False, 'description': 'If true, only check without making changes'},
    'create_summary_issue': {'type': 'boolean', 'required': False, 'description': 'Create a GitHub issue with the summary of changes'}
}
__examples__ = [
    {'dry_run': True},
    {'create_summary_issue': True},
    {}
]

async def fix_all_repositories(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix all repositories by running the three repair tools in sequence.
    """
    try:
        dry_run = kwargs.get('dry_run', False)
        create_summary_issue = kwargs.get('create_summary_issue', False)
        
        logger.info("Starting comprehensive repository fix process...")
        
        results = {
            "success": True,
            "dry_run": dry_run,
            "steps": {}
        }
        
        # Step 1: Check completeness to identify which repos need fixing
        logger.info("\n=== Step 1: Checking repository completeness ===")
        completeness_result = await check_laravel_react_completeness_tool()
        results["steps"]["completeness_check"] = completeness_result
        
        if not completeness_result.get("success"):
            logger.error("Failed to check repository completeness")
            return results
        
        # Identify repositories that need Laravel setup
        repos_needing_setup = []
        repos_needing_configs = []
        
        for repo_name, repo_info in completeness_result.get("repositories", {}).items():
            if repo_info.get("status") == "missing_starter_kit":
                repos_needing_setup.append(repo_name.split('/')[-1])  # Get just the repo name
            elif repo_info.get("status") == "partial_starter_kit":
                # Check if it's missing configs specifically
                missing_files = repo_info.get("missing_files", [])
                if any("tailwind" in f.lower() or "postcss" in f.lower() for f in missing_files):
                    repos_needing_configs.append(repo_name.split('/')[-1])
        
        # Step 2: Setup Laravel React starter for empty repositories
        if repos_needing_setup and not dry_run:
            logger.info(f"\n=== Step 2: Setting up Laravel starter for {len(repos_needing_setup)} repositories ===")
            setup_results = []
            
            for repo_name in repos_needing_setup:
                logger.info(f"Setting up {repo_name}...")
                result = await setup_laravel_react_starter_tool(
                    repo_name=repo_name,
                    preserve_docs=True
                )
                setup_results.append({
                    "repo": repo_name,
                    "result": result
                })
                
                if result.get("success"):
                    logger.info(f"✓ Successfully set up {repo_name}")
                else:
                    logger.error(f"✗ Failed to set up {repo_name}: {result.get('error')}")
            
            results["steps"]["laravel_setup"] = {
                "repos_processed": len(setup_results),
                "details": setup_results
            }
        else:
            logger.info("\n=== Step 2: Skipping Laravel setup (dry run or no repos need it) ===")
            results["steps"]["laravel_setup"] = {
                "repos_needing_setup": repos_needing_setup,
                "skipped": True,
                "reason": "dry_run" if dry_run else "no_repos_need_setup"
            }
        
        # Step 3: Add missing configs
        logger.info("\n=== Step 3: Adding missing Tailwind/PostCSS configs ===")
        configs_result = await add_missing_configs_tool(dry_run=dry_run)
        results["steps"]["add_configs"] = configs_result
        
        # Step 4: Update architecture documentation
        logger.info("\n=== Step 4: Updating ARCHITECTURE.md files ===")
        arch_result = await update_architecture_docs_tool(dry_run=dry_run)
        results["steps"]["update_architecture"] = arch_result
        
        # Generate summary
        summary_lines = [
            "# Repository Fix Summary\n",
            f"**Mode**: {'Dry Run' if dry_run else 'Live Changes'}\n",
            "## Completeness Check",
            f"- Repositories checked: {completeness_result.get('checked', 0)}",
            f"- Missing starter kit: {completeness_result.get('missing_starter_kit', 0)}",
            f"- Partial starter kit: {completeness_result.get('partial_starter_kit', 0)}",
            f"- Complete: {completeness_result.get('complete', 0)}\n"
        ]
        
        if not dry_run and repos_needing_setup:
            summary_lines.extend([
                "## Laravel Starter Setup",
                f"- Repositories processed: {len(repos_needing_setup)}",
                f"- Successful: {sum(1 for r in results['steps'].get('laravel_setup', {}).get('details', []) if r['result'].get('success'))}",
                f"- Failed: {sum(1 for r in results['steps'].get('laravel_setup', {}).get('details', []) if not r['result'].get('success'))}\n"
            ])
        
        summary_lines.extend([
            "## Missing Configs",
            f"- Repositories checked: {configs_result.get('checked', 0)}",
            f"- Updated: {configs_result.get('updated', 0)}",
            f"- Already complete: {configs_result.get('already_complete', 0)}\n",
            "## Architecture Documentation",
            f"- Repositories checked: {arch_result.get('checked', 0)}",
            f"- Updated: {arch_result.get('updated', 0)}",
            f"- Created: {arch_result.get('created', 0)}",
            f"- Already updated: {arch_result.get('already_updated', 0)}\n"
        ])
        
        # Add details of repositories that need attention
        if completeness_result.get('missing_starter_kit', 0) > 0:
            summary_lines.append("## Repositories Missing Starter Kit")
            for repo_name, info in completeness_result.get('repositories', {}).items():
                if info.get('status') == 'missing_starter_kit':
                    summary_lines.append(f"- {repo_name}")
            summary_lines.append("")
        
        summary = '\n'.join(summary_lines)
        results["summary"] = summary
        
        logger.info("\n" + summary)
        
        # Create summary issue if requested
        if create_summary_issue and not dry_run:
            try:
                github_creator = GitHubIssueCreator()
                issue_title = "Repository Infrastructure Update Summary"
                issue_body = summary + "\n\n---\n*This issue was automatically generated by the fix_all_repositories tool.*"
                
                # Create issue in the main cwmai repository
                issue_result = github_creator.create_issue(
                    repo_name="CodeWebMobile-AI/cwmai",
                    title=issue_title,
                    body=issue_body,
                    labels=["automated", "infrastructure"]
                )
                
                if issue_result["success"]:
                    results["summary_issue_url"] = issue_result.get("issue_url")
                    logger.info(f"Created summary issue: {issue_result.get('issue_url')}")
            except Exception as e:
                logger.error(f"Failed to create summary issue: {e}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error in fix_all_repositories: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

# Export for tool system
fix_all_repositories_tool = fix_all_repositories

if __name__ == "__main__":
    import sys
    
    async def test():
        # Test dry run
        print("Testing comprehensive repository fix (dry run)...")
        result = await fix_all_repositories(dry_run=True)
        print(f"Summary:\n{result.get('summary', 'No summary generated')}")
    
    asyncio.run(test())