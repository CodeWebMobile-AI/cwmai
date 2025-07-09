"""
Check Laravel React Completeness Tool
=====================================

Checks if repositories have the complete Laravel React starter kit files
and identifies which are missing the actual codebase or have outdated ARCHITECTURE.md files.
"""

import os
import asyncio
import logging
import subprocess
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import traceback

from scripts.state_manager import StateManager
from scripts.logger import logger

__description__ = "Check if managed repositories have complete Laravel React starter kit files and updated ARCHITECTURE.md"
__parameters__ = {
    'fix_missing': {'type': 'boolean', 'required': False, 'description': 'If true, create issues for repositories missing files'},
    'repo_name': {'type': 'string', 'required': False, 'description': 'Specific repository to check (if not provided, checks all)'}
}
__examples__ = [
    {'fix_missing': False},
    {'repo_name': 'reputation-ai'},
    {}
]

# Essential files that should exist in a Laravel React starter kit
ESSENTIAL_FILES = {
    # Laravel core files
    'composer.json': 'PHP dependencies configuration',
    'artisan': 'Laravel command-line tool',
    '.env.example': 'Environment variables template',
    'package.json': 'Node.js dependencies configuration',
    
    # Configuration files
    'vite.config.ts': 'Vite build configuration (or vite.config.js)',
    'tsconfig.json': 'TypeScript configuration',
    'tailwind.config.js': 'Tailwind CSS configuration',
    'postcss.config.js': 'PostCSS configuration',
    'phpunit.xml': 'PHPUnit testing configuration',
    
    # Laravel directories (we'll check if they exist)
    'app': 'Laravel application directory',
    'config': 'Laravel configuration directory',
    'database': 'Database migrations and seeders',
    'public': 'Public assets directory',
    'resources': 'Resources directory (views, js, css)',
    'routes': 'Application routes',
    'storage': 'Storage directory',
    'tests': 'Test files directory',
}

# Files that indicate a properly set up React frontend
REACT_INDICATORS = [
    'resources/js/app.tsx',
    'resources/js/app.jsx',
    'resources/js/Pages',
    'resources/js/Components',
]

async def check_file_exists(repo_name: str, file_path: str) -> bool:
    """Check if a file exists in the repository using gh CLI."""
    cmd = f"gh api repos/{repo_name}/contents/{file_path} --silent"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

async def get_repo_files(repo_name: str) -> List[str]:
    """Get list of files in repository root."""
    cmd = f"gh api repos/{repo_name}/contents --jq '.[].path'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    return []

async def check_architecture_file_quality(repo_name: str) -> Dict[str, Any]:
    """Check if ARCHITECTURE.md exists and whether it's the old or new format."""
    cmd = f"gh api repos/{repo_name}/contents/ARCHITECTURE.md --jq '.content' | base64 -d"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {"exists": False, "format": "missing"}
    
    content = result.stdout
    
    # Check for indicators of new vs old format
    # New format typically has more detailed sections and specific patterns
    new_format_indicators = [
        "## System Overview",
        "## Technical Stack", 
        "## Key Components",
        "## API Design",
        "## Database Schema",
        "## Security Considerations",
        "## Deployment Strategy"
    ]
    
    # Count how many new format indicators are present
    indicator_count = sum(1 for indicator in new_format_indicators if indicator in content)
    
    # Also check content length as new format is typically more comprehensive
    content_length = len(content)
    
    if indicator_count >= 4 and content_length > 2000:
        return {"exists": True, "format": "new", "quality": "good"}
    elif indicator_count >= 2:
        return {"exists": True, "format": "partial", "quality": "needs_update"}
    else:
        return {"exists": True, "format": "old", "quality": "outdated"}

async def check_laravel_react_completeness(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if repositories have complete Laravel React starter kit files.
    """
    try:
        fix_missing = kwargs.get('fix_missing', False)
        specific_repo = kwargs.get('repo_name')
        
        # Initialize components with absolute path
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        state_file_path = os.path.join(project_root, "system_state.json")
        state_manager = StateManager(local_path=state_file_path)
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
            "complete": 0,
            "missing_starter_kit": 0,
            "partial_starter_kit": 0,
            "outdated_architecture": 0,
            "repositories": {}
        }
        
        for repo_key, project in repos_to_check.items():
            repo_name = project.get('full_name')
            if not repo_name:
                logger.warning(f"Skipping {repo_key}: No full_name found")
                continue
                
            results["checked"] += 1
            logger.info(f"Checking {repo_name}...")
            
            try:
                # Get list of files in repo
                repo_files = await get_repo_files(repo_name)
                
                # Check essential files
                missing_files = []
                existing_files = []
                
                for file, description in ESSENTIAL_FILES.items():
                    if file in repo_files:
                        existing_files.append(file)
                    else:
                        # For vite.config, check both .ts and .js extensions
                        if file == 'vite.config.ts' and 'vite.config.js' in repo_files:
                            existing_files.append('vite.config.js')
                        else:
                            missing_files.append(f"{file} ({description})")
                
                # Check for React setup
                has_react = False
                for indicator in REACT_INDICATORS:
                    if await check_file_exists(repo_name, indicator):
                        has_react = True
                        break
                
                # Check ARCHITECTURE.md quality
                arch_status = await check_architecture_file_quality(repo_name)
                
                # Determine repository status
                if len(missing_files) == 0:
                    status = "complete"
                    results["complete"] += 1
                elif len(existing_files) <= 3:  # Only has docs files
                    status = "missing_starter_kit"
                    results["missing_starter_kit"] += 1
                else:
                    status = "partial_starter_kit"
                    results["partial_starter_kit"] += 1
                
                if arch_status["format"] in ["old", "missing"]:
                    results["outdated_architecture"] += 1
                
                # Store detailed results
                results["repositories"][repo_name] = {
                    "status": status,
                    "existing_files": len(existing_files),
                    "missing_files": missing_files,
                    "has_react_setup": has_react,
                    "architecture_status": arch_status,
                    "needs_attention": status != "complete" or arch_status["format"] != "new"
                }
                
                # Log status
                if status == "complete":
                    logger.info(f"✓ {repo_name}: Complete starter kit")
                elif status == "missing_starter_kit":
                    logger.warning(f"⚠ {repo_name}: Missing entire starter kit! Only has {len(existing_files)} files")
                else:
                    logger.warning(f"⚠ {repo_name}: Partial starter kit - missing {len(missing_files)} files")
                
                if arch_status["format"] != "new":
                    logger.warning(f"  → ARCHITECTURE.md is {arch_status['format']}")
                
                # Create issues if requested
                if fix_missing and results["repositories"][repo_name]["needs_attention"]:
                    issue_body = f"This repository needs attention:\n\n"
                    
                    if status == "missing_starter_kit":
                        issue_body += "## Missing Laravel React Starter Kit\n\n"
                        issue_body += "This repository appears to only have documentation files but is missing the actual Laravel React starter kit.\n\n"
                        issue_body += "**Action Required:** Initialize the repository with the Laravel React starter kit.\n\n"
                    elif status == "partial_starter_kit":
                        issue_body += "## Incomplete Starter Kit\n\n"
                        issue_body += f"Missing {len(missing_files)} essential files:\n"
                        for file in missing_files[:10]:  # Limit to first 10
                            issue_body += f"- {file}\n"
                        if len(missing_files) > 10:
                            issue_body += f"- ... and {len(missing_files) - 10} more\n"
                        issue_body += "\n"
                    
                    if arch_status["format"] != "new":
                        issue_body += "## ARCHITECTURE.md Status\n\n"
                        if arch_status["format"] == "missing":
                            issue_body += "- ARCHITECTURE.md file is missing\n"
                        elif arch_status["format"] == "old":
                            issue_body += "- ARCHITECTURE.md is using the old format and needs to be updated\n"
                        else:
                            issue_body += "- ARCHITECTURE.md is partially updated but needs completion\n"
                    
                    issue_body += "\n---\n*This issue was automatically created by the Laravel React completeness checker.*"
                    
                    # Create the issue
                    create_cmd = f"""gh issue create \
                        --repo {repo_name} \
                        --title "Repository Setup: Missing Laravel React Starter Kit Files" \
                        --body "{issue_body}" \
                        --label "setup,automated" 2>/dev/null"""
                    
                    create_result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
                    if create_result.returncode == 0:
                        logger.info(f"  → Created issue for {repo_name}")
                    
            except Exception as e:
                results["repositories"][repo_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {repo_name}: Error - {e}")
        
        # Summary
        summary = f"""
Laravel React Completeness Check Complete:
- Repositories checked: {results['checked']}
- Complete starter kits: {results['complete']}
- Missing entire starter kit: {results['missing_starter_kit']}
- Partial starter kit: {results['partial_starter_kit']}
- Outdated ARCHITECTURE.md: {results['outdated_architecture']}
"""
        
        if results['missing_starter_kit'] > 0:
            summary += f"\n⚠️  {results['missing_starter_kit']} repositories have NO starter kit files!"
        
        if fix_missing:
            summary += "\n\nIssues created for repositories needing attention."
            
        logger.info(summary)
        
        return {
            "success": True,
            "summary": summary.strip(),
            "fix_missing": fix_missing,
            **results
        }
        
    except Exception as e:
        error_msg = f"Error checking Laravel React completeness: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

# For testing
if __name__ == "__main__":
    import sys
    
    async def test():
        # Test general check
        print("Testing completeness check...")
        result = await check_laravel_react_completeness()
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test specific repo if provided
        if len(sys.argv) > 1:
            repo = sys.argv[1]
            print(f"\nTesting specific repo: {repo}")
            result = await check_laravel_react_completeness(repo_name=repo)
            print(f"Result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test())

# Export for tool system
check_laravel_react_completeness_tool = check_laravel_react_completeness