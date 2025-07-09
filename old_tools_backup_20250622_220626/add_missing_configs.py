"""
Add Missing Configs Tool
========================

Adds missing Tailwind CSS and PostCSS configuration files to repositories
that already have Laravel React but are missing these specific configs.
"""

import os
import asyncio
import logging
import subprocess
import json
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback

from scripts.state_manager import StateManager
from scripts.logger import logger

__description__ = "Add missing Tailwind CSS and PostCSS configuration files to repositories"
__parameters__ = {
    'dry_run': {'type': 'boolean', 'required': False, 'description': 'If true, only check without making changes'},
    'repo_name': {'type': 'string', 'required': False, 'description': 'Specific repository to update (if not provided, updates all)'}
}
__examples__ = [
    {'dry_run': True},
    {'repo_name': 'eco-track-ai'},
    {}
]

# Tailwind configuration
TAILWIND_CONFIG = """/** @type {import('tailwindcss').Config} */
export default {
    content: [
        './vendor/laravel/framework/src/Illuminate/Pagination/resources/views/*.blade.php',
        './storage/framework/views/*.php',
        './resources/views/**/*.blade.php',
        './resources/js/**/*.tsx',
        './resources/js/**/*.ts',
        './resources/js/**/*.jsx',
        './resources/js/**/*.js',
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Figtree', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            },
            colors: {
                primary: {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    200: '#bfdbfe',
                    300: '#93c5fd',
                    400: '#60a5fa',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1d4ed8',
                    800: '#1e40af',
                    900: '#1e3a8a',
                },
            },
        },
    },
    plugins: [require('@tailwindcss/forms'), require('@tailwindcss/typography')],
};
"""

# PostCSS configuration
POSTCSS_CONFIG = """export default {
    plugins: {
        'tailwindcss/nesting': {},
        tailwindcss: {},
        autoprefixer: {},
        ...(process.env.NODE_ENV === 'production' ? { cssnano: {} } : {})
    },
};
"""

# Alternative PostCSS config for older setups
POSTCSS_CONFIG_CJS = """module.exports = {
    plugins: {
        'tailwindcss/nesting': {},
        tailwindcss: {},
        autoprefixer: {},
        ...(process.env.NODE_ENV === 'production' ? { cssnano: {} } : {})
    },
};
"""

async def check_file_exists(repo_name: str, file_path: str) -> bool:
    """Check if a file exists in the repository using gh CLI."""
    cmd = f"gh api repos/{repo_name}/contents/{file_path} --silent"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

async def create_file_via_api(repo_name: str, file_path: str, content: str, message: str, branch: str = "main") -> bool:
    """Create a file in the repository using GitHub API."""
    try:
        # Encode content to base64
        content_b64 = base64.b64encode(content.encode()).decode()
        
        # Create the file
        create_data = {
            "message": message,
            "content": content_b64,
            "branch": branch
        }
        
        create_cmd = f"gh api repos/{repo_name}/contents/{file_path} -X PUT --input -"
        create_result = subprocess.run(
            create_cmd, 
            shell=True, 
            input=json.dumps(create_data),
            capture_output=True, 
            text=True
        )
        
        return create_result.returncode == 0
    except Exception as e:
        logger.error(f"Error creating file {file_path}: {e}")
        return False

async def add_missing_configs(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add missing Tailwind CSS and PostCSS configuration files to repositories.
    """
    try:
        dry_run = kwargs.get('dry_run', False)
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
            "updated": 0,
            "already_complete": 0,
            "failed": 0,
            "repositories": {}
        }
        
        for repo_key, project in repos_to_check.items():
            repo_name = project.get('full_name')
            if not repo_name:
                logger.warning(f"Skipping {repo_key}: No full_name found")
                continue
                
            results["checked"] += 1
            logger.info(f"Checking {repo_name} for missing configs...")
            
            try:
                # Check if repo has package.json (indicates it's a Laravel project)
                has_package = await check_file_exists(repo_name, "package.json")
                if not has_package:
                    results["repositories"][repo_name] = {
                        "status": "skipped",
                        "message": "Not a Laravel project (no package.json)"
                    }
                    continue
                
                # Check for missing files
                missing_files = []
                
                # Check Tailwind config
                has_tailwind_js = await check_file_exists(repo_name, "tailwind.config.js")
                has_tailwind_ts = await check_file_exists(repo_name, "tailwind.config.ts")
                if not has_tailwind_js and not has_tailwind_ts:
                    missing_files.append({
                        "path": "tailwind.config.js",
                        "content": TAILWIND_CONFIG,
                        "description": "Tailwind CSS configuration"
                    })
                
                # Check PostCSS config
                has_postcss_js = await check_file_exists(repo_name, "postcss.config.js")
                has_postcss_cjs = await check_file_exists(repo_name, "postcss.config.cjs")
                has_postcss_mjs = await check_file_exists(repo_name, "postcss.config.mjs")
                if not has_postcss_js and not has_postcss_cjs and not has_postcss_mjs:
                    # Check if package.json uses ES modules
                    package_cmd = f"gh api repos/{repo_name}/contents/package.json --jq '.content' | base64 -d"
                    package_result = subprocess.run(package_cmd, shell=True, capture_output=True, text=True)
                    
                    use_esm = False
                    if package_result.returncode == 0:
                        try:
                            package_json = json.loads(package_result.stdout)
                            use_esm = package_json.get("type") == "module"
                        except:
                            pass
                    
                    missing_files.append({
                        "path": "postcss.config.js",
                        "content": POSTCSS_CONFIG if use_esm else POSTCSS_CONFIG_CJS,
                        "description": "PostCSS configuration"
                    })
                
                if not missing_files:
                    results["already_complete"] += 1
                    results["repositories"][repo_name] = {
                        "status": "complete",
                        "message": "All configuration files present"
                    }
                    logger.info(f"✓ {repo_name}: All configs present")
                    continue
                
                # Process missing files
                if dry_run:
                    results["repositories"][repo_name] = {
                        "status": "would_update",
                        "missing": [f["description"] for f in missing_files],
                        "message": f"Would add {len(missing_files)} config file(s) (dry run)"
                    }
                    logger.info(f"→ {repo_name}: Would add {len(missing_files)} config(s) (dry run)")
                else:
                    # Add missing files
                    added = []
                    failed = []
                    
                    for file_info in missing_files:
                        success = await create_file_via_api(
                            repo_name,
                            file_info["path"],
                            file_info["content"],
                            f"Add {file_info['description']}\n\nAdded missing configuration file for Laravel React setup",
                            project.get('default_branch', 'main')
                        )
                        
                        if success:
                            added.append(file_info["description"])
                            logger.info(f"  ✓ Added {file_info['path']}")
                        else:
                            failed.append(file_info["description"])
                            logger.error(f"  ✗ Failed to add {file_info['path']}")
                    
                    if added and not failed:
                        results["updated"] += 1
                        results["repositories"][repo_name] = {
                            "status": "updated",
                            "added": added,
                            "message": f"Successfully added {len(added)} config file(s)"
                        }
                        logger.info(f"✓ {repo_name}: Added {len(added)} config(s)")
                    elif failed:
                        results["failed"] += 1
                        results["repositories"][repo_name] = {
                            "status": "partial",
                            "added": added,
                            "failed": failed,
                            "message": f"Added {len(added)}, failed {len(failed)} config file(s)"
                        }
                        logger.warning(f"⚠ {repo_name}: Partial update")
                    
            except Exception as e:
                results["failed"] += 1
                results["repositories"][repo_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {repo_name}: Error - {e}")
        
        # Summary
        summary = f"""
Missing Configs Check Complete:
- Repositories checked: {results['checked']}
- Already complete: {results['already_complete']}
- Updated: {results['updated']}
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
        error_msg = f"Error adding missing configs: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

# Export for tool system
add_missing_configs_tool = add_missing_configs

if __name__ == "__main__":
    import sys
    
    async def test():
        # Test dry run
        print("Testing dry run...")
        result = await add_missing_configs(dry_run=True)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test specific repo if provided
        if len(sys.argv) > 1:
            repo = sys.argv[1]
            print(f"\nTesting specific repo: {repo}")
            result = await add_missing_configs(repo_name=repo, dry_run=True)
            print(f"Result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test())