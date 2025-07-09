import asyncio
import subprocess
import json
import os
from typing import Dict, Any
from pathlib import Path

async def check_architecture_files(**kwargs: Any) -> Dict[str, Any]:
    """
    Checks all managed repositories for ARCHITECTURE.md files and evaluates their quality.

    __description__ = Checks for ARCHITECTURE.md files in managed repositories and assesses their quality based on a simple presence check. More sophisticated checks can be added later.

    __parameters__ = {
        "repositories": "List of repository names to check (e.g., ['org/repo1', 'org/repo2']). If not provided, all managed repositories are checked."
    }

    __examples__ = [
        {"repositories": ["my-org/repo1", "my-org/repo2"]},
        {}  # Checks all managed repositories
    ]
    """

    try:
        from scripts.state_manager import StateManager
        from scripts.storage_factory import StorageFactory
        from scripts.local_github_simulator import LocalGitHubSimulator

        # Initialize storage factory
        storage_factory = StorageFactory()
        storage_mode = storage_factory.storage_mode
        
        state_manager = StateManager()
        # Get repositories from the projects section of state
        state = state_manager.load_state()
        all_repos = list(state.get('projects', {}).keys())
        
        if storage_mode == "local":
            local_simulator = LocalGitHubSimulator()

        repos_to_check = kwargs.get("repositories")
        if repos_to_check is None:
            repos_to_check = all_repos
        else:
            # Validate repos_to_check against all_repos.  Only check repos that are managed.
            valid_repos_to_check = [repo for repo in repos_to_check if repo in all_repos]
            if len(valid_repos_to_check) != len(repos_to_check):
                invalid_repos = [repo for repo in repos_to_check if repo not in all_repos]
                print(f"Warning: The following repositories are not managed: {invalid_repos}")
                repos_to_check = valid_repos_to_check  # Only check the valid ones.


        results = {}
        for repo in repos_to_check:
            try:
                # First get the org/repo format from project data
                project_data = state.get('projects', {}).get(repo, {})
                full_name = project_data.get('full_name', f'CodeWebMobile-AI/{repo}')
                
                if storage_mode == "local":
                    # Check in local repository
                    repo_path = Path(local_simulator.repos_path) / full_name.replace('/', os.sep)
                    arch_file_path = repo_path / "ARCHITECTURE.md"
                    file_exists = arch_file_path.exists()
                else:
                    # Use gh api to check if file exists
                    command = ["gh", "api", f"repos/{full_name}/contents/ARCHITECTURE.md", "--silent"]
                    process = subprocess.run(command, capture_output=True, text=True, check=False)
                    
                    # Check if gh api succeeded (exit code 0 means file exists)
                    file_exists = process.returncode == 0

                results[repo] = {"architecture_file_exists": file_exists}

                print(f"Info: Repository {repo}: ARCHITECTURE.md exists: {file_exists}")
            except subprocess.CalledProcessError as e:
                print(f"Error checking repository {repo}: {e}")
                results[repo] = {"error": str(e)}
            except Exception as e:
                print(f"Unexpected error checking repository {repo}: {e}")
                results[repo] = {"error": str(e)}

        return results

    except ImportError as e:
        return {"error": f"Import error: {e}.  Make sure scripts.state_manager.StateManager and scripts.conversational_ai_logger are available."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}