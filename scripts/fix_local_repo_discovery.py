#!/usr/bin/env python3
"""
Fix Local Repository Discovery

Ensures all local repositories are properly registered in the state.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Dict, Any

from scripts.local_github_simulator import LocalGitHubSimulator


class LocalRepoFixer:
    """Fixes local repository discovery issues."""
    
    def __init__(self):
        """Initialize the fixer."""
        self.logger = logging.getLogger(__name__)
        self.simulator = LocalGitHubSimulator()
        self.base_path = Path("./local_repos")
        self.repos_path = self.base_path / "repositories"
        self.state_path = self.base_path / "state"
        
    def scan_and_register_all_repos(self) -> Dict[str, Any]:
        """Scan filesystem and register all found repositories.
        
        Returns:
            Summary of fixes applied
        """
        print("Scanning for local repositories...")
        
        summary = {
            'found_directories': 0,
            'already_registered': 0,
            'newly_registered': 0,
            'errors': []
        }
        
        # Get current state
        repos_file = self.state_path / "repositories.json"
        if repos_file.exists():
            with open(repos_file, 'r') as f:
                state = json.load(f)
        else:
            state = {
                "repositories": [],
                "total_count": 0,
                "last_updated": None,
                "metadata": {"version": "1.0", "storage_mode": "local"}
            }
        
        # Create a map of existing repos
        existing_repos = {repo["full_name"]: repo for repo in state["repositories"]}
        
        # Scan all organizations
        for org_dir in self.repos_path.iterdir():
            if not org_dir.is_dir():
                continue
                
            org_name = org_dir.name
            print(f"\nScanning organization: {org_name}")
            
            # Scan all repositories in organization
            for repo_dir in org_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                    
                repo_name = repo_dir.name
                full_name = f"{org_name}/{repo_name}"
                summary['found_directories'] += 1
                
                print(f"  Found: {full_name}")
                
                # Check if already registered
                if full_name in existing_repos:
                    print(f"    ✓ Already registered")
                    summary['already_registered'] += 1
                    continue
                
                # Register new repository
                try:
                    self._register_repository(state, org_name, repo_name, repo_dir)
                    print(f"    + Newly registered")
                    summary['newly_registered'] += 1
                except Exception as e:
                    error_msg = f"Failed to register {full_name}: {e}"
                    print(f"    ❌ {error_msg}")
                    summary['errors'].append(error_msg)
                    self.logger.error(error_msg)
        
        # Update total count
        state["total_count"] = len(state["repositories"])
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated state
        with open(repos_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("DISCOVERY SUMMARY")
        print("="*50)
        print(f"Total directories found: {summary['found_directories']}")
        print(f"Already registered: {summary['already_registered']}")
        print(f"Newly registered: {summary['newly_registered']}")
        print(f"Total repositories now: {state['total_count']}")
        
        if summary['errors']:
            print("\nErrors:")
            for error in summary['errors']:
                print(f"  - {error}")
        
        return summary
    
    def _register_repository(self, state: Dict[str, Any], org_name: str, 
                           repo_name: str, repo_path: Path):
        """Register a repository in the state.
        
        Args:
            state: Current state dictionary
            org_name: Organization name
            repo_name: Repository name
            repo_path: Path to repository
        """
        full_name = f"{org_name}/{repo_name}"
        
        # Try to load metadata
        metadata_path = repo_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Create metadata if missing
            metadata = {
                "name": repo_name,
                "full_name": full_name,
                "description": f"Local repository {repo_name}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "next_issue_number": 1,
                "organization": org_name
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create repository entry
        repo_entry = {
            "name": repo_name,
            "full_name": full_name,
            "description": metadata.get("description", ""),
            "path": str(repo_path),
            "created_at": metadata.get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": metadata.get("updated_at", datetime.now(timezone.utc).isoformat()),
            "default_branch": "main",
            "private": False,
            "html_url": f"file://{repo_path}"
        }
        
        # Ensure issues directory exists
        issues_dir = repo_path / "issues"
        issues_dir.mkdir(exist_ok=True)
        
        # Ensure .github directory exists
        github_dir = repo_path / ".github"
        github_dir.mkdir(exist_ok=True)
        
        # Add to state
        state["repositories"].append(repo_entry)
    
    def create_missing_repos_from_github(self) -> Dict[str, Any]:
        """Create local repos for any that exist on GitHub but not locally.
        
        Returns:
            Summary of created repositories
        """
        print("\nChecking for repositories that exist on GitHub but not locally...")
        
        summary = {
            'checked': 0,
            'created': 0,
            'errors': []
        }
        
        # Get list of expected repositories from the system state
        from scripts.state_manager import StateManager
        state_manager = StateManager()
        system_state = state_manager.load_state()
        
        projects = system_state.get('projects', {})
        
        for project_name, project_data in projects.items():
            summary['checked'] += 1
            full_name = project_data.get('full_name', f"CodeWebMobile-AI/{project_name}")
            
            # Check if it exists locally
            if not self.simulator.get_repository(full_name):
                print(f"  Creating local copy of: {full_name}")
                try:
                    self.simulator.create_repository(
                        name=project_name,
                        description=project_data.get('description', ''),
                        organization="CodeWebMobile-AI"
                    )
                    summary['created'] += 1
                    print(f"    ✓ Created")
                except Exception as e:
                    error_msg = f"Failed to create {full_name}: {e}"
                    print(f"    ❌ {error_msg}")
                    summary['errors'].append(error_msg)
        
        print(f"\nCreated {summary['created']} new local repositories")
        return summary


def main():
    """Main function to fix local repository discovery."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fixer = LocalRepoFixer()
    
    # First, scan and register all existing local repos
    discovery_summary = fixer.scan_and_register_all_repos()
    
    # Then, create any missing repos from GitHub
    creation_summary = fixer.create_missing_repos_from_github()
    
    # Final scan to ensure everything is registered
    print("\nRunning final scan...")
    final_summary = fixer.scan_and_register_all_repos()
    
    print("\n✅ Local repository discovery fixed!")
    
    return 0


if __name__ == "__main__":
    exit(main())