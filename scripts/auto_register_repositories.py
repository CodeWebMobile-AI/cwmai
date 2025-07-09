#!/usr/bin/env python3
"""
Automatic Repository Registration System

This module provides automatic registration of new repositories to prevent
discovery issues. It can be integrated into the project creation workflow
or run as a periodic maintenance task.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import Dict, List, Set, Optional
import asyncio


class RepositoryAutoRegistrar:
    """Automatically registers new repositories found in the filesystem."""
    
    def __init__(self, 
                 repos_base_path: str = None,
                 state_path: str = None):
        """Initialize the auto-registrar.
        
        Args:
            repos_base_path: Base path where repositories are stored
            state_path: Path to state directory containing repositories.json
        """
        if repos_base_path:
            self.repos_base_path = Path(repos_base_path)
        else:
            self.repos_base_path = Path.cwd() / "local_repos" / "repositories"
            
        if state_path:
            self.state_path = Path(state_path)
        else:
            self.state_path = Path.cwd() / "local_repos" / "state"
        self.repos_json_path = self.state_path / "repositories.json"
        self.logger = logging.getLogger(__name__)
        
    def scan_for_repositories(self) -> Dict[str, List[str]]:
        """Scan filesystem for all repository directories.
        
        Returns:
            Dictionary mapping organization names to lists of repository names
        """
        found_repos = {}
        
        if not self.repos_base_path.exists():
            self.logger.warning(f"Repository base path does not exist: {self.repos_base_path}")
            return found_repos
            
        # Scan for organization directories
        for org_dir in self.repos_base_path.iterdir():
            if org_dir.is_dir() and not org_dir.name.startswith('.'):
                org_name = org_dir.name
                repos = []
                
                # Scan for repository directories within organization
                for repo_dir in org_dir.iterdir():
                    if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                        # Verify it's a valid repository (has .git or metadata.json)
                        if self._is_valid_repository(repo_dir):
                            repos.append(repo_dir.name)
                
                if repos:
                    found_repos[org_name] = repos
                    
        return found_repos
        
    def _is_valid_repository(self, repo_path: Path) -> bool:
        """Check if a directory is a valid repository.
        
        Args:
            repo_path: Path to potential repository directory
            
        Returns:
            True if directory appears to be a valid repository
        """
        # Check for .git directory or metadata.json file
        return (repo_path / '.git').exists() or (repo_path / 'metadata.json').exists()
        
    def load_registered_repositories(self) -> Dict[str, Set[str]]:
        """Load currently registered repositories from repositories.json.
        
        Returns:
            Dictionary mapping organization names to sets of repository names
        """
        registered = {}
        
        if not self.repos_json_path.exists():
            return registered
            
        try:
            with open(self.repos_json_path, 'r') as f:
                data = json.load(f)
                
            for repo in data.get('repositories', []):
                full_name = repo.get('full_name', '')
                if '/' in full_name:
                    org, name = full_name.split('/', 1)
                    if org not in registered:
                        registered[org] = set()
                    registered[org].add(name)
                    
        except Exception as e:
            self.logger.error(f"Error loading repositories.json: {e}")
            
        return registered
        
    def register_new_repositories(self, dry_run: bool = False) -> Dict[str, List[str]]:
        """Register any new repositories found in the filesystem.
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            Dictionary of newly registered repositories by organization
        """
        # Scan filesystem
        found_repos = self.scan_for_repositories()
        
        # Load registered repositories
        registered_repos = self.load_registered_repositories()
        
        # Find unregistered repositories
        new_repos = {}
        for org, repos in found_repos.items():
            registered_in_org = registered_repos.get(org, set())
            unregistered = set(repos) - registered_in_org
            
            if unregistered:
                new_repos[org] = sorted(list(unregistered))
                
        if not new_repos:
            self.logger.info("No new repositories found to register")
            return {}
            
        # Report findings
        total_new = sum(len(repos) for repos in new_repos.values())
        self.logger.info(f"Found {total_new} new repositories to register")
        
        for org, repos in new_repos.items():
            self.logger.info(f"  {org}: {', '.join(repos)}")
            
        if dry_run:
            self.logger.info("Dry run mode - no changes made")
            return new_repos
            
        # Load full repositories.json data
        try:
            with open(self.repos_json_path, 'r') as f:
                repos_data = json.load(f)
        except:
            repos_data = {
                "repositories": [],
                "total_count": 0,
                "metadata": {"version": "1.0", "storage_mode": "local"}
            }
            
        # Add new repositories
        for org, repos in new_repos.items():
            for repo_name in repos:
                repo_path = self.repos_base_path / org / repo_name
                
                # Try to load metadata
                metadata_path = repo_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        description = metadata.get('description', '')
                    except:
                        description = f"AI-powered {repo_name.replace('-', ' ')} application"
                else:
                    description = f"AI-powered {repo_name.replace('-', ' ')} application"
                
                # Create repository entry
                repo_entry = {
                    "name": repo_name,
                    "full_name": f"{org}/{repo_name}",
                    "description": description,
                    "path": str(repo_path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "default_branch": "main",
                    "private": False,
                    "html_url": f"file://{repo_path}",
                    "language": None,
                    "languages_url": None,
                    "open_issues_count": 0,
                    "forks_count": 0,
                    "stargazers_count": 0
                }
                
                repos_data['repositories'].append(repo_entry)
                self.logger.info(f"Registered: {org}/{repo_name}")
                
        # Update metadata
        repos_data['total_count'] = len(repos_data['repositories'])
        repos_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save updated repositories.json
        with open(self.repos_json_path, 'w') as f:
            json.dump(repos_data, f, indent=2)
            
        self.logger.info(f"Successfully registered {total_new} new repositories")
        return new_repos
        
    def setup_file_watcher(self):
        """Set up a file system watcher to detect new repositories in real-time.
        
        This method uses asyncio and watchdog library (if available) to monitor
        the repository directories for changes.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class RepositoryCreationHandler(FileSystemEventHandler):
                def __init__(self, registrar):
                    self.registrar = registrar
                    
                def on_created(self, event):
                    if event.is_directory:
                        # Check if this might be a new repository
                        path = Path(event.src_path)
                        if path.parent.parent == self.registrar.repos_base_path:
                            # Wait a moment for repository initialization
                            import time
                            time.sleep(2)
                            
                            # Try to register new repositories
                            self.registrar.register_new_repositories()
            
            # Set up observer
            observer = Observer()
            handler = RepositoryCreationHandler(self)
            observer.schedule(handler, str(self.repos_base_path), recursive=True)
            observer.start()
            
            self.logger.info("File system watcher started for automatic repository registration")
            return observer
            
        except ImportError:
            self.logger.warning("watchdog library not available - file system watching disabled")
            return None


class RepositoryRegistrationHook:
    """Hook to be integrated into project creation workflow."""
    
    @staticmethod
    def post_create_hook(repo_full_name: str, repo_path: str, metadata: Optional[Dict] = None):
        """Hook to be called after a new repository is created.
        
        Args:
            repo_full_name: Full repository name (org/repo)
            repo_path: Path to the repository
            metadata: Optional metadata about the repository
        """
        registrar = RepositoryAutoRegistrar()
        
        # Ensure repository is registered
        registrar.register_new_repositories()
        
        logging.info(f"Post-creation registration check completed for {repo_full_name}")


async def periodic_registration_check(interval_minutes: int = 5):
    """Run periodic checks for new repositories.
    
    Args:
        interval_minutes: How often to check for new repositories
    """
    registrar = RepositoryAutoRegistrar()
    
    while True:
        try:
            # Check for new repositories
            new_repos = registrar.register_new_repositories()
            
            if new_repos:
                logging.info(f"Periodic check registered {sum(len(r) for r in new_repos.values())} new repositories")
                
        except Exception as e:
            logging.error(f"Error in periodic registration check: {e}")
            
        # Wait for next check
        await asyncio.sleep(interval_minutes * 60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatically register new repositories')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--watch', action='store_true', help='Watch for new repositories continuously')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in minutes (for --watch mode)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    registrar = RepositoryAutoRegistrar()
    
    if args.watch:
        # Watch mode
        print(f"Starting repository watcher (checking every {args.interval} minutes)")
        
        # Try to set up file system watcher
        observer = registrar.setup_file_watcher()
        
        # Run periodic checks
        try:
            asyncio.run(periodic_registration_check(args.interval))
        except KeyboardInterrupt:
            print("\nStopping repository watcher")
            if observer:
                observer.stop()
                observer.join()
    else:
        # One-time registration
        registrar.register_new_repositories(dry_run=args.dry_run)


if __name__ == "__main__":
    main()