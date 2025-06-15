"""
Repository Cleanup Manager

Automatically detects and cleans up references to deleted repositories
from all system state files. This prevents the system from trying to
work on repositories that no longer exist.
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RepositoryCleanupManager:
    """Manages automatic cleanup of deleted repository references."""
    
    def __init__(self):
        """Initialize the cleanup manager."""
        self.logger = logger
        self.state_files = {
            'system_state': 'system_state.json',
            'orchestrator_state': 'continuous_orchestrator_state.json',
            'task_state': 'task_state.json',
            'task_history': 'task_history.json'
        }
        self.cleanup_log_file = 'repository_cleanup.log'
        self.cleanup_history = self._load_cleanup_history()
        
    def _load_cleanup_history(self) -> List[Dict[str, Any]]:
        """Load cleanup history from log file."""
        if os.path.exists(self.cleanup_log_file):
            try:
                with open(self.cleanup_log_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_cleanup_history(self):
        """Save cleanup history to log file."""
        # Keep only last 100 entries
        if len(self.cleanup_history) > 100:
            self.cleanup_history = self.cleanup_history[-100:]
        
        with open(self.cleanup_log_file, 'w') as f:
            json.dump(self.cleanup_history, f, indent=2)
    
    def get_valid_repositories(self) -> Set[str]:
        """Get set of valid repositories from system state."""
        if not os.path.exists(self.state_files['system_state']):
            return set()
        
        try:
            with open(self.state_files['system_state'], 'r') as f:
                system_state = json.load(f)
            
            valid_repos = set()
            # Check both 'projects' and 'repositories' keys
            valid_repos.update(system_state.get('projects', {}).keys())
            valid_repos.update(system_state.get('repositories', {}).keys())
            
            return valid_repos
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")
            return set()
    
    def detect_deleted_repositories(self) -> List[str]:
        """Detect repositories that exist in state files but not in system state."""
        valid_repos = self.get_valid_repositories()
        deleted_repos = set()
        
        # Check orchestrator state
        if os.path.exists(self.state_files['orchestrator_state']):
            try:
                with open(self.state_files['orchestrator_state'], 'r') as f:
                    orch_state = json.load(f)
                
                # Check work queue
                for item in orch_state.get('work_queue', []):
                    repo = item.get('repository')
                    if repo and repo not in valid_repos:
                        deleted_repos.add(repo)
                
                # Check completed work
                for item in orch_state.get('completed_work', []):
                    repo = item.get('repository')
                    if repo and repo not in valid_repos:
                        deleted_repos.add(repo)
            except Exception as e:
                self.logger.error(f"Error checking orchestrator state: {e}")
        
        # Check task state
        if os.path.exists(self.state_files['task_state']):
            try:
                with open(self.state_files['task_state'], 'r') as f:
                    task_state = json.load(f)
                
                for task_id, task_data in task_state.get('tasks', {}).items():
                    repo = task_data.get('repository')
                    if repo and repo not in valid_repos:
                        deleted_repos.add(repo)
            except Exception as e:
                self.logger.error(f"Error checking task state: {e}")
        
        return list(deleted_repos)
    
    def cleanup_orchestrator_state(self, repos_to_remove: List[str]) -> int:
        """Clean up orchestrator state file."""
        if not os.path.exists(self.state_files['orchestrator_state']):
            return 0
        
        try:
            with open(self.state_files['orchestrator_state'], 'r') as f:
                data = json.load(f)
            
            removed_count = 0
            
            # Clean work queue
            if 'work_queue' in data:
                original_len = len(data['work_queue'])
                data['work_queue'] = [
                    item for item in data['work_queue']
                    if item.get('repository') not in repos_to_remove
                ]
                removed_count += original_len - len(data['work_queue'])
            
            # Clean completed work
            if 'completed_work' in data:
                original_len = len(data['completed_work'])
                data['completed_work'] = [
                    item for item in data['completed_work']
                    if item.get('repository') not in repos_to_remove
                ]
                removed_count += original_len - len(data['completed_work'])
            
            if removed_count > 0:
                # Create backup
                backup_path = f"{self.state_files['orchestrator_state']}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.state_files['orchestrator_state'], backup_path)
                
                # Save cleaned data
                with open(self.state_files['orchestrator_state'], 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.logger.info(f"Cleaned orchestrator state: removed {removed_count} items")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning orchestrator state: {e}")
            return 0
    
    def cleanup_task_state(self, repos_to_remove: List[str]) -> int:
        """Clean up task state file."""
        if not os.path.exists(self.state_files['task_state']):
            return 0
        
        try:
            with open(self.state_files['task_state'], 'r') as f:
                data = json.load(f)
            
            removed_count = 0
            
            if 'tasks' in data:
                original_tasks = data['tasks'].copy()
                for task_id, task_data in original_tasks.items():
                    if task_data.get('repository') in repos_to_remove:
                        del data['tasks'][task_id]
                        removed_count += 1
            
            if removed_count > 0:
                # Create backup
                backup_path = f"{self.state_files['task_state']}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.state_files['task_state'], backup_path)
                
                # Save cleaned data
                with open(self.state_files['task_state'], 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.logger.info(f"Cleaned task state: removed {removed_count} tasks")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning task state: {e}")
            return 0
    
    def cleanup_task_history(self, repos_to_remove: List[str]) -> int:
        """Clean up task history file."""
        if not os.path.exists(self.state_files['task_history']):
            return 0
        
        try:
            with open(self.state_files['task_history'], 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return 0
            
            original_len = len(data)
            cleaned_history = []
            
            for entry in data:
                details = entry.get('details', {})
                title = details.get('title', '')
                
                # Check if title mentions any deleted repo
                mentions_deleted = any(repo in title for repo in repos_to_remove)
                
                if not mentions_deleted:
                    cleaned_history.append(entry)
            
            removed_count = original_len - len(cleaned_history)
            
            if removed_count > 0:
                # Create backup
                backup_path = f"{self.state_files['task_history']}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.state_files['task_history'], backup_path)
                
                # Save cleaned data
                with open(self.state_files['task_history'], 'w') as f:
                    json.dump(cleaned_history, f, indent=2)
                
                self.logger.info(f"Cleaned task history: removed {removed_count} entries")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning task history: {e}")
            return 0
    
    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Perform automatic cleanup of deleted repositories.
        
        Args:
            force: Force cleanup even if no deleted repos detected
            
        Returns:
            Cleanup summary dictionary
        """
        self.logger.info("Starting automatic repository cleanup check...")
        
        # Detect deleted repositories
        deleted_repos = self.detect_deleted_repositories()
        
        if not deleted_repos and not force:
            self.logger.info("No deleted repositories detected. System is clean.")
            return {
                'deleted_repos': [],
                'items_removed': 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        self.logger.warning(f"Detected {len(deleted_repos)} deleted repositories: {', '.join(deleted_repos)}")
        
        # Perform cleanup
        total_removed = 0
        cleanup_summary = {
            'orchestrator_state': 0,
            'task_state': 0,
            'task_history': 0
        }
        
        if deleted_repos:
            cleanup_summary['orchestrator_state'] = self.cleanup_orchestrator_state(deleted_repos)
            cleanup_summary['task_state'] = self.cleanup_task_state(deleted_repos)
            cleanup_summary['task_history'] = self.cleanup_task_history(deleted_repos)
            
            total_removed = sum(cleanup_summary.values())
        
        # Record cleanup in history
        cleanup_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'deleted_repos': deleted_repos,
            'items_removed': total_removed,
            'details': cleanup_summary
        }
        
        self.cleanup_history.append(cleanup_record)
        self._save_cleanup_history()
        
        self.logger.info(f"Cleanup completed. Total items removed: {total_removed}")
        
        return cleanup_record
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status and history."""
        return {
            'valid_repositories': list(self.get_valid_repositories()),
            'detected_deleted': self.detect_deleted_repositories(),
            'last_cleanup': self.cleanup_history[-1] if self.cleanup_history else None,
            'cleanup_count': len(self.cleanup_history)
        }


# Convenience function for integration
async def check_and_cleanup_repositories():
    """Check and cleanup deleted repositories (async wrapper)."""
    manager = RepositoryCleanupManager()
    return manager.perform_cleanup()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository cleanup manager')
    parser.add_argument('--check', action='store_true', help='Check for deleted repositories')
    parser.add_argument('--cleanup', action='store_true', help='Perform cleanup')
    parser.add_argument('--status', action='store_true', help='Show cleanup status')
    parser.add_argument('--force', action='store_true', help='Force cleanup')
    
    args = parser.parse_args()
    
    manager = RepositoryCleanupManager()
    
    if args.check:
        deleted = manager.detect_deleted_repositories()
        if deleted:
            print(f"Detected {len(deleted)} deleted repositories:")
            for repo in deleted:
                print(f"  - {repo}")
        else:
            print("No deleted repositories detected.")
    
    elif args.cleanup:
        result = manager.perform_cleanup(force=args.force)
        print(f"Cleanup completed:")
        print(f"  Deleted repos: {', '.join(result['deleted_repos']) or 'None'}")
        print(f"  Items removed: {result['items_removed']}")
    
    elif args.status:
        status = manager.get_cleanup_status()
        print(f"Repository Cleanup Status:")
        print(f"  Valid repositories: {len(status['valid_repositories'])}")
        print(f"  Detected deleted: {len(status['detected_deleted'])}")
        if status['last_cleanup']:
            print(f"  Last cleanup: {status['last_cleanup']['timestamp']}")
            print(f"  Items removed: {status['last_cleanup']['items_removed']}")
        print(f"  Total cleanups: {status['cleanup_count']}")
    
    else:
        parser.print_help()