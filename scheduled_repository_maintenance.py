#!/usr/bin/env python3
"""
Scheduled Repository Maintenance

This script performs regular maintenance tasks on all repositories:
1. Checks for and fixes missing customizations
2. Updates generic descriptions
3. Generates missing architecture documents
4. Cleans up deleted repository references
5. Ensures all projects are properly configured

Can be run via cron or scheduled task runner.
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.repository_cleanup_manager import RepositoryCleanupManager
from fix_repository_customizations import RepositoryCustomizationFixer
from scripts.state_manager import StateManager
from scripts.ai_brain import AIBrain
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repository_maintenance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScheduledRepositoryMaintenance:
    """Performs scheduled maintenance on all repositories."""
    
    def __init__(self, github_token: str, use_ai: bool = True):
        """Initialize maintenance runner.
        
        Args:
            github_token: GitHub personal access token
            use_ai: Whether to use AI for intelligent fixes
        """
        self.github_token = github_token
        self.use_ai = use_ai
        self.logger = logger
        
        # Initialize components
        self.cleanup_manager = RepositoryCleanupManager()
        self.ai_brain = AIBrain() if use_ai else None
        self.customization_fixer = RepositoryCustomizationFixer(github_token, self.ai_brain)
        self.state_manager = StateManager()
        
        # Track maintenance history
        self.maintenance_log_file = 'repository_maintenance_history.json'
        self.maintenance_history = self._load_maintenance_history()
        
    def _load_maintenance_history(self) -> List[Dict[str, Any]]:
        """Load maintenance history from file."""
        if os.path.exists(self.maintenance_log_file):
            try:
                with open(self.maintenance_log_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_maintenance_history(self):
        """Save maintenance history to file."""
        # Keep only last 30 days of history
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        self.maintenance_history = [
            entry for entry in self.maintenance_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        with open(self.maintenance_log_file, 'w') as f:
            json.dump(self.maintenance_history, f, indent=2)
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run all maintenance tasks.
        
        Returns:
            Summary of maintenance performed
        """
        self.logger.info("=" * 60)
        self.logger.info("🔧 Starting scheduled repository maintenance")
        self.logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        self.logger.info("=" * 60)
        
        maintenance_summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tasks_performed': [],
            'issues_found': 0,
            'issues_fixed': 0,
            'errors': []
        }
        
        try:
            # Task 1: Clean up deleted repositories
            self.logger.info("\n📋 Task 1: Cleaning up deleted repository references...")
            cleanup_result = await self._run_cleanup_task()
            maintenance_summary['tasks_performed'].append({
                'task': 'cleanup_deleted_repos',
                'result': cleanup_result
            })
            
            # Task 2: Fix repository customizations
            self.logger.info("\n📋 Task 2: Checking and fixing repository customizations...")
            customization_result = await self._run_customization_fixes()
            maintenance_summary['tasks_performed'].append({
                'task': 'fix_customizations',
                'result': customization_result
            })
            maintenance_summary['issues_found'] += customization_result.get('total_issues_found', 0)
            maintenance_summary['issues_fixed'] += customization_result.get('total_fixes_applied', 0)
            
            # Task 3: Update repository discovery
            self.logger.info("\n📋 Task 3: Updating repository discovery...")
            discovery_result = await self._update_repository_discovery()
            maintenance_summary['tasks_performed'].append({
                'task': 'update_discovery',
                'result': discovery_result
            })
            
            # Task 4: Generate missing architecture documents
            self.logger.info("\n📋 Task 4: Generating missing architecture documents...")
            architecture_result = await self._generate_missing_architectures()
            maintenance_summary['tasks_performed'].append({
                'task': 'generate_architectures',
                'result': architecture_result
            })
            
            # Task 5: Verify all repositories are properly configured
            self.logger.info("\n📋 Task 5: Verifying repository configurations...")
            verification_result = await self._verify_repository_configurations()
            maintenance_summary['tasks_performed'].append({
                'task': 'verify_configurations',
                'result': verification_result
            })
            
            # Calculate overall status
            maintenance_summary['status'] = 'completed'
            maintenance_summary['duration_seconds'] = (
                datetime.now(timezone.utc) - datetime.fromisoformat(maintenance_summary['timestamp'])
            ).total_seconds()
            
        except Exception as e:
            self.logger.error(f"❌ Maintenance failed: {e}")
            maintenance_summary['status'] = 'failed'
            maintenance_summary['errors'].append(str(e))
        
        # Save to history
        self.maintenance_history.append(maintenance_summary)
        self._save_maintenance_history()
        
        # Log summary
        self._log_summary(maintenance_summary)
        
        return maintenance_summary
    
    async def _run_cleanup_task(self) -> Dict[str, Any]:
        """Run repository cleanup task."""
        try:
            result = self.cleanup_manager.perform_cleanup()
            self.logger.info(f"✅ Cleanup completed: {result['items_removed']} items removed")
            return result
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return {'error': str(e), 'items_removed': 0}
    
    async def _run_customization_fixes(self) -> Dict[str, Any]:
        """Run repository customization fixes."""
        try:
            # First scan for issues
            scan_result = await self.customization_fixer.scan_repositories(fix_mode=False)
            
            # If issues found and AI is enabled, fix them
            if scan_result['repositories_with_issues'] > 0 and self.use_ai:
                self.logger.info(f"🔧 Found issues in {scan_result['repositories_with_issues']} repositories, applying fixes...")
                fix_result = await self.customization_fixer.scan_repositories(fix_mode=True)
                return fix_result
            
            return scan_result
            
        except Exception as e:
            self.logger.error(f"❌ Customization fix failed: {e}")
            return {'error': str(e), 'total_issues_found': 0, 'total_fixes_applied': 0}
    
    async def _update_repository_discovery(self) -> Dict[str, Any]:
        """Update repository discovery in system state."""
        try:
            from fix_repository_discovery import fix_repository_discovery
            success = await fix_repository_discovery()
            
            if success:
                # Reload state to get updated repositories
                self.state_manager.load_state()
                projects = self.state_manager.state.get('projects', {})
                
                return {
                    'status': 'success',
                    'repositories_discovered': len(projects),
                    'last_discovery': datetime.now(timezone.utc).isoformat()
                }
            else:
                return {'status': 'failed', 'repositories_discovered': 0}
                
        except Exception as e:
            self.logger.error(f"❌ Repository discovery failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _generate_missing_architectures(self) -> Dict[str, Any]:
        """Generate architecture documents for repositories missing them."""
        if not self.ai_brain:
            self.logger.warning("⚠️  AI brain not available - skipping architecture generation")
            return {'status': 'skipped', 'reason': 'AI not enabled'}
        
        try:
            from scripts.architecture_generator import ArchitectureGenerator
            from github import Github
            
            github = Github(self.github_token)
            org = github.get_organization("CodeWebMobile-AI")
            generator = ArchitectureGenerator(self.github_token, self.ai_brain)
            
            generated_count = 0
            failed_count = 0
            
            for repo in org.get_repos():
                if repo.name in ['cwmai', '.github', 'cwmai.git']:
                    continue
                
                # Check if architecture exists
                try:
                    repo.get_contents("ARCHITECTURE.md")
                    continue  # Already has architecture
                except:
                    pass  # Needs architecture
                
                # Generate architecture
                try:
                    self.logger.info(f"📐 Generating architecture for {repo.name}...")
                    content = await generator.generate_for_repository(repo.full_name)
                    
                    if content:
                        repo.create_file(
                            "ARCHITECTURE.md",
                            "Add comprehensive architecture documentation",
                            content
                        )
                        generated_count += 1
                        self.logger.info(f"✅ Architecture generated for {repo.name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate architecture for {repo.name}: {e}")
                    failed_count += 1
            
            return {
                'status': 'completed',
                'generated': generated_count,
                'failed': failed_count
            }
            
        except Exception as e:
            self.logger.error(f"❌ Architecture generation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _verify_repository_configurations(self) -> Dict[str, Any]:
        """Verify all repositories are properly configured."""
        try:
            from github import Github
            
            github = Github(self.github_token)
            org = github.get_organization("CodeWebMobile-AI")
            
            verification_results = {
                'total_checked': 0,
                'properly_configured': 0,
                'issues_found': [],
                'recommendations': []
            }
            
            for repo in org.get_repos():
                if repo.name in ['cwmai', '.github', 'cwmai.git']:
                    continue
                
                verification_results['total_checked'] += 1
                repo_issues = []
                
                # Check various aspects
                if not repo.description or len(repo.description) < 20:
                    repo_issues.append('Missing or inadequate description')
                
                if not repo.get_topics():
                    repo_issues.append('No topics/tags set')
                
                if not repo.has_issues:
                    repo_issues.append('Issues are disabled')
                
                if repo_issues:
                    verification_results['issues_found'].append({
                        'repository': repo.name,
                        'issues': repo_issues
                    })
                else:
                    verification_results['properly_configured'] += 1
            
            # Add recommendations
            if verification_results['issues_found']:
                verification_results['recommendations'].append(
                    "Run maintenance with --fix flag to automatically resolve these issues"
                )
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"❌ Verification failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _log_summary(self, summary: Dict[str, Any]):
        """Log maintenance summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 MAINTENANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {summary['status']}")
        self.logger.info(f"Duration: {summary.get('duration_seconds', 0):.1f} seconds")
        self.logger.info(f"Issues Found: {summary['issues_found']}")
        self.logger.info(f"Issues Fixed: {summary['issues_fixed']}")
        
        if summary['errors']:
            self.logger.error(f"Errors: {len(summary['errors'])}")
            for error in summary['errors']:
                self.logger.error(f"  - {error}")
        
        self.logger.info("\nTasks Performed:")
        for task in summary['tasks_performed']:
            self.logger.info(f"  - {task['task']}: {task['result'].get('status', 'completed')}")
        
        self.logger.info("=" * 60)
    
    async def check_maintenance_needed(self) -> bool:
        """Check if maintenance is needed based on various criteria.
        
        Returns:
            True if maintenance should be run
        """
        # Check when last maintenance was run
        if self.maintenance_history:
            last_run = datetime.fromisoformat(self.maintenance_history[-1]['timestamp'])
            hours_since_last = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
            
            # Run at least once every 24 hours
            if hours_since_last < 24:
                self.logger.info(f"Last maintenance run {hours_since_last:.1f} hours ago - not needed yet")
                return False
        
        # Always run if no history
        return True


async def main():
    """Main function to run scheduled maintenance."""
    import argparse
    
    # Load environment
    load_dotenv('.env.local')
    
    parser = argparse.ArgumentParser(description='Run scheduled repository maintenance')
    parser.add_argument('--force', action='store_true', help='Force maintenance even if recently run')
    parser.add_argument('--no-ai', action='store_true', help='Run without AI (basic maintenance only)')
    parser.add_argument('--check-only', action='store_true', help='Check if maintenance is needed without running')
    
    args = parser.parse_args()
    
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.error("❌ GITHUB_TOKEN not found in environment")
        return 1
    
    # Create maintenance runner
    maintenance = ScheduledRepositoryMaintenance(github_token, use_ai=not args.no_ai)
    
    # Check if maintenance is needed
    if args.check_only:
        needed = await maintenance.check_maintenance_needed()
        if needed:
            logger.info("✅ Maintenance is needed")
            return 0
        else:
            logger.info("ℹ️  Maintenance is not needed at this time")
            return 1
    
    # Check if we should run
    if not args.force:
        if not await maintenance.check_maintenance_needed():
            logger.info("ℹ️  Skipping maintenance - recently run")
            return 0
    
    # Run maintenance
    summary = await maintenance.run_maintenance()
    
    # Return appropriate exit code
    if summary['status'] == 'completed':
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)