"""
Enhanced Duplicate Detector

Unified duplicate detection for both local and GitHub modes.
Prevents duplicate project creation across the entire portfolio.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from github import Github, Auth

from scripts.duplicate_prevention_system import ProjectMetadata
from scripts.intelligent_duplicate_detector import IntelligentDuplicateDetector


class EnhancedDuplicateDetector(IntelligentDuplicateDetector):
    """Enhanced duplicate detector that works with both local and GitHub repositories."""
    
    def __init__(self, local_repos_path: Path, ai_brain, github_token: Optional[str] = None):
        """Initialize enhanced duplicate detector.
        
        Args:
            local_repos_path: Path to local repositories directory
            ai_brain: AI brain for intelligent analysis
            github_token: GitHub token for fetching remote repositories
        """
        super().__init__(local_repos_path, ai_brain)
        
        self.github_token = github_token or os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        self.github_client = None
        
        if self.github_token:
            try:
                auth = Auth.Token(self.github_token)
                self.github_client = Github(auth=auth)
                self.logger.info("GitHub client initialized for duplicate detection")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GitHub client: {e}")
    
    def load_all_metadata(self) -> Dict[str, ProjectMetadata]:
        """Load metadata from all repositories (local and GitHub)."""
        # First load local repositories
        super().load_all_metadata()
        
        # Then load GitHub repositories if available
        if self.github_client:
            self._load_github_repositories()
        
        self.logger.info(f"Loaded metadata for {len(self._metadata_cache)} total projects")
        return self._metadata_cache
    
    def _load_github_repositories(self):
        """Load repositories from GitHub organization."""
        try:
            org_name = "CodeWebMobile-AI"
            org = self.github_client.get_organization(org_name)
            
            self.logger.info(f"Loading repositories from GitHub organization: {org_name}")
            
            for repo in org.get_repos():
                # Skip if already loaded from local
                if repo.name in self._metadata_cache:
                    continue
                
                # Create metadata from GitHub repo
                metadata = ProjectMetadata(
                    repository_name=repo.name,
                    display_name=repo.name,
                    description=repo.description or "",
                    tech_stack="Unknown",  # Can't easily determine from GitHub API
                    project_type="Unknown",
                    key_features=[],  # Would need to analyze repo content
                    created_at=repo.created_at.isoformat() if repo.created_at else datetime.now().isoformat(),
                    created_by="GitHub",
                    problem_domain=repo.description or "",
                    target_audience="",
                    monetization_model="",
                    core_functionality=repo.description or ""
                )
                
                # Extract more info from topics if available
                topics = list(repo.get_topics())
                if topics:
                    metadata.key_features = topics[:5]  # Use first 5 topics as features
                
                self._metadata_cache[repo.name] = metadata
                self.logger.debug(f"Loaded GitHub repository: {repo.name}")
                
        except Exception as e:
            self.logger.error(f"Error loading GitHub repositories: {e}")
    
    async def check_duplicate_with_enhanced_matching(
        self,
        new_project: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for duplicates with enhanced matching across all sources.
        
        Args:
            new_project: Dictionary with project details
            
        Returns:
            Enhanced duplicate detection results
        """
        # Use the parent class method but ensure all repos are loaded
        self.load_all_metadata()  # This now loads both local and GitHub
        
        # First check exact name matching
        project_name = new_project.get('name', '').lower()
        for repo_name in self._metadata_cache.keys():
            if repo_name.lower() == project_name:
                return {
                    'is_duplicate': True,
                    'duplicate_confidence': 1.0,
                    'ai_analysis': f"Exact name match found: {repo_name}",
                    'recommendation': 'exact_duplicate',
                    'similar_to': repo_name
                }
        
        # Use simple AI duplicate check instead of complex one
        from scripts.simple_duplicate_check import simple_ai_duplicate_check
        
        # Build summary of existing projects
        projects_summary = []
        for name, metadata in list(self._metadata_cache.items())[:20]:  # Limit to 20 to avoid token limits
            desc = metadata.description if hasattr(metadata, 'description') else 'No description'
            projects_summary.append(f"- {name}: {desc}")
        
        existing_summary = '\n'.join(projects_summary)
        
        # Perform simple AI check
        result = await simple_ai_duplicate_check(self.ai_brain, new_project, existing_summary)
        
        # Ensure backward compatibility
        if 'similar_projects' not in result:
            result['similar_projects'] = []
            if result.get('similar_to') and result['similar_to'] != 'none':
                result['similar_projects'] = [{
                    'name': result['similar_to'],
                    'similarity_score': result.get('duplicate_confidence', 0.0)
                }]
        
        return result