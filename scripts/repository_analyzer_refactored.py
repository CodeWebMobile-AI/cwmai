"""
Repository Analyzer (Refactored)

Analyzes repositories by primarily reading from metadata.json files,
with fallback to direct repository analysis only when necessary.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

from scripts.repository_data_provider import (
    RepositoryDataProvider, 
    GitHubDataProvider, 
    LocalDataProvider
)
from scripts.storage_mode import StorageMode
from scripts.storage_factory import StorageFactory


class MetadataBasedAnalyzer:
    """Analyzes repositories using metadata.json as the primary source."""
    
    def __init__(self, data_provider: RepositoryDataProvider, logger: logging.Logger):
        self.data_provider = data_provider
        self.logger = logger
        self.base_path = Path("local_repos/repositories")
        
    def get_metadata(self, repo_full_name: str) -> Optional[Dict[str, Any]]:
        """Read metadata.json for a repository."""
        org, name = repo_full_name.split('/')
        metadata_path = self.base_path / org / name / "metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata.json for {repo_full_name}: {e}")
        
        return None
    
    def analyze_from_metadata(self, repo_full_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis from metadata.json content."""
        # Get basic repository info from data provider
        try:
            repo_info = self.data_provider.get_repository(repo_full_name)
        except Exception as e:
            self.logger.error(f"Failed to get repository info: {e}")
            repo_info = None
            
        # Build analysis from metadata
        analysis = {
            'repository': repo_full_name,
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            'metadata_based': True,
            'metadata_last_updated': metadata.get('last_updated', 'unknown')
        }
        
        # Basic info from repository
        if repo_info:
            analysis['basic_info'] = {
                'name': repo_info.name,
                'description': repo_info.description or metadata.get('description', ''),
                'language': metadata.get('code_metrics', {}).get('languages', {}).get('primary', 'Unknown'),
                'topics': repo_info.topics,
                'created_at': repo_info.created_at.isoformat(),
                'updated_at': repo_info.updated_at.isoformat(),
                'size': repo_info.size,
                'default_branch': repo_info.default_branch,
                'has_issues': repo_info.has_issues,
                'open_issues_count': repo_info.open_issues_count
            }
        else:
            # Fallback to metadata only
            analysis['basic_info'] = {
                'name': metadata.get('repository_name', repo_full_name.split('/')[-1]),
                'description': metadata.get('description', ''),
                'language': 'PHP',  # Default tech stack
                'topics': [],
                'created_at': metadata.get('created_at', datetime.now(timezone.utc).isoformat()),
                'updated_at': metadata.get('last_updated', datetime.now(timezone.utc).isoformat()),
                'size': int(metadata.get('code_metrics', {}).get('total_size_kb', 0)),
                'default_branch': 'main',
                'has_issues': True,
                'open_issues_count': 0
            }
            
        # Health metrics from metadata
        health_score = metadata.get('health_score', {})
        if isinstance(health_score, dict):
            analysis['health_metrics'] = {
                'health_score': health_score.get('score', 50),
                'health_category': health_score.get('category', 'unknown'),
                'health_factors': health_score.get('factors', [])
            }
        else:
            # Calculate basic health from metadata
            analysis['health_metrics'] = self._calculate_health_from_metadata(metadata)
            
        # Code analysis from metadata
        code_metrics = metadata.get('code_metrics', {})
        analysis['code_analysis'] = {
            'file_types': code_metrics.get('file_types', {}),
            'languages': code_metrics.get('languages', {}),
            'total_files': code_metrics.get('total_files', 0),
            'total_lines': code_metrics.get('total_lines', 0),
            'test_coverage': 'has_tests' if metadata.get('test_info', {}).get('has_tests') else 'unknown',
            'documentation': metadata.get('documentation', {}).get('documentation_files', [])
        }
        
        # Issues analysis - simplified since we're in local mode
        analysis['issues_analysis'] = {
            'total_open': 0,
            'bug_count': 0,
            'feature_requests': 0,
            'high_priority': 0,
            'recent_issues': []
        }
        
        # Technical stack from metadata
        analysis['technical_stack'] = {
            'primary_language': 'PHP',
            'frameworks': ['Laravel', 'React', 'TypeScript'],
            'dependencies': metadata.get('dependencies', {}).get('package_managers', []),
            'infrastructure': []
        }
        
        # Architecture and monetization from metadata
        analysis['architecture'] = metadata.get('architecture', {
            'document_exists': metadata.get('documentation', {}).get('has_architecture', False)
        })
        
        analysis['monetization_strategy'] = metadata.get('monetization_strategy', {
            'document_exists': metadata.get('documentation', {}).get('has_monetization', False)
        })
        
        # Improvement opportunities and needs from metadata
        analysis['specific_needs'] = metadata.get('identified_needs', [])
        analysis['improvement_opportunities'] = self._find_opportunities_from_metadata(metadata)
        
        return analysis
        
    def _calculate_health_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health metrics from metadata."""
        score = 50  # Base score
        
        # Documentation factors
        if metadata.get('readme', {}).get('exists'):
            score += 10
        if metadata.get('documentation', {}).get('has_license'):
            score += 5
            
        # Test factors  
        if metadata.get('test_info', {}).get('has_tests'):
            score += 15
            
        # Code quality factors
        if metadata.get('code_metrics', {}).get('total_files', 0) > 10:
            score += 5
            
        # Architecture
        if metadata.get('documentation', {}).get('has_architecture'):
            score += 10
            
        score = max(0, min(100, score))
        
        return {
            'health_score': score,
            'health_category': 'good' if score >= 70 else 'fair' if score >= 50 else 'poor',
            'needs_attention': score < 70
        }
        
    def _find_opportunities_from_metadata(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find improvement opportunities from metadata."""
        opportunities = []
        
        # CI/CD opportunity
        doc_files = metadata.get('documentation', {}).get('documentation_files', [])
        has_ci = any('.github/workflows' in f or 'ci' in f.lower() for f in doc_files)
        
        if not has_ci:
            opportunities.append({
                'type': 'automation',
                'description': 'Add GitHub Actions for CI/CD',
                'impact': 'high',
                'effort': 'low'
            })
            
        # Testing opportunity
        if not metadata.get('test_info', {}).get('has_tests'):
            opportunities.append({
                'type': 'testing',
                'description': 'Implement unit tests',
                'impact': 'high', 
                'effort': 'medium'
            })
            
        return opportunities


class RepositoryAnalyzer:
    """Main repository analyzer that orchestrates the analysis process."""
    
    def __init__(self, github_token: str = None, ai_brain=None, use_cache: bool = True, 
                 state_manager=None, storage_mode: str = None):
        """Initialize repository analyzer."""
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain
        self.use_cache = use_cache
        self.state_manager = state_manager
        
        # Determine storage mode
        self.storage_mode = storage_mode or os.getenv('STORAGE_MODE', StorageMode.GITHUB)
        self.storage_factory = StorageFactory(self.storage_mode)
        
        # Initialize data provider based on mode
        if self.storage_factory.is_local_mode():
            from scripts.local_github_simulator import LocalGitHubSimulator
            self.local_simulator = LocalGitHubSimulator()
            self.data_provider = LocalDataProvider(self.local_simulator)
            self.logger.info("RepositoryAnalyzer running in LOCAL mode")
        else:
            self.github_token = github_token or os.getenv('GITHUB_TOKEN')
            if self.github_token:
                from github import Github
                github_client = Github(self.github_token)
                self.data_provider = GitHubDataProvider(github_client)
            else:
                raise ValueError("GitHub token required for GitHub mode")
            self.logger.info("RepositoryAnalyzer running in GITHUB mode")
            
        # Initialize metadata analyzer
        self.metadata_analyzer = MetadataBasedAnalyzer(self.data_provider, self.logger)
        
    async def analyze_repository(self, repo_full_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Analyze a repository, preferring metadata.json when available."""
        print(f"Analyzing repository: {repo_full_name}")
        
        try:
            # First, try to get analysis from metadata
            metadata = self.metadata_analyzer.get_metadata(repo_full_name)
            
            if metadata and not force_refresh:
                # Check if metadata is recent enough (within 24 hours)
                last_updated = metadata.get('last_updated')
                if last_updated:
                    try:
                        updated_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        age_hours = (datetime.now(timezone.utc) - updated_time).total_seconds() / 3600
                        
                        if age_hours < 24:
                            print(f"Using metadata-based analysis for: {repo_full_name}")
                            analysis = self.metadata_analyzer.analyze_from_metadata(repo_full_name, metadata)
                            
                            # Add AI insights if available
                            if self.ai_brain:
                                analysis['ai_insights'] = await self._generate_ai_insights(analysis)
                                
                            return analysis
                    except Exception as e:
                        self.logger.warning(f"Failed to parse metadata timestamp: {e}")
            
            # If metadata not available or too old, fall back to direct analysis
            print(f"Metadata not available or outdated, performing direct analysis")
            return await self._perform_direct_analysis(repo_full_name)
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository {repo_full_name}: {e}")
            return {
                'repository': repo_full_name,
                'error': str(e),
                'analyzed_at': datetime.now(timezone.utc).isoformat()
            }
            
    async def _perform_direct_analysis(self, repo_full_name: str) -> Dict[str, Any]:
        """Perform direct repository analysis (fallback method)."""
        # This would contain the original analysis logic
        # For now, return a minimal analysis
        return {
            'repository': repo_full_name,
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            'direct_analysis': True,
            'message': 'Direct analysis not fully implemented in refactored version'
        }
        
    async def _generate_ai_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights about the repository."""
        prompt = f"""
        Analyze this repository and provide strategic insights:
        
        Repository: {analysis.get('repository')}
        Health Score: {analysis.get('health_metrics', {}).get('health_score', 0)}
        Total Files: {analysis.get('code_analysis', {}).get('total_files', 0)}
        Has Tests: {analysis.get('code_analysis', {}).get('test_coverage') == 'has_tests'}
        
        Specific Needs: {json.dumps(analysis.get('specific_needs', []), indent=2)}
        
        Provide insights on:
        1. What is the most critical need for this repository right now?
        2. What type of tasks would provide the most value?
        3. What should be the immediate priority?
        
        Return strategic insights as JSON with keys: critical_need, high_value_tasks, immediate_priority
        """
        
        try:
            response = await self.ai_brain.generate_response(prompt)
            # Parse response and return insights
            return {'ai_generated': True, 'insights': response}
        except Exception as e:
            self.logger.error(f"Failed to generate AI insights: {e}")
            return {}