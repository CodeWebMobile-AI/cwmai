"""
Repository Analysis Cache

Caches and persists deep AI analysis results for repositories to avoid
redundant analysis and track changes over time.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class RepositoryAnalysisCache:
    """Manages caching and persistence of repository analysis results."""
    
    def __init__(self, cache_dir: str = "repository_analysis_cache", 
                 cache_duration_hours: int = 24):
        """Initialize the analysis cache.
        
        Args:
            cache_dir: Directory to store cached analysis
            cache_duration_hours: How long to consider cache valid
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories for different analysis types
        (self.cache_dir / "lifecycle").mkdir(exist_ok=True)
        (self.cache_dir / "code_quality").mkdir(exist_ok=True)
        (self.cache_dir / "ai_insights").mkdir(exist_ok=True)
        
    def get_cache_path(self, repo_name: str, analysis_type: str) -> Path:
        """Get the cache file path for a repository analysis.
        
        Args:
            repo_name: Repository name (owner/repo)
            analysis_type: Type of analysis (lifecycle, code_quality, ai_insights)
            
        Returns:
            Path to cache file
        """
        # Replace / with _ for filesystem compatibility
        safe_name = repo_name.replace("/", "_")
        return self.cache_dir / analysis_type / f"{safe_name}.json"
    
    def save_analysis(self, repo_name: str, analysis: Dict[str, Any]) -> None:
        """Save complete analysis results with deep AI insights.
        
        Args:
            repo_name: Repository name
            analysis: Complete analysis results
        """
        try:
            # Save lifecycle analysis
            if 'lifecycle_analysis' in analysis:
                self._save_component(
                    repo_name, 
                    'lifecycle',
                    analysis['lifecycle_analysis']
                )
            
            # Save code quality assessment
            if 'code_analysis' in analysis:
                self._save_component(
                    repo_name,
                    'code_quality',
                    {
                        **analysis['code_analysis'],
                        'health_metrics': analysis.get('health_metrics', {}),
                        'technical_stack': analysis.get('technical_stack', {})
                    }
                )
            
            # Save AI insights
            if 'ai_insights' in analysis:
                self._save_component(
                    repo_name,
                    'ai_insights',
                    analysis['ai_insights']
                )
            
            # Save complete analysis
            complete_path = self.get_cache_path(repo_name, "complete")
            complete_path.parent.mkdir(exist_ok=True)
            
            cached_analysis = {
                **analysis,
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'cache_version': '1.0'
            }
            
            with open(complete_path, 'w') as f:
                json.dump(cached_analysis, f, indent=2)
                
            self.logger.info(f"Saved analysis cache for {repo_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis cache for {repo_name}: {e}")
    
    def _save_component(self, repo_name: str, component_type: str, 
                       data: Dict[str, Any]) -> None:
        """Save a specific component of the analysis.
        
        Args:
            repo_name: Repository name
            component_type: Type of component
            data: Component data
        """
        cache_path = self.get_cache_path(repo_name, component_type)
        
        cached_data = {
            'repository': repo_name,
            'component_type': component_type,
            'data': data,
            'cached_at': datetime.now(timezone.utc).isoformat()
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cached_data, f, indent=2)
    
    def get_analysis(self, repo_name: str, 
                    force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get cached analysis if available and valid.
        
        Args:
            repo_name: Repository name
            force_refresh: Ignore cache and return None
            
        Returns:
            Cached analysis or None if not available/expired
        """
        if force_refresh:
            return None
            
        complete_path = self.get_cache_path(repo_name, "complete")
        
        if not complete_path.exists():
            return None
            
        try:
            with open(complete_path, 'r') as f:
                cached = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached['cached_at'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) - cached_time > self.cache_duration:
                self.logger.info(f"Cache expired for {repo_name}")
                return None
                
            self.logger.info(f"Using cached analysis for {repo_name}")
            return cached
            
        except Exception as e:
            self.logger.error(f"Error reading cache for {repo_name}: {e}")
            return None
    
    def get_component(self, repo_name: str, component_type: str) -> Optional[Dict[str, Any]]:
        """Get a specific cached component.
        
        Args:
            repo_name: Repository name
            component_type: Component type
            
        Returns:
            Cached component data or None
        """
        cache_path = self.get_cache_path(repo_name, component_type)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached['cached_at'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) - cached_time > self.cache_duration:
                return None
                
            return cached['data']
            
        except Exception as e:
            self.logger.error(f"Error reading {component_type} cache for {repo_name}: {e}")
            return None
    
    def get_lifecycle_history(self, repo_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical lifecycle analysis for tracking progression.
        
        Args:
            repo_name: Repository name
            limit: Maximum number of historical entries
            
        Returns:
            List of lifecycle analyses over time
        """
        history_path = self.cache_dir / "lifecycle_history" / f"{repo_name.replace('/', '_')}.json"
        
        if not history_path.exists():
            return []
            
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            return history[-limit:]
        except Exception as e:
            self.logger.error(f"Error reading lifecycle history for {repo_name}: {e}")
            return []
    
    def save_lifecycle_history(self, repo_name: str, lifecycle_data: Dict[str, Any]) -> None:
        """Append lifecycle analysis to history for tracking progression.
        
        Args:
            repo_name: Repository name
            lifecycle_data: Lifecycle analysis data
        """
        history_dir = self.cache_dir / "lifecycle_history"
        history_dir.mkdir(exist_ok=True)
        
        history_path = history_dir / f"{repo_name.replace('/', '_')}.json"
        
        # Load existing history
        history = []
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except:
                pass
        
        # Append new entry
        history.append({
            **lifecycle_data,
            'recorded_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving lifecycle history for {repo_name}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache.
        
        Returns:
            Cache statistics
        """
        stats = {
            'total_cached_repos': 0,
            'lifecycle_analyses': 0,
            'code_quality_assessments': 0,
            'ai_insights': 0,
            'total_cache_size_mb': 0
        }
        
        # Count files and calculate size
        total_size = 0
        for path in self.cache_dir.rglob('*.json'):
            if path.is_file():
                total_size += path.stat().st_size
                
                if 'lifecycle' in str(path):
                    stats['lifecycle_analyses'] += 1
                elif 'code_quality' in str(path):
                    stats['code_quality_assessments'] += 1
                elif 'ai_insights' in str(path):
                    stats['ai_insights'] += 1
                elif 'complete' in str(path):
                    stats['total_cached_repos'] += 1
        
        stats['total_cache_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return stats
    
    def clear_cache(self, repo_name: Optional[str] = None) -> None:
        """Clear cache for a specific repository or all repositories.
        
        Args:
            repo_name: Repository name or None to clear all
        """
        if repo_name:
            # Clear specific repository
            safe_name = repo_name.replace("/", "_")
            for path in self.cache_dir.rglob(f"*{safe_name}*"):
                if path.is_file():
                    path.unlink()
            self.logger.info(f"Cleared cache for {repo_name}")
        else:
            # Clear all cache
            for path in self.cache_dir.rglob('*.json'):
                if path.is_file():
                    path.unlink()
            self.logger.info("Cleared all repository analysis cache")