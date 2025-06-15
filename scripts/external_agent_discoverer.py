"""
External Agent Discoverer

Discovers and analyzes external AI agent repositories from GitHub and other sources.
Identifies valuable capabilities, patterns, and architectures that can be integrated
into CWMAI's self-improvement system.
"""

import os
import json
import time
import hashlib
import asyncio
import tempfile
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
import git
import ast
import re
from collections import defaultdict
import logging
from pathlib import Path

# Import CWMAI components
from repository_exclusion import should_process_repo
from state_manager import StateManager


class CapabilityType(Enum):
    """Types of capabilities that can be discovered."""
    TASK_ORCHESTRATION = "task_orchestration"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    AUTONOMOUS_DECISION_MAKING = "autonomous_decision_making"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_HANDLING = "error_handling"
    SECURITY_PATTERNS = "security_patterns"
    API_INTEGRATION = "api_integration"
    DATA_PROCESSING = "data_processing"
    LEARNING_ALGORITHMS = "learning_algorithms"


@dataclass
class RepositoryAnalysis:
    """Analysis results for an external repository."""
    url: str
    name: str
    description: str
    language: str
    stars: int
    forks: int
    last_updated: str
    health_score: float
    capabilities: List[CapabilityType]
    architecture_patterns: List[str]
    key_files: List[str]
    integration_difficulty: float
    license: str
    documentation_quality: float
    test_coverage: float
    performance_indicators: Dict[str, Any]
    security_assessment: Dict[str, Any]
    compatibility_score: float
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiscoveryConfig:
    """Configuration for external agent discovery."""
    github_token: Optional[str] = None
    max_repositories_per_scan: int = 50
    min_stars: int = 10
    max_age_days: int = 365
    required_languages: Set[str] = field(default_factory=lambda: {'Python', 'JavaScript', 'TypeScript'})
    search_topics: List[str] = field(default_factory=lambda: [
        'ai-agents', 'autonomous-ai', 'multi-agent-systems', 'swarm-intelligence',
        'ai-orchestration', 'intelligent-automation', 'ai-task-management',
        'agent-coordination', 'distributed-ai', 'ai-workflow'
    ])
    excluded_patterns: List[str] = field(default_factory=lambda: [
        'tutorial', 'example', 'demo', 'toy', 'simple', 'basic',
        'hello-world', 'getting-started', 'beginner'
    ])


class ExternalAgentDiscoverer:
    """Discovers and analyzes external AI agent repositories."""
    
    def __init__(self, config: Optional[DiscoveryConfig] = None, state_manager: Optional[StateManager] = None):
        """Initialize the external agent discoverer.
        
        Args:
            config: Discovery configuration
            state_manager: State manager for persistence
        """
        self.config = config or DiscoveryConfig()
        self.state_manager = state_manager or StateManager()
        
        # GitHub API setup
        self.github_token = self.config.github_token or os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        self.github_headers = {}
        if self.github_token:
            self.github_headers['Authorization'] = f'token {self.github_token}'
        
        # Discovery state
        self.discovered_repositories: Dict[str, RepositoryAnalysis] = {}
        self.discovery_history: List[Dict[str, Any]] = []
        self.last_discovery_time: Optional[datetime] = None
        
        # Cache and storage
        self.cache_dir = Path('.external_agent_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load previous discoveries
        self._load_discovery_state()
    
    async def discover_trending_agents(self) -> List[RepositoryAnalysis]:
        """Discover trending AI agent repositories.
        
        Returns:
            List of analyzed repositories sorted by potential value
        """
        self.logger.info("Starting discovery of trending AI agent repositories")
        
        discovered_repos = []
        
        # Search each topic
        for topic in self.config.search_topics:
            try:
                repos = await self._search_github_topic(topic)
                discovered_repos.extend(repos)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error searching topic {topic}: {e}")
        
        # Remove duplicates and filter
        unique_repos = self._deduplicate_repositories(discovered_repos)
        filtered_repos = await self._filter_and_analyze_repositories(unique_repos)
        
        # Sort by potential value
        valuable_repos = sorted(filtered_repos, key=self._calculate_repository_value, reverse=True)
        
        # Limit to configured maximum
        final_repos = valuable_repos[:self.config.max_repositories_per_scan]
        
        # Update discovery state
        self._update_discovery_state(final_repos)
        
        self.logger.info(f"Discovered {len(final_repos)} valuable AI agent repositories")
        
        return final_repos
    
    async def analyze_repository_capabilities(self, repo_url: str) -> Optional[RepositoryAnalysis]:
        """Analyze a specific repository for capabilities and patterns.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Repository analysis or None if analysis failed
        """
        self.logger.info(f"Analyzing repository: {repo_url}")
        
        try:
            # Get repository metadata
            repo_info = await self._get_repository_info(repo_url)
            if not repo_info:
                return None
            
            # Clone repository for analysis
            analysis_dir = await self._clone_repository_safely(repo_url)
            if not analysis_dir:
                return None
            
            try:
                # Perform comprehensive analysis
                analysis = RepositoryAnalysis(
                    url=repo_url,
                    name=repo_info['name'],
                    description=repo_info.get('description', ''),
                    language=repo_info.get('language', 'Unknown'),
                    stars=repo_info.get('stargazers_count', 0),
                    forks=repo_info.get('forks_count', 0),
                    last_updated=repo_info.get('updated_at', ''),
                    health_score=0.0,
                    capabilities=[],
                    architecture_patterns=[],
                    key_files=[],
                    integration_difficulty=0.0,
                    license=repo_info.get('license', {}).get('name', 'Unknown') if repo_info.get('license') else 'Unknown',
                    documentation_quality=0.0,
                    test_coverage=0.0,
                    performance_indicators={},
                    security_assessment={},
                    compatibility_score=0.0
                )
                
                # Analyze repository contents
                await self._analyze_repository_structure(analysis_dir, analysis)
                await self._detect_capabilities(analysis_dir, analysis)
                await self._assess_architecture_patterns(analysis_dir, analysis)
                await self._evaluate_integration_difficulty(analysis_dir, analysis)
                await self._assess_quality_metrics(analysis_dir, analysis)
                
                # Calculate final scores
                analysis.health_score = self._calculate_health_score(analysis)
                analysis.compatibility_score = self._calculate_compatibility_score(analysis)
                
                return analysis
                
            finally:
                # Cleanup cloned repository
                import shutil
                if os.path.exists(analysis_dir):
                    shutil.rmtree(analysis_dir)
        
        except Exception as e:
            self.logger.error(f"Error analyzing repository {repo_url}: {e}")
            return None
    
    async def get_capability_recommendations(self, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations for capabilities to integrate based on current performance.
        
        Args:
            current_performance: Current system performance metrics
            
        Returns:
            List of capability recommendations
        """
        recommendations = []
        
        # Analyze performance gaps
        gaps = self._identify_performance_gaps(current_performance)
        
        # Find repositories with capabilities that address these gaps
        for gap in gaps:
            matching_repos = self._find_repositories_for_capability(gap['capability_type'])
            
            for repo in matching_repos[:3]:  # Top 3 for each gap
                recommendation = {
                    'repository': repo.name,
                    'url': repo.url,
                    'capability_type': gap['capability_type'],
                    'addresses_gap': gap['description'],
                    'integration_difficulty': repo.integration_difficulty,
                    'expected_improvement': gap['potential_improvement'],
                    'compatibility_score': repo.compatibility_score,
                    'priority': self._calculate_recommendation_priority(gap, repo)
                }
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def _search_github_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search GitHub for repositories with a specific topic."""
        url = "https://api.github.com/search/repositories"
        
        # Build search query
        query_parts = [
            f"topic:{topic}",
            f"language:Python",
            f"stars:>{self.config.min_stars}",
            "archived:false"
        ]
        
        # Add date filter
        since_date = (datetime.now() - timedelta(days=self.config.max_age_days)).strftime('%Y-%m-%d')
        query_parts.append(f"pushed:>{since_date}")
        
        query = " ".join(query_parts)
        
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 20
        }
        
        try:
            response = requests.get(url, headers=self.github_headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            repositories = data.get('items', [])
            
            self.logger.debug(f"Found {len(repositories)} repositories for topic: {topic}")
            
            return repositories
            
        except Exception as e:
            self.logger.error(f"Error searching GitHub topic {topic}: {e}")
            return []
    
    async def _get_repository_info(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed repository information from GitHub API."""
        # Extract owner and repo name from URL
        if 'github.com/' not in repo_url:
            return None
        
        parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
        if len(parts) < 2:
            return None
        
        owner, repo = parts[0], parts[1]
        
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            response = requests.get(api_url, headers=self.github_headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting repository info for {repo_url}: {e}")
            return None
    
    async def _clone_repository_safely(self, repo_url: str) -> Optional[str]:
        """Clone repository to a temporary directory for analysis."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix='agent_analysis_')
            
            # Clone with depth limit for security
            result = subprocess.run([
                'git', 'clone', '--depth', '1', '--quiet', repo_url, temp_dir
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return temp_dir
            else:
                self.logger.warning(f"Failed to clone {repo_url}: {result.stderr}")
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                return None
                
        except Exception as e:
            self.logger.error(f"Error cloning repository {repo_url}: {e}")
            return None
    
    async def _analyze_repository_structure(self, repo_path: str, analysis: RepositoryAnalysis):
        """Analyze repository structure and identify key files."""
        key_files = []
        
        # Look for important files
        important_patterns = [
            ('README.md', 'documentation'),
            ('requirements.txt', 'dependencies'),
            ('setup.py', 'package_setup'),
            ('Dockerfile', 'containerization'),
            ('docker-compose.yml', 'orchestration'),
            ('main.py', 'entry_point'),
            ('app.py', 'application'),
            ('agent.py', 'agent_core'),
            ('swarm.py', 'swarm_logic'),
            ('coordinator.py', 'coordination'),
            ('orchestrator.py', 'orchestration'),
            ('scheduler.py', 'scheduling'),
            ('task_manager.py', 'task_management'),
            ('ai_brain.py', 'ai_intelligence'),
            ('config.py', 'configuration'),
            ('settings.py', 'settings')
        ]
        
        # Find Python files for analysis
        python_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Check for important files
                for pattern, file_type in important_patterns:
                    if file.lower() == pattern.lower():
                        key_files.append({
                            'path': relative_path,
                            'type': file_type,
                            'size': os.path.getsize(file_path)
                        })
                
                # Collect Python files
                if file.endswith('.py'):
                    python_files.append(relative_path)
        
        analysis.key_files = key_files
        
        # Store Python files for capability detection
        analysis.performance_indicators['python_files'] = python_files
        analysis.performance_indicators['total_files'] = len(python_files)
    
    async def _detect_capabilities(self, repo_path: str, analysis: RepositoryAnalysis):
        """Detect AI agent capabilities in the repository."""
        capabilities = set()
        
        # Capability detection patterns
        capability_patterns = {
            CapabilityType.TASK_ORCHESTRATION: [
                r'task.*orchestrat', r'workflow.*manag', r'job.*schedul',
                r'task.*queue', r'task.*manag', r'pipeline.*execut'
            ],
            CapabilityType.MULTI_AGENT_COORDINATION: [
                r'multi.*agent', r'agent.*coordination', r'agent.*communication',
                r'distributed.*agent', r'agent.*network', r'consensus'
            ],
            CapabilityType.SWARM_INTELLIGENCE: [
                r'swarm', r'collective.*intelligence', r'emergent.*behavior',
                r'particle.*swarm', r'ant.*colony', r'bee.*algorithm'
            ],
            CapabilityType.AUTONOMOUS_DECISION_MAKING: [
                r'autonomous.*decision', r'decision.*tree', r'policy.*learning',
                r'reinforcement.*learning', r'q.*learning', r'decision.*making'
            ],
            CapabilityType.PERFORMANCE_OPTIMIZATION: [
                r'performance.*optim', r'resource.*manag', r'load.*balanc',
                r'cache', r'memory.*optim', r'speed.*up'
            ],
            CapabilityType.ERROR_HANDLING: [
                r'error.*handling', r'exception.*manag', r'fault.*toleran',
                r'retry.*logic', r'circuit.*breaker', r'recovery'
            ],
            CapabilityType.SECURITY_PATTERNS: [
                r'security', r'authentication', r'authorization',
                r'encryption', r'secure.*communication', r'access.*control'
            ],
            CapabilityType.API_INTEGRATION: [
                r'api.*integration', r'rest.*api', r'webhook',
                r'http.*client', r'api.*wrapper', r'service.*client'
            ],
            CapabilityType.DATA_PROCESSING: [
                r'data.*processing', r'data.*pipeline', r'etl',
                r'data.*transform', r'data.*clean', r'batch.*process'
            ],
            CapabilityType.LEARNING_ALGORITHMS: [
                r'machine.*learning', r'neural.*network', r'deep.*learning',
                r'training', r'model.*learning', r'adaptive.*learning'
            ]
        }
        
        # Scan Python files
        python_files = analysis.performance_indicators.get('python_files', [])
        
        for py_file in python_files:
            file_path = os.path.join(repo_path, py_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for capability patterns
                content_lower = content.lower()
                
                for capability_type, patterns in capability_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content_lower):
                            capabilities.add(capability_type)
                            break
                
            except Exception as e:
                self.logger.debug(f"Error reading file {py_file}: {e}")
        
        analysis.capabilities = list(capabilities)
    
    async def _assess_architecture_patterns(self, repo_path: str, analysis: RepositoryAnalysis):
        """Assess architecture patterns used in the repository."""
        patterns = set()
        
        # Architecture pattern indicators
        pattern_indicators = {
            'microservices': ['docker', 'kubernetes', 'service', 'api'],
            'event_driven': ['event', 'message', 'queue', 'publish', 'subscribe'],
            'plugin_architecture': ['plugin', 'extension', 'module', 'component'],
            'layered_architecture': ['layer', 'tier', 'mvc', 'mvp'],
            'actor_model': ['actor', 'message_passing', 'concurrent'],
            'pipeline_pattern': ['pipeline', 'stage', 'transform', 'filter'],
            'observer_pattern': ['observer', 'listener', 'callback', 'notify'],
            'strategy_pattern': ['strategy', 'algorithm', 'policy'],
            'factory_pattern': ['factory', 'builder', 'creator'],
            'singleton_pattern': ['singleton', 'instance'],
            'dependency_injection': ['inject', 'container', 'provider']
        }
        
        # Scan for patterns
        python_files = analysis.performance_indicators.get('python_files', [])
        
        for py_file in python_files[:20]:  # Limit to first 20 files for performance
            file_path = os.path.join(repo_path, py_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                content_lower = content.lower()
                
                for pattern_name, indicators in pattern_indicators.items():
                    if any(indicator in content_lower for indicator in indicators):
                        patterns.add(pattern_name)
                
            except Exception as e:
                self.logger.debug(f"Error analyzing architecture in {py_file}: {e}")
        
        analysis.architecture_patterns = list(patterns)
    
    async def _evaluate_integration_difficulty(self, repo_path: str, analysis: RepositoryAnalysis):
        """Evaluate how difficult it would be to integrate capabilities."""
        difficulty_factors = {
            'complexity': 0.0,
            'dependencies': 0.0,
            'compatibility': 0.0,
            'documentation': 0.0,
            'testing': 0.0
        }
        
        # Check complexity
        python_files = analysis.performance_indicators.get('python_files', [])
        total_lines = 0
        complex_files = 0
        
        for py_file in python_files[:10]:  # Sample files
            file_path = os.path.join(repo_path, py_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    if len(lines) > 500:  # Complex file
                        complex_files += 1
                
            except Exception:
                pass
        
        if python_files:
            avg_lines_per_file = total_lines / min(len(python_files), 10)
            complexity_ratio = complex_files / min(len(python_files), 10)
            
            difficulty_factors['complexity'] = min(1.0, avg_lines_per_file / 200.0)
            difficulty_factors['complexity'] += complexity_ratio * 0.5
        
        # Check dependencies
        requirements_file = os.path.join(repo_path, 'requirements.txt')
        if os.path.exists(requirements_file):
            try:
                with open(requirements_file, 'r') as f:
                    deps = f.readlines()
                    # More dependencies = higher difficulty
                    difficulty_factors['dependencies'] = min(1.0, len(deps) / 20.0)
            except Exception:
                pass
        
        # Check documentation
        readme_files = [f for f in analysis.key_files if f['type'] == 'documentation']
        if readme_files:
            difficulty_factors['documentation'] = 0.2  # Has documentation
        else:
            difficulty_factors['documentation'] = 0.8  # No documentation
        
        # Check testing
        test_files = [f for f in python_files if 'test' in f.lower()]
        if test_files:
            test_ratio = len(test_files) / len(python_files)
            difficulty_factors['testing'] = max(0.1, 1.0 - test_ratio)
        else:
            difficulty_factors['testing'] = 0.9  # No tests
        
        # Calculate overall difficulty
        overall_difficulty = sum(difficulty_factors.values()) / len(difficulty_factors)
        analysis.integration_difficulty = min(1.0, overall_difficulty)
    
    async def _assess_quality_metrics(self, repo_path: str, analysis: RepositoryAnalysis):
        """Assess repository quality metrics."""
        # Documentation quality
        readme_files = [f for f in analysis.key_files if f['type'] == 'documentation']
        if readme_files:
            readme_path = os.path.join(repo_path, readme_files[0]['path'])
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read()
                
                # Simple documentation quality assessment
                quality_indicators = ['installation', 'usage', 'example', 'api', 'configuration']
                quality_score = sum(1 for indicator in quality_indicators if indicator in readme_content.lower())
                analysis.documentation_quality = quality_score / len(quality_indicators)
            except Exception:
                analysis.documentation_quality = 0.1
        else:
            analysis.documentation_quality = 0.0
        
        # Test coverage estimation
        python_files = analysis.performance_indicators.get('python_files', [])
        test_files = [f for f in python_files if 'test' in f.lower()]
        
        if python_files:
            analysis.test_coverage = len(test_files) / len(python_files)
        else:
            analysis.test_coverage = 0.0
        
        # Security assessment
        security_issues = 0
        
        # Check for potential security issues
        for py_file in python_files[:5]:  # Sample files
            file_path = os.path.join(repo_path, py_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for potential security issues
                security_patterns = [
                    r'exec\s*\(',
                    r'eval\s*\(',
                    r'subprocess\.call',
                    r'os\.system',
                    r'shell=True'
                ]
                
                for pattern in security_patterns:
                    if re.search(pattern, content):
                        security_issues += 1
                
            except Exception:
                pass
        
        analysis.security_assessment = {
            'potential_issues': security_issues,
            'security_score': max(0.0, 1.0 - (security_issues / 10.0))
        }
    
    def _calculate_health_score(self, analysis: RepositoryAnalysis) -> float:
        """Calculate overall repository health score."""
        factors = {
            'stars': min(1.0, analysis.stars / 100.0),
            'recent_activity': 1.0 if analysis.last_updated else 0.0,
            'documentation': analysis.documentation_quality,
            'testing': analysis.test_coverage,
            'security': analysis.security_assessment.get('security_score', 0.5),
            'license': 1.0 if analysis.license != 'Unknown' else 0.0,
            'capabilities': min(1.0, len(analysis.capabilities) / 5.0)
        }
        
        return sum(factors.values()) / len(factors)
    
    def _calculate_compatibility_score(self, analysis: RepositoryAnalysis) -> float:
        """Calculate compatibility score with CWMAI architecture."""
        score = 0.5  # Base score
        
        # Language compatibility
        if analysis.language == 'Python':
            score += 0.3
        
        # Architecture pattern compatibility
        compatible_patterns = {
            'plugin_architecture', 'pipeline_pattern', 'observer_pattern',
            'strategy_pattern', 'factory_pattern', 'dependency_injection'
        }
        
        matching_patterns = set(analysis.architecture_patterns) & compatible_patterns
        score += (len(matching_patterns) / len(compatible_patterns)) * 0.2
        
        # Capability relevance
        high_value_capabilities = {
            CapabilityType.TASK_ORCHESTRATION,
            CapabilityType.MULTI_AGENT_COORDINATION,
            CapabilityType.PERFORMANCE_OPTIMIZATION,
            CapabilityType.ERROR_HANDLING
        }
        
        matching_capabilities = set(analysis.capabilities) & high_value_capabilities
        score += (len(matching_capabilities) / len(high_value_capabilities)) * 0.3
        
        return min(1.0, score)
    
    def _calculate_repository_value(self, analysis: RepositoryAnalysis) -> float:
        """Calculate overall value score for prioritization."""
        weights = {
            'health_score': 0.3,
            'compatibility_score': 0.25,
            'capability_count': 0.2,
            'stars_normalized': 0.15,
            'low_integration_difficulty': 0.1
        }
        
        scores = {
            'health_score': analysis.health_score,
            'compatibility_score': analysis.compatibility_score,
            'capability_count': min(1.0, len(analysis.capabilities) / 5.0),
            'stars_normalized': min(1.0, analysis.stars / 500.0),
            'low_integration_difficulty': 1.0 - analysis.integration_difficulty
        }
        
        return sum(weights[factor] * scores[factor] for factor in weights)
    
    def _deduplicate_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate repositories."""
        seen_urls = set()
        unique_repos = []
        
        for repo in repositories:
            url = repo.get('html_url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_repos.append(repo)
        
        return unique_repos
    
    async def _filter_and_analyze_repositories(self, repositories: List[Dict[str, Any]]) -> List[RepositoryAnalysis]:
        """Filter and analyze repositories."""
        analyses = []
        
        for repo_info in repositories:
            # Basic filtering
            if not self._should_analyze_repository(repo_info):
                continue
            
            # Analyze repository
            analysis = await self.analyze_repository_capabilities(repo_info['html_url'])
            if analysis and analysis.health_score > 0.3:  # Minimum quality threshold
                analyses.append(analysis)
        
        return analyses
    
    def _should_analyze_repository(self, repo_info: Dict[str, Any]) -> bool:
        """Check if repository should be analyzed."""
        # Check exclusion patterns
        name = repo_info.get('name', '').lower()
        description = repo_info.get('description', '').lower()
        
        for pattern in self.config.excluded_patterns:
            if pattern in name or pattern in description:
                return False
        
        # Check if repository is processable
        if not should_process_repo(repo_info.get('full_name', '')):
            return False
        
        # Check minimum requirements
        if repo_info.get('stargazers_count', 0) < self.config.min_stars:
            return False
        
        if repo_info.get('language') not in self.config.required_languages:
            return False
        
        return True
    
    def _identify_performance_gaps(self, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance gaps that could be addressed by external capabilities."""
        gaps = []
        
        # Task completion rate gap
        task_completion = current_performance.get('task_completion_rate', 0)
        if task_completion < 0.8:
            gaps.append({
                'capability_type': CapabilityType.TASK_ORCHESTRATION,
                'description': 'Low task completion rate',
                'current_value': task_completion,
                'target_value': 0.9,
                'potential_improvement': 0.9 - task_completion
            })
        
        # Error handling gap
        error_rate = current_performance.get('error_rate', 0)
        if error_rate > 0.1:
            gaps.append({
                'capability_type': CapabilityType.ERROR_HANDLING,
                'description': 'High error rate',
                'current_value': error_rate,
                'target_value': 0.05,
                'potential_improvement': error_rate - 0.05
            })
        
        # Performance optimization gap
        response_time = current_performance.get('avg_response_time', 5.0)
        if response_time > 3.0:
            gaps.append({
                'capability_type': CapabilityType.PERFORMANCE_OPTIMIZATION,
                'description': 'Slow response times',
                'current_value': response_time,
                'target_value': 2.0,
                'potential_improvement': (response_time - 2.0) / response_time
            })
        
        return gaps
    
    def _find_repositories_for_capability(self, capability_type: CapabilityType) -> List[RepositoryAnalysis]:
        """Find repositories that have a specific capability."""
        matching_repos = []
        
        for repo in self.discovered_repositories.values():
            if capability_type in repo.capabilities:
                matching_repos.append(repo)
        
        # Sort by compatibility and health
        matching_repos.sort(key=lambda r: (r.compatibility_score + r.health_score) / 2, reverse=True)
        
        return matching_repos
    
    def _calculate_recommendation_priority(self, gap: Dict[str, Any], repo: RepositoryAnalysis) -> float:
        """Calculate priority score for a capability recommendation."""
        # Higher priority for bigger gaps and better repositories
        gap_severity = gap['potential_improvement']
        repo_quality = (repo.health_score + repo.compatibility_score) / 2
        integration_ease = 1.0 - repo.integration_difficulty
        
        return gap_severity * 0.4 + repo_quality * 0.4 + integration_ease * 0.2
    
    def _update_discovery_state(self, repositories: List[RepositoryAnalysis]):
        """Update discovery state with new findings."""
        # Update discovered repositories
        for repo in repositories:
            self.discovered_repositories[repo.url] = repo
        
        # Update discovery history
        discovery_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'repositories_found': len(repositories),
            'total_repositories': len(self.discovered_repositories),
            'top_capabilities': self._get_top_capabilities(repositories)
        }
        
        self.discovery_history.append(discovery_record)
        
        # Keep only last 100 discovery records
        if len(self.discovery_history) > 100:
            self.discovery_history = self.discovery_history[-100:]
        
        self.last_discovery_time = datetime.now(timezone.utc)
        
        # Save state
        self._save_discovery_state()
    
    def _get_top_capabilities(self, repositories: List[RepositoryAnalysis]) -> Dict[str, int]:
        """Get count of top capabilities found."""
        capability_counts = defaultdict(int)
        
        for repo in repositories:
            for capability in repo.capabilities:
                capability_counts[capability.value] += 1
        
        return dict(capability_counts)
    
    def _load_discovery_state(self):
        """Load previous discovery state."""
        state_file = self.cache_dir / 'discovery_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Load discovered repositories
                for repo_data in state.get('repositories', []):
                    repo = RepositoryAnalysis(
                        url=repo_data['url'],
                        name=repo_data['name'],
                        description=repo_data['description'],
                        language=repo_data['language'],
                        stars=repo_data['stars'],
                        forks=repo_data['forks'],
                        last_updated=repo_data['last_updated'],
                        health_score=repo_data['health_score'],
                        capabilities=[CapabilityType(c) for c in repo_data['capabilities']],
                        architecture_patterns=repo_data['architecture_patterns'],
                        key_files=repo_data['key_files'],
                        integration_difficulty=repo_data['integration_difficulty'],
                        license=repo_data['license'],
                        documentation_quality=repo_data['documentation_quality'],
                        test_coverage=repo_data['test_coverage'],
                        performance_indicators=repo_data['performance_indicators'],
                        security_assessment=repo_data['security_assessment'],
                        compatibility_score=repo_data['compatibility_score'],
                        discovered_at=datetime.fromisoformat(repo_data['discovered_at'])
                    )
                    
                    self.discovered_repositories[repo.url] = repo
                
                # Load history
                self.discovery_history = state.get('history', [])
                
                last_time = state.get('last_discovery_time')
                if last_time:
                    self.last_discovery_time = datetime.fromisoformat(last_time)
                
            except Exception as e:
                self.logger.error(f"Error loading discovery state: {e}")
    
    def _save_discovery_state(self):
        """Save discovery state to disk."""
        state_file = self.cache_dir / 'discovery_state.json'
        
        try:
            # Prepare repository data
            repo_data = []
            for repo in self.discovered_repositories.values():
                repo_dict = {
                    'url': repo.url,
                    'name': repo.name,
                    'description': repo.description,
                    'language': repo.language,
                    'stars': repo.stars,
                    'forks': repo.forks,
                    'last_updated': repo.last_updated,
                    'health_score': repo.health_score,
                    'capabilities': [c.value for c in repo.capabilities],
                    'architecture_patterns': repo.architecture_patterns,
                    'key_files': repo.key_files,
                    'integration_difficulty': repo.integration_difficulty,
                    'license': repo.license,
                    'documentation_quality': repo.documentation_quality,
                    'test_coverage': repo.test_coverage,
                    'performance_indicators': repo.performance_indicators,
                    'security_assessment': repo.security_assessment,
                    'compatibility_score': repo.compatibility_score,
                    'discovered_at': repo.discovered_at.isoformat()
                }
                repo_data.append(repo_dict)
            
            state = {
                'repositories': repo_data,
                'history': self.discovery_history,
                'last_discovery_time': self.last_discovery_time.isoformat() if self.last_discovery_time else None,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving discovery state: {e}")
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        total_repos = len(self.discovered_repositories)
        
        if total_repos == 0:
            return {
                'total_repositories_discovered': 0,
                'discovery_runs': len(self.discovery_history),
                'last_discovery': None,
                'capabilities_found': {},
                'average_health_score': 0.0,
                'top_repositories': []
            }
        
        # Calculate statistics
        capabilities_count = defaultdict(int)
        health_scores = []
        
        for repo in self.discovered_repositories.values():
            health_scores.append(repo.health_score)
            for capability in repo.capabilities:
                capabilities_count[capability.value] += 1
        
        # Get top repositories
        top_repos = sorted(
            self.discovered_repositories.values(),
            key=self._calculate_repository_value,
            reverse=True
        )[:10]
        
        return {
            'total_repositories_discovered': total_repos,
            'discovery_runs': len(self.discovery_history),
            'last_discovery': self.last_discovery_time.isoformat() if self.last_discovery_time else None,
            'capabilities_found': dict(capabilities_count),
            'average_health_score': sum(health_scores) / len(health_scores),
            'top_repositories': [{
                'name': repo.name,
                'url': repo.url,
                'health_score': repo.health_score,
                'compatibility_score': repo.compatibility_score,
                'capabilities': [c.value for c in repo.capabilities]
            } for repo in top_repos]
        }


async def demonstrate_external_agent_discovery():
    """Demonstrate external agent discovery."""
    print("=== External Agent Discovery Demo ===\n")
    
    # Create discoverer
    config = DiscoveryConfig(max_repositories_per_scan=10)
    discoverer = ExternalAgentDiscoverer(config)
    
    # Discover trending agents
    print("Discovering trending AI agent repositories...")
    repositories = await discoverer.discover_trending_agents()
    
    print(f"\nDiscovered {len(repositories)} repositories:")
    for repo in repositories[:5]:
        print(f"\n{repo.name} ({repo.stars} stars)")
        print(f"  URL: {repo.url}")
        print(f"  Description: {repo.description[:100]}...")
        print(f"  Health Score: {repo.health_score:.2f}")
        print(f"  Compatibility: {repo.compatibility_score:.2f}")
        print(f"  Capabilities: {[c.value for c in repo.capabilities]}")
        print(f"  Architecture: {repo.architecture_patterns}")
    
    # Get capability recommendations
    print("\n=== Capability Recommendations ===")
    
    current_performance = {
        'task_completion_rate': 0.65,  # Low completion rate
        'error_rate': 0.15,            # High error rate
        'avg_response_time': 4.5       # Slow response
    }
    
    recommendations = await discoverer.get_capability_recommendations(current_performance)
    
    print(f"\nFound {len(recommendations)} recommendations:")
    for rec in recommendations[:3]:
        print(f"\n{rec['repository']}")
        print(f"  Addresses: {rec['addresses_gap']}")
        print(f"  Capability: {rec['capability_type'].value}")
        print(f"  Priority: {rec['priority']:.2f}")
        print(f"  Integration Difficulty: {rec['integration_difficulty']:.2f}")
        print(f"  Expected Improvement: {rec['expected_improvement']:.2f}")
    
    # Show statistics
    print("\n=== Discovery Statistics ===")
    stats = discoverer.get_discovery_statistics()
    
    print(f"Total repositories: {stats['total_repositories_discovered']}")
    print(f"Discovery runs: {stats['discovery_runs']}")
    print(f"Average health score: {stats['average_health_score']:.2f}")
    print(f"Capabilities found: {stats['capabilities_found']}")


if __name__ == "__main__":
    asyncio.run(demonstrate_external_agent_discovery())