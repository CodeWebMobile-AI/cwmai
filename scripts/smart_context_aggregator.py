"""
Smart Context Aggregator - Unified Intelligence Gathering

This module provides comprehensive context aggregation from multiple sources
to enable intelligent task generation with full awareness of the system state.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import aiohttp
from pathlib import Path

from repository_analyzer import RepositoryAnalyzer
from repository_exclusion import should_process_repo


@dataclass
class ContextSource:
    """Represents a context data source."""
    name: str
    type: str  # 'repository', 'external', 'system', 'historical'
    priority: int  # 1-10, higher = more important
    data: Dict[str, Any]
    timestamp: datetime
    quality_score: float  # 0-1, data quality/reliability
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedContext:
    """Comprehensive context from all sources."""
    repository_health: Dict[str, float]
    technology_distribution: Dict[str, List[str]]
    recent_activities: List[Dict[str, Any]]
    market_insights: List[Dict[str, Any]]
    system_capabilities: Dict[str, float]
    resource_availability: Dict[str, Any]
    strategic_priorities: List[str]
    performance_metrics: Dict[str, float]
    dependency_graph: Dict[str, List[str]]
    cross_repo_patterns: List[Dict[str, Any]]
    external_signals: List[Dict[str, Any]]
    historical_patterns: Dict[str, Any]
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


class SmartContextAggregator:
    """Aggregates context from multiple sources intelligently."""
    
    def __init__(self, repository_analyzer: Optional[RepositoryAnalyzer] = None,
                 ai_brain=None, redis_client=None):
        """Initialize the context aggregator.
        
        Args:
            repository_analyzer: Repository analysis tool
            ai_brain: AI brain for intelligent analysis
            redis_client: Redis client for caching
        """
        self.repository_analyzer = repository_analyzer or RepositoryAnalyzer()
        self.ai_brain = ai_brain
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Context sources registry
        self.context_sources: Dict[str, ContextSource] = {}
        self.source_weights = {
            'repository': 0.3,
            'external': 0.2,
            'system': 0.25,
            'historical': 0.25
        }
        
        # Caching
        self.context_cache: Dict[str, AggregatedContext] = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Quality thresholds
        self.min_quality_threshold = 0.6
        self.staleness_threshold = timedelta(hours=24)
        
    async def gather_comprehensive_context(self, 
                                         scope: str = "full",
                                         force_refresh: bool = False) -> AggregatedContext:
        """Gather comprehensive context from all sources.
        
        Args:
            scope: Context scope ('full', 'repositories', 'external')
            force_refresh: Force refresh bypassing cache
            
        Returns:
            Aggregated context
        """
        cache_key = f"context_{scope}"
        
        # Check cache unless forced refresh
        if not force_refresh and cache_key in self.context_cache:
            cached = self.context_cache[cache_key]
            if datetime.now(timezone.utc) - cached.timestamp < self.cache_duration:
                self.logger.info(f"Returning cached context for scope: {scope}")
                return cached
        
        self.logger.info(f"Gathering comprehensive context for scope: {scope}")
        
        # Gather from different sources in parallel
        tasks = []
        
        if scope in ["full", "repositories"]:
            tasks.append(self._gather_repository_context())
        
        if scope in ["full", "external"]:
            tasks.append(self._gather_external_context())
            
        if scope in ["full", "system"]:
            tasks.append(self._gather_system_context())
            
        if scope in ["full", "historical"]:
            tasks.append(self._gather_historical_context())
        
        # Execute gathering tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Context gathering task {i} failed: {result}")
        
        # Aggregate all contexts
        aggregated = await self._aggregate_contexts()
        
        # Cache the result
        self.context_cache[cache_key] = aggregated
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_context_in_redis(cache_key, aggregated)
        
        return aggregated
    
    async def _gather_repository_context(self) -> None:
        """Gather context from repositories."""
        self.logger.info("Gathering repository context")
        
        try:
            # Get list of repositories
            repos_data = await self._get_active_repositories()
            
            repo_health = {}
            tech_distribution = defaultdict(list)
            recent_activities = []
            cross_repo_patterns = []
            
            # Analyze each repository
            for repo in repos_data:
                repo_name = repo.get('full_name', repo.get('name', ''))
                
                if not should_process_repo(repo_name):
                    continue
                
                # Get repository analysis
                analysis = await self.repository_analyzer.analyze_repository(repo_name)
                
                # Extract health score
                health = analysis.get('health_metrics', {}).get('health_score', 0) / 100.0
                repo_health[repo_name] = health
                
                # Extract technology stack
                tech_stack = analysis.get('technical_stack', {})
                for category, techs in tech_stack.items():
                    if isinstance(techs, list):
                        tech_distribution[repo_name].extend(techs)
                
                # Extract recent activities
                recent = analysis.get('recent_activity', {})
                if recent:
                    recent_activities.append({
                        'repository': repo_name,
                        'last_commit': recent.get('last_commit_date'),
                        'active_contributors': recent.get('active_contributors', 0),
                        'commit_frequency': recent.get('commit_frequency', 'low')
                    })
            
            # Identify cross-repository patterns
            cross_repo_patterns = self._identify_cross_repo_patterns(
                repo_health, tech_distribution
            )
            
            # Create context source
            self.context_sources['repository'] = ContextSource(
                name='repository',
                type='repository',
                priority=9,
                data={
                    'health_scores': repo_health,
                    'technology_distribution': dict(tech_distribution),
                    'recent_activities': recent_activities,
                    'cross_repo_patterns': cross_repo_patterns
                },
                timestamp=datetime.now(timezone.utc),
                quality_score=0.9
            )
            
        except Exception as e:
            self.logger.error(f"Failed to gather repository context: {e}")
            self.context_sources['repository'] = ContextSource(
                name='repository',
                type='repository',
                priority=9,
                data={},
                timestamp=datetime.now(timezone.utc),
                quality_score=0.0
            )
    
    async def _gather_external_context(self) -> None:
        """Gather context from external sources."""
        self.logger.info("Gathering external context")
        
        market_insights = []
        external_signals = []
        
        try:
            # Gather market trends (simulated for now)
            market_insights = await self._fetch_market_trends()
            
            # Gather security advisories
            security_signals = await self._fetch_security_advisories()
            external_signals.extend(security_signals)
            
            # Gather technology trends
            tech_trends = await self._fetch_technology_trends()
            external_signals.extend(tech_trends)
            
            quality_score = 0.7  # External data is less reliable
            
        except Exception as e:
            self.logger.error(f"Failed to gather external context: {e}")
            quality_score = 0.0
        
        self.context_sources['external'] = ContextSource(
            name='external',
            type='external',
            priority=7,
            data={
                'market_insights': market_insights,
                'external_signals': external_signals
            },
            timestamp=datetime.now(timezone.utc),
            quality_score=quality_score
        )
    
    async def _gather_system_context(self) -> None:
        """Gather context from system state."""
        self.logger.info("Gathering system context")
        
        try:
            # Load system state
            system_state = await self._load_system_state()
            
            # Extract capabilities
            capabilities = {}
            if 'capabilities' in system_state:
                for cap in system_state['capabilities']:
                    if isinstance(cap, dict):
                        capabilities[cap.get('name', 'unknown')] = cap.get('score', 0.5)
                    else:
                        capabilities[str(cap)] = 0.7
            
            # Extract resource availability
            resources = {
                'ai_capacity': system_state.get('ai_capacity', 1.0),
                'processing_slots': system_state.get('processing_slots', 5),
                'memory_available': system_state.get('memory_available', True),
                'rate_limits': system_state.get('rate_limits', {})
            }
            
            # Extract performance metrics
            performance = {
                'task_completion_rate': system_state.get('completion_rate', 0.8),
                'average_task_time': system_state.get('avg_task_time', 4.0),
                'success_rate': system_state.get('success_rate', 0.85),
                'error_rate': system_state.get('error_rate', 0.05)
            }
            
            self.context_sources['system'] = ContextSource(
                name='system',
                type='system',
                priority=8,
                data={
                    'capabilities': capabilities,
                    'resources': resources,
                    'performance': performance,
                    'state': system_state
                },
                timestamp=datetime.now(timezone.utc),
                quality_score=0.95
            )
            
        except Exception as e:
            self.logger.error(f"Failed to gather system context: {e}")
            self.context_sources['system'] = ContextSource(
                name='system',
                type='system',
                priority=8,
                data={},
                timestamp=datetime.now(timezone.utc),
                quality_score=0.0
            )
    
    async def _gather_historical_context(self) -> None:
        """Gather historical patterns and learnings."""
        self.logger.info("Gathering historical context")
        
        try:
            # Load task history
            task_history = await self._load_task_history()
            
            # Analyze patterns
            patterns = self._analyze_historical_patterns(task_history)
            
            # Extract success/failure patterns
            success_patterns = patterns.get('success_patterns', {})
            failure_patterns = patterns.get('failure_patterns', {})
            
            # Time-based patterns
            time_patterns = self._analyze_time_patterns(task_history)
            
            self.context_sources['historical'] = ContextSource(
                name='historical',
                type='historical',
                priority=6,
                data={
                    'patterns': patterns,
                    'success_patterns': success_patterns,
                    'failure_patterns': failure_patterns,
                    'time_patterns': time_patterns,
                    'task_history_size': len(task_history)
                },
                timestamp=datetime.now(timezone.utc),
                quality_score=0.85
            )
            
        except Exception as e:
            self.logger.error(f"Failed to gather historical context: {e}")
            self.context_sources['historical'] = ContextSource(
                name='historical',
                type='historical',
                priority=6,
                data={},
                timestamp=datetime.now(timezone.utc),
                quality_score=0.0
            )
    
    async def _aggregate_contexts(self) -> AggregatedContext:
        """Aggregate all context sources into unified context."""
        self.logger.info("Aggregating contexts from all sources")
        
        # Initialize aggregated data
        repo_health = {}
        tech_distribution = {}
        recent_activities = []
        market_insights = []
        capabilities = {}
        resources = {}
        priorities = []
        performance = {}
        dependency_graph = {}
        cross_patterns = []
        external_signals = []
        historical = {}
        
        # Process each context source
        for source_name, source in self.context_sources.items():
            if source.quality_score < self.min_quality_threshold:
                self.logger.warning(f"Skipping low-quality source: {source_name}")
                continue
            
            data = source.data
            
            if source.type == 'repository':
                repo_health.update(data.get('health_scores', {}))
                tech_distribution.update(data.get('technology_distribution', {}))
                recent_activities.extend(data.get('recent_activities', []))
                cross_patterns.extend(data.get('cross_repo_patterns', []))
                
            elif source.type == 'external':
                market_insights.extend(data.get('market_insights', []))
                external_signals.extend(data.get('external_signals', []))
                
            elif source.type == 'system':
                capabilities.update(data.get('capabilities', {}))
                resources.update(data.get('resources', {}))
                performance.update(data.get('performance', {}))
                
            elif source.type == 'historical':
                historical.update(data.get('patterns', {}))
        
        # Calculate strategic priorities based on all data
        priorities = await self._calculate_strategic_priorities(
            repo_health, market_insights, historical
        )
        
        # Build dependency graph
        dependency_graph = await self._build_dependency_graph(
            tech_distribution, cross_patterns
        )
        
        # Calculate overall quality score
        quality_scores = [s.quality_score for s in self.context_sources.values()]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return AggregatedContext(
            repository_health=repo_health,
            technology_distribution=tech_distribution,
            recent_activities=recent_activities,
            market_insights=market_insights,
            system_capabilities=capabilities,
            resource_availability=resources,
            strategic_priorities=priorities,
            performance_metrics=performance,
            dependency_graph=dependency_graph,
            cross_repo_patterns=cross_patterns,
            external_signals=external_signals,
            historical_patterns=historical,
            quality_score=overall_quality,
            timestamp=datetime.now(timezone.utc),
            metadata={
                'sources': list(self.context_sources.keys()),
                'aggregation_method': 'weighted_merge'
            }
        )
    
    def _identify_cross_repo_patterns(self, health_scores: Dict[str, float],
                                    tech_distribution: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Identify patterns across repositories."""
        patterns = []
        
        # Find repositories with similar tech stacks
        tech_groups = defaultdict(list)
        for repo, techs in tech_distribution.items():
            tech_key = ','.join(sorted(techs[:3]))  # Top 3 techs as key
            tech_groups[tech_key].append(repo)
        
        for tech_key, repos in tech_groups.items():
            if len(repos) > 1:
                patterns.append({
                    'pattern': 'shared_technology',
                    'repositories': repos,
                    'technology': tech_key,
                    'confidence': 0.8,
                    'opportunity': 'Create shared components'
                })
        
        # Find repositories with similar health issues
        low_health_repos = [r for r, h in health_scores.items() if h < 0.6]
        if len(low_health_repos) > 1:
            patterns.append({
                'pattern': 'collective_health_issue',
                'repositories': low_health_repos,
                'confidence': 0.9,
                'opportunity': 'Systematic improvement needed'
            })
        
        return patterns
    
    async def _calculate_strategic_priorities(self, health_scores: Dict[str, float],
                                            market_insights: List[Dict[str, Any]],
                                            historical: Dict[str, Any]) -> List[str]:
        """Calculate strategic priorities from context."""
        priorities = []
        
        # Health-based priorities
        avg_health = sum(health_scores.values()) / len(health_scores) if health_scores else 0
        if avg_health < 0.7:
            priorities.append("Improve overall repository health")
        
        # Market-based priorities
        for insight in market_insights[:3]:
            if insight.get('importance', 0) > 0.7:
                priorities.append(f"Address market trend: {insight.get('trend', 'Unknown')}")
        
        # Historical success patterns
        success_patterns = historical.get('success_patterns', {})
        for pattern, success_rate in success_patterns.items():
            if success_rate > 0.8:
                priorities.append(f"Leverage successful pattern: {pattern}")
        
        return priorities[:5]  # Top 5 priorities
    
    async def _build_dependency_graph(self, tech_distribution: Dict[str, List[str]],
                                    cross_patterns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build dependency graph between repositories."""
        graph = defaultdict(list)
        
        # Technology-based dependencies
        for repo1, techs1 in tech_distribution.items():
            for repo2, techs2 in tech_distribution.items():
                if repo1 != repo2:
                    shared_techs = set(techs1) & set(techs2)
                    if len(shared_techs) > 2:
                        graph[repo1].append(repo2)
        
        # Pattern-based dependencies
        for pattern in cross_patterns:
            if pattern.get('pattern') == 'shared_technology':
                repos = pattern.get('repositories', [])
                for i, repo1 in enumerate(repos):
                    for repo2 in repos[i+1:]:
                        if repo2 not in graph[repo1]:
                            graph[repo1].append(repo2)
        
        return dict(graph)
    
    async def _get_active_repositories(self) -> List[Dict[str, Any]]:
        """Get list of active repositories."""
        # This would connect to your repository source
        # For now, returning mock data structure
        return []
    
    async def _fetch_market_trends(self) -> List[Dict[str, Any]]:
        """Fetch current market trends."""
        # In production, this would call external APIs
        return [
            {
                'trend': 'AI Integration',
                'importance': 0.9,
                'growth_rate': 0.3,
                'keywords': ['ai', 'ml', 'llm', 'automation']
            },
            {
                'trend': 'Security Focus',
                'importance': 0.85,
                'growth_rate': 0.2,
                'keywords': ['security', 'auth', 'encryption', '2fa']
            }
        ]
    
    async def _fetch_security_advisories(self) -> List[Dict[str, Any]]:
        """Fetch security advisories."""
        return [
            {
                'type': 'security_advisory',
                'severity': 'medium',
                'affects': ['auth', 'jwt'],
                'action': 'Update authentication libraries'
            }
        ]
    
    async def _fetch_technology_trends(self) -> List[Dict[str, Any]]:
        """Fetch technology trends."""
        return [
            {
                'type': 'tech_trend',
                'technology': 'React Server Components',
                'adoption_rate': 0.4,
                'recommendation': 'Consider for new projects'
            }
        ]
    
    async def _load_system_state(self) -> Dict[str, Any]:
        """Load current system state."""
        try:
            state_file = Path('system_state.json')
            if state_file.exists():
                with open(state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
        
        return {}
    
    async def _load_task_history(self) -> List[Dict[str, Any]]:
        """Load task history."""
        try:
            history_file = Path('task_history.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.error(f"Failed to load task history: {e}")
        
        return []
    
    def _analyze_historical_patterns(self, task_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in task history."""
        patterns = {
            'task_type_distribution': Counter(),
            'success_patterns': {},
            'failure_patterns': {},
            'completion_times': defaultdict(list),
            'complexity_trends': []
        }
        
        for task in task_history:
            task_type = task.get('type', 'unknown')
            patterns['task_type_distribution'][task_type] += 1
            
            if task.get('status') == 'completed':
                success_key = f"{task_type}_success"
                patterns['success_patterns'][success_key] = patterns['success_patterns'].get(success_key, 0) + 1
            elif task.get('status') == 'failed':
                failure_key = f"{task_type}_failure"
                patterns['failure_patterns'][failure_key] = patterns['failure_patterns'].get(failure_key, 0) + 1
            
            if 'completion_time' in task:
                patterns['completion_times'][task_type].append(task['completion_time'])
        
        return patterns
    
    def _analyze_time_patterns(self, task_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time-based patterns."""
        time_patterns = {
            'peak_hours': Counter(),
            'day_of_week': Counter(),
            'monthly_trends': defaultdict(int)
        }
        
        for task in task_history:
            if 'created_at' in task:
                try:
                    created = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
                    time_patterns['peak_hours'][created.hour] += 1
                    time_patterns['day_of_week'][created.strftime('%A')] += 1
                    time_patterns['monthly_trends'][created.strftime('%Y-%m')] += 1
                except Exception:
                    pass
        
        return time_patterns
    
    async def _store_context_in_redis(self, key: str, context: AggregatedContext) -> None:
        """Store context in Redis for distributed access."""
        if not self.redis_client:
            return
        
        try:
            # Serialize context
            context_data = {
                'repository_health': context.repository_health,
                'technology_distribution': context.technology_distribution,
                'recent_activities': context.recent_activities,
                'market_insights': context.market_insights,
                'quality_score': context.quality_score,
                'timestamp': context.timestamp.isoformat()
            }
            
            await self.redis_client.setex(
                f"context:{key}",
                int(self.cache_duration.total_seconds()),
                json.dumps(context_data)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store context in Redis: {e}")
    
    async def get_context_quality_report(self) -> Dict[str, Any]:
        """Get quality report for current context."""
        report = {
            'sources': {},
            'overall_quality': 0.0,
            'staleness': {},
            'coverage': {},
            'recommendations': []
        }
        
        now = datetime.now(timezone.utc)
        
        for source_name, source in self.context_sources.items():
            age = now - source.timestamp
            is_stale = age > self.staleness_threshold
            
            report['sources'][source_name] = {
                'quality_score': source.quality_score,
                'age_hours': age.total_seconds() / 3600,
                'is_stale': is_stale,
                'priority': source.priority
            }
            
            if is_stale:
                report['recommendations'].append(f"Refresh {source_name} context (stale)")
            
            if source.quality_score < self.min_quality_threshold:
                report['recommendations'].append(f"Improve {source_name} data quality")
        
        # Calculate overall quality
        if self.context_sources:
            quality_scores = [s.quality_score for s in self.context_sources.values()]
            report['overall_quality'] = sum(quality_scores) / len(quality_scores)
        
        # Coverage analysis
        expected_sources = {'repository', 'external', 'system', 'historical'}
        actual_sources = set(self.context_sources.keys())
        missing_sources = expected_sources - actual_sources
        
        report['coverage'] = {
            'expected': list(expected_sources),
            'actual': list(actual_sources),
            'missing': list(missing_sources),
            'percentage': len(actual_sources) / len(expected_sources) * 100
        }
        
        if missing_sources:
            report['recommendations'].append(f"Add missing sources: {', '.join(missing_sources)}")
        
        return report
    
    async def update_context_source(self, source_name: str, 
                                  data: Dict[str, Any],
                                  quality_score: float = 0.8) -> None:
        """Update a specific context source."""
        source_type = 'custom'
        if source_name in ['repository', 'external', 'system', 'historical']:
            source_type = source_name
        
        self.context_sources[source_name] = ContextSource(
            name=source_name,
            type=source_type,
            priority=5,
            data=data,
            timestamp=datetime.now(timezone.utc),
            quality_score=quality_score
        )
        
        # Invalidate cache
        self.context_cache.clear()
        
        self.logger.info(f"Updated context source: {source_name}")