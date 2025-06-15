"""
Optimized Task Generator - Performance-Enhanced Task Generation

This module provides optimized task generation with parallel processing,
smart caching, and batch operations to reduce AI API calls and latency.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import zlib

from intelligent_task_generator import IntelligentTaskGenerator
from smart_context_aggregator import SmartContextAggregator
from repository_analyzer import RepositoryAnalyzer


@dataclass
class BatchRequest:
    """Represents a batch of related AI requests."""
    request_id: str
    requests: List[Dict[str, Any]]
    request_type: str  # 'analysis', 'generation', 'validation'
    priority: int
    created_at: datetime


@dataclass
class CachedAnalysis:
    """Cached repository analysis with metadata."""
    repository: str
    analysis: Dict[str, Any]
    compressed_data: bytes
    hash: str
    timestamp: datetime
    access_count: int = 0


class OptimizedTaskGenerator:
    """Performance-optimized task generator with parallel processing and smart caching."""
    
    def __init__(self, base_generator: IntelligentTaskGenerator,
                 context_aggregator: SmartContextAggregator,
                 repository_analyzer: RepositoryAnalyzer,
                 redis_client=None):
        """Initialize optimized generator.
        
        Args:
            base_generator: Base intelligent task generator
            context_aggregator: Context aggregator
            repository_analyzer: Repository analyzer
            redis_client: Redis client for distributed caching
        """
        self.base_generator = base_generator
        self.context_aggregator = context_aggregator
        self.repository_analyzer = repository_analyzer
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Optimization settings
        self.enable_parallel_processing = True
        self.enable_batch_requests = True
        self.enable_compression = True
        self.enable_semantic_cache = True
        
        # Cache configuration
        self.analysis_cache: Dict[str, CachedAnalysis] = {}
        self.semantic_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(hours=6)
        self.max_cache_size = 100
        
        # Batch processing
        self.batch_queue: List[BatchRequest] = []
        self.batch_size_threshold = 5
        self.batch_time_threshold = timedelta(seconds=2)
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_requests': 0,
            'batch_requests': 0,
            'total_time_saved': 0.0
        }
    
    async def generate_tasks_optimized(self, repositories: List[str], 
                                     context: Dict[str, Any],
                                     count: int = 3) -> List[Dict[str, Any]]:
        """Generate tasks with optimized performance.
        
        Args:
            repositories: List of repositories to analyze
            context: System context
            count: Number of tasks to generate
            
        Returns:
            Generated tasks
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting optimized task generation for {len(repositories)} repositories")
        
        # Phase 1: Parallel repository analysis with caching
        analyses = await self._analyze_repositories_parallel(repositories)
        
        # Phase 2: Batch context enhancement
        enhanced_context = await self._enhance_context_batch(context, analyses)
        
        # Phase 3: Parallel task generation
        task_promises = []
        for repo, analysis in analyses.items():
            if analysis:
                task_promise = self._generate_repository_tasks(
                    repo, analysis, enhanced_context, count_per_repo=max(1, count // len(repositories))
                )
                task_promises.append(task_promise)
        
        # Execute task generation in parallel
        all_tasks = await asyncio.gather(*task_promises, return_exceptions=True)
        
        # Flatten and filter results
        final_tasks = []
        for tasks in all_tasks:
            if isinstance(tasks, list):
                final_tasks.extend(tasks)
            elif isinstance(tasks, Exception):
                self.logger.error(f"Task generation failed: {tasks}")
        
        # Phase 4: Batch validation and scoring
        if self.enable_batch_requests:
            final_tasks = await self._validate_tasks_batch(final_tasks)
        
        # Phase 5: Deduplicate and select top tasks
        unique_tasks = self._deduplicate_with_semantic_matching(final_tasks)
        selected_tasks = sorted(unique_tasks, 
                              key=lambda t: t.get('quality_score', 0), 
                              reverse=True)[:count]
        
        # Log performance metrics
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Optimized generation completed in {elapsed:.2f}s")
        self.logger.info(f"Performance metrics: {self.metrics}")
        
        return selected_tasks
    
    async def _analyze_repositories_parallel(self, repositories: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple repositories in parallel with caching.
        
        Args:
            repositories: List of repository names
            
        Returns:
            Repository analyses keyed by name
        """
        analyses = {}
        tasks = []
        
        for repo in repositories:
            # Check cache first
            cached = self._get_cached_analysis(repo)
            if cached:
                analyses[repo] = cached.analysis
                self.metrics['cache_hits'] += 1
                self.logger.debug(f"Cache hit for repository: {repo}")
            else:
                # Schedule for parallel analysis
                self.metrics['cache_misses'] += 1
                task = self._analyze_and_cache_repository(repo)
                tasks.append((repo, task))
        
        # Execute parallel analyses
        if tasks and self.enable_parallel_processing:
            self.logger.info(f"Analyzing {len(tasks)} repositories in parallel")
            self.metrics['parallel_requests'] += len(tasks)
            
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (repo, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to analyze {repo}: {result}")
                else:
                    analyses[repo] = result
        
        return analyses
    
    async def _analyze_and_cache_repository(self, repository: str) -> Dict[str, Any]:
        """Analyze repository and cache results.
        
        Args:
            repository: Repository name
            
        Returns:
            Repository analysis
        """
        # Perform analysis
        analysis = await self.repository_analyzer.analyze_repository(repository)
        
        # Compress and cache
        if self.enable_compression:
            compressed = zlib.compress(json.dumps(analysis).encode())
            cache_entry = CachedAnalysis(
                repository=repository,
                analysis=analysis,
                compressed_data=compressed,
                hash=hashlib.md5(compressed).hexdigest(),
                timestamp=datetime.now(timezone.utc)
            )
            self._store_cached_analysis(repository, cache_entry)
            
            # Also store in Redis if available
            if self.redis_client:
                await self._store_in_redis(f"repo_analysis:{repository}", cache_entry)
        
        return analysis
    
    def _get_cached_analysis(self, repository: str) -> Optional[CachedAnalysis]:
        """Get cached repository analysis.
        
        Args:
            repository: Repository name
            
        Returns:
            Cached analysis or None
        """
        # Check local cache
        if repository in self.analysis_cache:
            cached = self.analysis_cache[repository]
            if datetime.now(timezone.utc) - cached.timestamp < self.cache_ttl:
                cached.access_count += 1
                return cached
        
        # Check Redis if available
        if self.redis_client:
            # This would be async in production
            pass
        
        return None
    
    def _store_cached_analysis(self, repository: str, cache_entry: CachedAnalysis) -> None:
        """Store analysis in cache with size management.
        
        Args:
            repository: Repository name
            cache_entry: Cache entry to store
        """
        # Manage cache size
        if len(self.analysis_cache) >= self.max_cache_size:
            # Evict least recently accessed
            lru_repo = min(self.analysis_cache.items(), 
                          key=lambda x: x[1].access_count)[0]
            del self.analysis_cache[lru_repo]
        
        self.analysis_cache[repository] = cache_entry
    
    async def _enhance_context_batch(self, base_context: Dict[str, Any], 
                                   analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance context with batch processing.
        
        Args:
            base_context: Base context
            analyses: Repository analyses
            
        Returns:
            Enhanced context
        """
        if not self.enable_batch_requests:
            return await self.context_aggregator.gather_comprehensive_context()
        
        # Prepare batch request for context enhancement
        batch_data = {
            'base_context': base_context,
            'repository_count': len(analyses),
            'health_scores': {repo: a.get('health_metrics', {}).get('health_score', 0) 
                            for repo, a in analyses.items()},
            'tech_stacks': {repo: a.get('technical_stack', {}) 
                          for repo, a in analyses.items()}
        }
        
        # Single AI call for comprehensive context
        prompt = f"""
        Analyze the system state and provide enhanced context insights:
        
        System Overview:
        - Active Repositories: {len(analyses)}
        - Average Health: {sum(batch_data['health_scores'].values()) / len(analyses) if analyses else 0:.1%}
        - Technology Distribution: {self._summarize_tech_distribution(batch_data['tech_stacks'])}
        
        Provide:
        1. Cross-repository patterns
        2. System-wide priorities
        3. Market alignment opportunities
        4. Risk areas requiring attention
        
        Return as structured JSON with insights.
        """
        
        response = await self.base_generator.ai_brain.generate_enhanced_response(prompt)
        insights = self._parse_json_response(response)
        
        # Merge with base context
        enhanced = base_context.copy()
        enhanced.update(insights)
        enhanced['repository_analyses'] = analyses
        
        return enhanced
    
    async def _generate_repository_tasks(self, repository: str, 
                                       analysis: Dict[str, Any],
                                       context: Dict[str, Any],
                                       count_per_repo: int) -> List[Dict[str, Any]]:
        """Generate tasks for a single repository.
        
        Args:
            repository: Repository name
            analysis: Repository analysis
            context: Enhanced context
            count_per_repo: Tasks to generate per repository
            
        Returns:
            Generated tasks
        """
        tasks = []
        
        # Use base generator with optimized parameters
        for _ in range(count_per_repo):
            task = await self.base_generator.generate_task_for_repository(
                repository, analysis, context
            )
            if task and not task.get('skip'):
                tasks.append(task)
        
        return tasks
    
    async def _validate_tasks_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and score tasks in batch.
        
        Args:
            tasks: Tasks to validate
            
        Returns:
            Validated tasks with scores
        """
        if not tasks:
            return tasks
        
        # Prepare batch validation request
        task_summaries = [
            {
                'id': i,
                'title': t.get('title', ''),
                'type': t.get('type', ''),
                'repository': t.get('repository', ''),
                'description': t.get('description', '')[:200]  # Truncate for efficiency
            }
            for i, t in enumerate(tasks)
        ]
        
        prompt = f"""
        Validate and score these tasks for quality and value:
        
        Tasks:
        {json.dumps(task_summaries, indent=2)}
        
        For each task, provide:
        - quality_score: 0-1 based on clarity, feasibility, and value
        - priority_adjustment: any priority changes needed
        - duplicate_risk: likelihood this duplicates existing work
        - improvement_suggestion: brief suggestion if score < 0.7
        
        Return as JSON array with task IDs.
        """
        
        response = await self.base_generator.ai_brain.generate_enhanced_response(prompt)
        validations = self._parse_json_response(response)
        
        # Apply validation results
        if isinstance(validations, list):
            for validation in validations:
                task_id = validation.get('id')
                if task_id is not None and task_id < len(tasks):
                    tasks[task_id]['quality_score'] = validation.get('quality_score', 0.5)
                    if 'priority_adjustment' in validation:
                        tasks[task_id]['priority'] = validation['priority_adjustment']
        
        self.metrics['batch_requests'] += 1
        return tasks
    
    def _deduplicate_with_semantic_matching(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate tasks using semantic matching.
        
        Args:
            tasks: Tasks to deduplicate
            
        Returns:
            Unique tasks
        """
        if not self.enable_semantic_cache:
            return tasks
        
        unique_tasks = []
        seen_semantics = []
        
        for task in tasks:
            # Create semantic signature
            signature = self._create_semantic_signature(task)
            
            # Check against seen signatures
            is_duplicate = False
            for seen_sig in seen_semantics:
                if self._semantic_similarity(signature, seen_sig) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tasks.append(task)
                seen_semantics.append(signature)
        
        self.logger.info(f"Deduplicated {len(tasks)} tasks to {len(unique_tasks)} unique tasks")
        return unique_tasks
    
    def _create_semantic_signature(self, task: Dict[str, Any]) -> str:
        """Create semantic signature for task.
        
        Args:
            task: Task to signature
            
        Returns:
            Semantic signature
        """
        # Combine key elements
        elements = [
            task.get('type', ''),
            task.get('repository', ''),
            ' '.join(task.get('title', '').lower().split()[:5]),  # First 5 words
            task.get('priority', '')
        ]
        
        return '|'.join(elements)
    
    def _semantic_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate semantic similarity between signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Similarity score (0-1)
        """
        parts1 = sig1.split('|')
        parts2 = sig2.split('|')
        
        if len(parts1) != len(parts2):
            return 0.0
        
        # Weight different parts
        weights = [0.3, 0.2, 0.4, 0.1]  # type, repo, title, priority
        similarity = 0.0
        
        for i, (p1, p2) in enumerate(zip(parts1, parts2)):
            if i < len(weights):
                if p1 == p2:
                    similarity += weights[i]
                elif i == 2:  # Title - use word overlap
                    words1 = set(p1.split())
                    words2 = set(p2.split())
                    if words1 and words2:
                        overlap = len(words1 & words2) / len(words1 | words2)
                        similarity += weights[i] * overlap
        
        return similarity
    
    def _summarize_tech_distribution(self, tech_stacks: Dict[str, Dict[str, Any]]) -> str:
        """Summarize technology distribution across repositories.
        
        Args:
            tech_stacks: Technology stacks by repository
            
        Returns:
            Summary string
        """
        all_techs = []
        for stack in tech_stacks.values():
            for category, techs in stack.items():
                if isinstance(techs, list):
                    all_techs.extend(techs)
        
        if not all_techs:
            return "No technology data"
        
        # Count occurrences
        tech_counts = defaultdict(int)
        for tech in all_techs:
            tech_counts[tech] += 1
        
        # Get top technologies
        top_techs = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return ', '.join([f"{tech}({count})" for tech, count in top_techs])
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed data
        """
        content = response.get('content', '')
        try:
            import re
            json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON: {e}")
        return {}
    
    async def _store_in_redis(self, key: str, data: Any) -> None:
        """Store data in Redis (placeholder for actual implementation).
        
        Args:
            key: Redis key
            data: Data to store
        """
        # This would use actual Redis client in production
        pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report.
        
        Returns:
            Performance metrics and recommendations
        """
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = self.metrics['cache_hits'] / total_requests if total_requests > 0 else 0
        
        report = {
            'metrics': self.metrics,
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'optimizations_enabled': {
                'parallel_processing': self.enable_parallel_processing,
                'batch_requests': self.enable_batch_requests,
                'compression': self.enable_compression,
                'semantic_cache': self.enable_semantic_cache
            },
            'cache_status': {
                'entries': len(self.analysis_cache),
                'max_size': self.max_cache_size,
                'ttl_hours': self.cache_ttl.total_seconds() / 3600
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if cache_hit_rate < 0.5:
            report['recommendations'].append("Consider increasing cache TTL for better hit rate")
        
        if self.metrics['parallel_requests'] == 0:
            report['recommendations'].append("Enable parallel processing for better performance")
        
        if self.metrics['batch_requests'] < 5:
            report['recommendations'].append("Increase batch processing to reduce API calls")
        
        return report